import argparse
import os
from typing import Dict, List
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np


from torch.profiler import profile, record_function, ProfilerActivity

from pydantic_core import from_json

# local imports
import datasets
from training import GracefulKiller
from noise_scheduler import NoiseScheduler
from config import ProgressiveDistillationExperimentConfig, DeviceEnum, ModelTypeEnum
from model import get_model, MLP, MLPSPS, AnyModel
from metric import metric_nearest_distance

from ddpm import eval_iteration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    arguments = config = parser.parse_args()

    with open(arguments.config, 'r') as f:
        config_json_data = f.read()

    experiment_config = ProgressiveDistillationExperimentConfig.model_validate_json(config_json_data)

    dataset = datasets.get_dataset(experiment_config.dataset, n=100000)
    dataset_frame_numpy: np.ndarray = np.vstack([ t.numpy() for t in dataset.tensors ])
    dataloader = DataLoader(
        dataset, batch_size=experiment_config.train_batch_size, shuffle=True, drop_last=True
    )

    device: DeviceEnum = experiment_config.nn_device # type: ignore
    teacher_model = get_model(experiment_config)
    teacher_model = teacher_model.to(device)

    teacher_state_dict = torch.load(experiment_config.teacher_checkpoint)
    teacher_model.load_state_dict(teacher_state_dict)

    class NoopKiller:
        kill_now = False
    killer = NoopKiller()
    # killer = GracefulKiller()

    global_step = 0
    frames = []
    losses = []
    metric_max = []
    metric_sum = []
    metric_mean = []


    for distillation_step in range(0, experiment_config.distillation_steps):

        student_model = get_model(experiment_config)
        student_model.load_state_dict(teacher_model.state_dict())

        # todo change model timesteps count
        distillation_factor = experiment_config.distillation_factor
        current_num_timesteps = int(experiment_config.num_timesteps / (distillation_factor**distillation_step))
        print("current_num_timesteps", current_num_timesteps)
        teacher_noise_scheduler = NoiseScheduler(
            num_timesteps=current_num_timesteps,
            beta_schedule=experiment_config.beta_schedule,
            device=device,
        )
        student_noise_scheduler = NoiseScheduler(
            num_timesteps=int(current_num_timesteps / distillation_factor),
            beta_schedule=experiment_config.beta_schedule,
            device=device,
        )

        optimizer = torch.optim.AdamW(
            student_model.parameters(),
            lr=experiment_config.learning_rate,
        )

        with torch.no_grad():
            frame = eval_iteration(experiment_config, student_model, student_noise_scheduler, device=device, epoch=distillation_step, prefix=f'student_no_training_')
            metrics_value = metric_nearest_distance(frame, dataset_frame_numpy)
            print("metrics_value", metrics_value)
            eval_iteration(experiment_config, teacher_model, teacher_noise_scheduler, device=device, epoch=distillation_step, prefix=f'teacher_')

        student_model.train()
        teacher_model.eval()

        for epoch in range(experiment_config.num_epochs):
            if killer.kill_now:
                break

            student_model.train()
            progress_bar = tqdm(total=len(dataloader))
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(dataloader):
                if killer.kill_now:
                    break

                batch = batch[0].to(device)

                if experiment_config.nn_model_type == ModelTypeEnum.mlp_sps:
                    timesteps = torch.full([ batch.shape[0] ], random.randint(2, teacher_noise_scheduler.num_timesteps - 1))
                else:
                    timesteps = torch.randint(
                        2, teacher_noise_scheduler.num_timesteps, (batch.shape[0],),
                        device=device,
                    ).long()

                # для учителя нам нужны только четные таймстемпы, тк именно с них будет начинаться
                # и обучаться ученик
                timesteps = timesteps - torch.remainder(timesteps, 2)
                timesteps_next = timesteps - 1

                noise = torch.randn(batch.shape, device=device)
                noisy = teacher_noise_scheduler.add_noise(batch, noise, timesteps)

                sigma_t = teacher_noise_scheduler.sqrt_one_minus_alphas_cumprod[timesteps].unsqueeze(1)
                sigma_tss = teacher_noise_scheduler.sqrt_one_minus_alphas_cumprod[timesteps_next].unsqueeze(1)
                alpha_t = teacher_noise_scheduler.sqrt_alphas_cumprod[timesteps].unsqueeze(1)
                alpha_tss = teacher_noise_scheduler.sqrt_alphas_cumprod[timesteps_next].unsqueeze(1)

                # 2 step of teacher model
                with torch.no_grad():
                    if experiment_config.nn_model_type == ModelTypeEnum.mlp_sps:
                        # model: MLPSPS
                        teacher_noise_pred1 = teacher_model.forward_single_timestep(noisy, timesteps)
                    else:
                        teacher_noise_pred1 = teacher_model(noisy, timesteps)

                    empty_noize = torch.zeros_like(teacher_noise_pred1)

                    teacher_sample1 = teacher_noise_scheduler.step(teacher_noise_pred1, timesteps, noisy, noise=empty_noize)
                    # teacher_sample1 = alpha_t * teacher_noise_pred1 +

                    if experiment_config.nn_model_type == ModelTypeEnum.mlp_sps:
                        # model: MLPSPS
                        teacher_noise_pred2 = teacher_model.forward_single_timestep(teacher_sample1, timesteps_next)
                    else:
                        teacher_noise_pred2 = teacher_model(teacher_sample1, timesteps_next)

                    teacher_sample2 = teacher_noise_scheduler.step(teacher_noise_pred2, timesteps_next, teacher_sample1, noise=empty_noize)

                if experiment_config.nn_model_type == ModelTypeEnum.mlp_sps:
                    # model: MLPSPS
                    student_noise_pred = student_model.forward_single_timestep(noisy, timesteps)
                else:
                    student_noise_pred = student_model(noisy, timesteps)

                # из статьи
                # teacher_noise_total = (teacher_sample2 - (sigma_tss / sigma_t ) * noisy) / ( alpha_tss - (sigma_tss / sigma_t) * alpha_t )
                # без делителя
                # teacher_noise_total = (teacher_sample2 - (sigma_tss / sigma_t ) * noisy) # / ( alpha_tss - (sigma_tss / sigma_t) * alpha_t )
                # просто разница -- плохо работает
                # teacher_noise_total = teacher_sample2 - noisy

                # teacher_noise_total = (teacher_noise_pred2 - (sigma_tss / sigma_t ) * teacher_noise_pred1) / ( alpha_tss - (sigma_tss / sigma_t) * alpha_t )

                # самодельный просто сумма с двух предыдущих шагов -- не очень плохо
                teacher_noise_total = teacher_noise_pred2 + teacher_noise_pred1

                # snr = alpha_t ** 2 / sigma_t ** 2
                # loss_weight = torch.masked_fill(snr, snr > 1, 1)
                loss = F.mse_loss(student_noise_pred, teacher_noise_total, reduction='none')
                loss = loss.mean()

                assert loss.item() < 10

                loss.backward(loss)

                nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "step": global_step}
                losses.append(loss.detach().item())
                progress_bar.set_postfix(logs)
                global_step += 1

            progress_bar.close()

            if epoch % experiment_config.save_images_step == 0 or epoch == experiment_config.num_epochs - 1:
                with torch.no_grad():
                    # generate data with the model to later visualize the learning process

                    student_num_timesteps = int(current_num_timesteps / distillation_factor)
                    frame = eval_iteration(experiment_config, student_model, student_noise_scheduler,  device=device, epoch=epoch, prefix=f'student_{student_num_timesteps}_')
                    frames.append(frame)

                    metrics_value = metric_nearest_distance(frame, dataset_frame_numpy)
                    print("metrics_value", metrics_value)
                    metric_max.append(metrics_value.value_max)
                    metric_sum.append(metrics_value.value_sum)
                    metric_mean.append(metrics_value.value_mean)
                    if metrics_value.value_mean > 1.0:
                        raise Exception("too bad metrics")
            # Epoch end

        # Next distillation step
        teacher_model = student_model


    print("Saving model...")
    os.makedirs(experiment_config.outdir, exist_ok=True) # type: ignore
    torch.save(student_model.state_dict(), f"{experiment_config.outdir}/model.pth")

    experiment_name = experiment_config.experiment_name
    outdir: str = experiment_config.outdir # type: ignore
    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))
    plt.figure(figsize=(10, 10))
    plt.plot(losses)
    plt.title(f"[{experiment_name}] Loss")
    plt.savefig(f"{outdir}/loss.png")
    plt.close()

    np.save(f"{outdir}/loss.npy", np.array(losses))
    plt.figure(figsize=(10, 10))
    plt.plot(losses)
    plt.title(f"[{experiment_name}] Loss")
    plt.savefig(f"{outdir}/loss.png")
    plt.close()

    print("Saving frames...")
    frames = np.stack(frames)
    np.save(f"{outdir}/frames.npy", frames)

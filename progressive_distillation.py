import argparse
import os
from typing import Dict, List
import random

import wandb

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from torch.profiler import profile, record_function, ProfilerActivity

from pydantic_core import from_json

# local imports
import datasets
from src.training import GracefulKiller, DiffusionTraining
from noise_scheduler import RawNoiseScheduler, DDPMScheduleConfig
from config import ProgressiveDistillationExperimentConfig, DeviceEnum, ModelTypeEnum
from model import get_model, MLP, MLPSPS, AnyModel
from metric import metric_nearest_distance

from ddpm import eval_iteration


class ProgressiveDistillationTraining(DiffusionTraining):

    def __init__(self, experiment_config: ProgressiveDistillationExperimentConfig):

        self.config = experiment_config

    def prepare_timesteps(self):

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
        student_timesteps = (timesteps / 2).long()

        return timesteps, timesteps_next, student_timesteps

    def training_step(
            self,
            teacher_model: AnyModel,
            teacher_noise_scheduler: RawNoiseScheduler,
            student_model: AnyModel,
            student_noise_scheduler: RawNoiseScheduler,
            ):

        timesteps, timesteps_next, student_timesteps = self.prepare_timesteps()

        # todo как-то мы будем использовать этот шум?
        noise = torch.randn(batch.shape, device=device)
        noisy = teacher_noise_scheduler.add_noise(batch, noise, timesteps)

        # 2 step of teacher model
        with torch.no_grad():
            teacher_noise_pred1 = teacher_model(noisy, timesteps)
            teacher_sample1 = teacher_noise_scheduler.step(teacher_noise_pred1, timesteps, noisy)

            teacher_noise_pred2 = teacher_model(teacher_sample1, timesteps_next)

        student_noise_pred = student_model(noisy, student_timesteps)

        # самодельный вывел из формул
        student_noise_scale = student_noise_scheduler.progressive_distillation_student_scale(student_timesteps)
        teacher_noise_total = teacher_noise_scheduler.progressive_distillation_teacher_total_noise(
            teacher_noise_pred1, teacher_noise_pred2,
            timesteps, timesteps_next,
            student_noise_scale
        )

        assert student_noise_pred.shape == teacher_noise_total.shape, f"{student_noise_pred.shape} != {teacher_noise_total.shape}"
        loss = F.mse_loss(student_noise_pred, teacher_noise_total)

        return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    arguments = config = parser.parse_args()

    with open(arguments.config, 'r') as f:
        config_json_data = f.read()

    experiment_config = ProgressiveDistillationExperimentConfig.model_validate_json(config_json_data)

    run = wandb.init(
        project="diffusion_sps",
        name=experiment_config.experiment_name,
        notes=f"From config {arguments.config}",
        config=experiment_config.model_dump()
    )

    dataset = datasets.get_dataset(experiment_config.dataset, n=100000)
    dataset_frame_numpy: np.ndarray = np.vstack([ t.numpy() for t in dataset.tensors ])
    dataset_frame_numpy = dataset_frame_numpy[:1000]
    dataloader = DataLoader(
        dataset, batch_size=experiment_config.train_batch_size, shuffle=True, drop_last=True
    )

    device: DeviceEnum = experiment_config.nn_device # type: ignore
    teacher_model = get_model(experiment_config)
    teacher_model = teacher_model.to(device)

    initial_teacher_schedule_config = DDPMScheduleConfig(
        num_timesteps=experiment_config.num_timesteps,
        beta_schedule=experiment_config.beta_schedule,
        device=device
    )
    teacher_noise_scheduler = RawNoiseScheduler.from_ddpm_schedule_config(initial_teacher_schedule_config)

    if experiment_config.teacher_checkpoint is not None:
        teacher_state_dict = torch.load(experiment_config.teacher_checkpoint)
        teacher_model.load_state_dict(teacher_state_dict)

    class NoopKiller:
        kill_now = False
    # killer = NoopKiller()
    killer = GracefulKiller()

    pd_training = ProgressiveDistillationTraining(experiment_config)

    for distillation_step in range(1, experiment_config.distillation_steps+1):
        if killer.kill_now:
            break

        student_model = get_model(experiment_config)
        student_model.load_state_dict(teacher_model.state_dict())

        # todo ablation study
        # сохранение таймстемпов
        student_timesteps_scale = 2**distillation_step
        student_model.set_time_mlp_scale(student_timesteps_scale)

        student_num_timesteps = int(experiment_config.num_timesteps / student_timesteps_scale)
        print("student_num_timesteps", student_num_timesteps, "student_timesteps_scale", student_timesteps_scale)

        # todo взять формулы для шедулера из формулы!
        # todo прямо сейчас это шедулер с увеличивающейся дисперсией
        student_beta_scale = 2**distillation_step if experiment_config.student_scheduler_beta_correction else 1
        student_ddpm_schedule_config = DDPMScheduleConfig(
            num_timesteps=student_num_timesteps,
            beta_schedule=experiment_config.beta_schedule,
            beta_end=(initial_teacher_schedule_config.beta_end * student_beta_scale),
            device=device
        )
        student_noise_scheduler = RawNoiseScheduler.from_ddpm_schedule_config(student_ddpm_schedule_config)

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

                loss = pd_training.training_step(
                    teacher_model,
                    teacher_noise_scheduler,
                    student_model,
                    student_noise_scheduler,
                )

                assert loss.item() < 10

                loss.backward(loss)

                nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

                progress_bar.update(1)

                logs = {"loss": loss.detach().item()}
                run.log(logs)

                progress_bar.set_postfix(logs)

            progress_bar.close()

            if epoch % experiment_config.save_images_step == 0 or epoch == experiment_config.num_epochs - 1:
                with torch.no_grad():
                    # generate data with the model to later visualize the learning process

                    frame = eval_iteration(experiment_config, student_model, student_noise_scheduler,  device=device, epoch=epoch, prefix=f'student_{student_noise_scheduler.num_timesteps}_')

                    metrics_value = metric_nearest_distance(frame, dataset_frame_numpy)
                    print("metrics_value", metrics_value)

                    validation_logs = {
                        ("validation/"+ k): v for k, v in metrics_value.model_dump().items()
                    }

                    frame_table = wandb.Table(data=frame, columns=["x", "y"])
                    plot_title = f"Timesteps={student_noise_scheduler.num_timesteps} Epoch {epoch}"
                    validation_logs[f"frames/{plot_title}"] = wandb.plot.scatter(frame_table, "x", "y", title=plot_title, split_table=True)
                    run.log(validation_logs)

                    if metrics_value.value_mean > 1.0:
                        raise Exception("too bad metrics")
            # Epochs end

        print("Saving model...")
        os.makedirs(experiment_config.outdir, exist_ok=True) # type: ignore
        torch.save(student_model.state_dict(), f"{experiment_config.outdir}/model_{student_noise_scheduler.num_timesteps}.pth")

        # Next distillation step
        teacher_model = student_model
        teacher_noise_scheduler = student_noise_scheduler

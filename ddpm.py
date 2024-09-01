import argparse
import os
import signal
from typing import Dict, List
import random
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

import datasets

from torch.profiler import profile, record_function, ProfilerActivity

from pydantic_core import from_json

# local imports
from src.training import GracefulKiller
from noise_scheduler import RawNoiseScheduler, DDPMScheduleConfig
from config import ExperimentConfig, DeviceEnum, ModelTypeEnum
from model import get_model, MLP, MLPSPS, AnyModel
from metric import metric_nearest_distance

# Import the W&B Python Library
import wandb

def train_iteration(experiment_config: ExperimentConfig, model: AnyModel, optimizer, batch, device: DeviceEnum):
    batch = batch[0].to(device)
    noise = torch.randn(batch.shape, device=device)

    if experiment_config.nn_model_type == ModelTypeEnum.mlp_sps:
        timesteps = torch.full([ batch.shape[0] ], random.randint(0, noise_scheduler.num_timesteps - 1))
    else:
        timesteps = torch.randint(
            0, noise_scheduler.num_timesteps, (batch.shape[0],),
            device=device,
        ).long()

    noisy = noise_scheduler.add_noise(batch, noise, timesteps)

    noise_pred = model(noisy, timesteps)

    loss = F.mse_loss(noise_pred, noise)
    loss.backward(loss)

    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    optimizer.zero_grad()

    return loss


def eval_iteration(experiment_config: ExperimentConfig, model: AnyModel, noise_scheduler: RawNoiseScheduler, device: DeviceEnum, scheduler_step=1, epoch=0, prefix=''):

    model.eval()
    sample = torch.randn(experiment_config.eval_batch_size, 2, device=device)
    timesteps_reversed = noise_scheduler.prepare_timesteps_for_sampling(step=scheduler_step)
    for i, t in enumerate(tqdm(timesteps_reversed)):

        full_timesteps = torch.full([experiment_config.eval_batch_size], t).long().to(device)
        with torch.no_grad():
            residual = model(sample, full_timesteps)
        sample = noise_scheduler.step(residual, full_timesteps, sample)
    sample_npy = sample.cpu().numpy()

    imgdir: str = experiment_config.imgdir # type: ignore
    os.makedirs(imgdir, exist_ok=True)

    xmin, xmax = -6, 6
    ymin, ymax = -6, 6

    plt.figure(figsize=(10, 10))
    plt.scatter(sample_npy[:, 0], sample_npy[:, 1], alpha=0.5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    image_path = f"{imgdir}/{prefix}{epoch}.png"
    plt.savefig(image_path)
    plt.title(f"{prefix}Epoch {epoch}")
    plt.close()

    return sample_npy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    arguments = config = parser.parse_args()

    # 1. Start a W&B Run

    #  2. Capture a dictionary of hyperparameters
    with open(arguments.config, 'r') as f:
        config_json_data = f.read()

    experiment_config = ExperimentConfig.model_validate_json(config_json_data)

    run = wandb.init(
        project="diffusion_sps",
        name=experiment_config.experiment_name,
        notes=f"From config {arguments.config}",
        config=experiment_config.model_dump(),
    )

    dataset = datasets.get_dataset(experiment_config.dataset)
    dataset_frame_numpy: np.ndarray = np.vstack([ t.numpy() for t in dataset.tensors ])
    dataloader = DataLoader(
        dataset, batch_size=experiment_config.train_batch_size, shuffle=True, drop_last=True
    )

    device: DeviceEnum = experiment_config.nn_device # type: ignore
    model = get_model(experiment_config)
    model = model.to(device)

    ddpm_schedule_config = DDPMScheduleConfig(
        num_timesteps=experiment_config.num_timesteps,
        beta_schedule=experiment_config.beta_schedule,
        device=device
    )
    noise_scheduler = RawNoiseScheduler.from_ddpm_schedule_config(ddpm_schedule_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment_config.learning_rate,
    )

    killer = GracefulKiller()

    outdir: str = experiment_config.outdir # type: ignore

    print("Training model...")
    for epoch in range(experiment_config.num_epochs):
        if killer.kill_now:
            break

        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            if killer.kill_now:
                break

            loss = train_iteration(experiment_config, model, optimizer, batch, device=device)

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item()
            }
            run.log(logs)

            progress_bar.set_postfix(logs)
        progress_bar.close()

        if epoch % experiment_config.save_images_step == 0 or epoch == experiment_config.num_epochs - 1:
            with torch.no_grad():
                # generate data with the model to later visualize the learning process
                frame = eval_iteration(experiment_config, model, noise_scheduler, device=device, epoch=epoch)

                metrics_value = metric_nearest_distance(frame, dataset_frame_numpy)
                validation_logs = {
                    ("validation/"+ k): v for k, v in metrics_value.model_dump().items()
                }

                frame_table = wandb.Table(data=frame, columns=["x", "y"])
                plot_title = f"Epoch {epoch}"
                validation_logs[f"frames/{plot_title}"] = wandb.plot.scatter(frame_table, "x", "y", title=plot_title, split_table=True)
                run.log(validation_logs)

    print("Saving model...")
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")

    experiment_name = experiment_config.experiment_name
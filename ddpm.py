import argparse
import os
import signal

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
from noise_scheduler import NoiseScheduler
from config import ExperimentConfig, DeviceEnum
from model import get_model, MLP, MLPSPS, AnyModel

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, signum, frame):
    self.kill_now = True

def train_iteration(experiment_config: ExperimentConfig, model: AnyModel, optimizer, device: DeviceEnum):
    batch = batch[0].to(device)
    noise = torch.randn(batch.shape, device=device)

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


def eval_iteration(experiment_config: ExperimentConfig, model: AnyModel, device: DeviceEnum):

    model.eval()
    sample = torch.randn(config.eval_batch_size, 2, device=device)
    timesteps = torch.tensor(list(range(len(noise_scheduler)))[::-1], dtype=torch.long)
    for i, t in enumerate(tqdm(timesteps)):
        if killer.kill_now:
            break

        t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long().to(device)
        with torch.no_grad():
            residual = model(sample, t)
        sample = noise_scheduler.step(residual, t[0], sample)
    sample_npy = sample.cpu().numpy()

    imgdir = experiment_config.imgdir()
    os.makedirs(imgdir, exist_ok=True)

    xmin, xmax = -6, 6
    ymin, ymax = -6, 6

    plt.figure(figsize=(10, 10))
    plt.scatter(sample_npy[:, 0], sample_npy[:, 1], alpha=0.5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    image_path = f"{imgdir}/{epoch}.png"
    plt.savefig(image_path)
    plt.title(f"Epoch {epoch}")
    plt.close()

    return sample_npy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    config = parser.parse_args()

    experiment_config = ExperimentConfig.model_validate_json(from_json(config.config))

    dataset = datasets.get_dataset(config.dataset)
    dataloader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True
    )

    if config.device == 'auto':
        device = DeviceEnum.cpu
        if torch.backends.mps.is_available():
            device = DeviceEnum.mps
        elif torch.cuda.is_available():
            device = DeviceEnum.cuda
    else:
        device = config.device

    model = get_model(experiment_config)
    model = model.to(device)

    outdir = experiment_config.outdir()

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    killer = GracefulKiller()

    global_step = 0
    frames = []
    losses = []
    print("Training model...")
    for epoch in range(config.num_epochs):
        if killer.kill_now:
            break

        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            if killer.kill_now:
                break

            loss = train_iteration(experiment_config, model, optimizer, device=device)

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(logs)
            global_step += 1
        progress_bar.close()

        if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
            # generate data with the model to later visualize the learning process
            frame = eval_iteration(experiment_config, model, device=device)
            frames.append(frame)

    print("Saving model...")
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))
    plt.figure(figsize=(10, 10))
    plt.plot(losses)
    plt.title("Loss")
    plt.savefig(f"{outdir}/loss.png")
    plt.close()

    print("Saving frames...")
    frames = np.stack(frames)
    np.save(f"{outdir}/frames.npy", frames)

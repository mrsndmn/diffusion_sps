import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import numpy as np

# local imports
from config import ProgressiveDistillationExperimentConfig, DeviceEnum, ModelTypeEnum
from model import get_model, MLP, MLPSPS, AnyModel
import datasets
from noise_scheduler import NoiseScheduler

def test_pd():

    with open('configs/progressive_distillation/progressive_distillation.json', 'r') as f:
        config_json_data = f.read()

    experiment_config = ProgressiveDistillationExperimentConfig.model_validate_json(config_json_data)

    dataset = datasets.get_dataset(experiment_config.dataset, n=100000)
    dataset_frame_numpy: np.ndarray = np.vstack([ t.numpy() for t in dataset.tensors ])
    dataloader = DataLoader(
        dataset, batch_size=experiment_config.num_timesteps - 2, shuffle=True, drop_last=True
    )

    device: DeviceEnum = experiment_config.nn_device # type: ignore
    teacher_model = get_model(experiment_config)
    teacher_model = teacher_model.to(device)

    teacher_noise_scheduler = NoiseScheduler(
        num_timesteps=experiment_config.num_timesteps,
        beta_schedule=experiment_config.beta_schedule,
        device=device,
    )

    student_model = get_model(experiment_config)
    student_model.load_state_dict(teacher_model.state_dict())

    timesteps = torch.arange(teacher_noise_scheduler.num_timesteps).long().to(device)
    timesteps = timesteps[2:]
    timesteps_next = timesteps - 1

    batch = next(iter(dataloader))
    batch = batch[0].to(device)

    noise = torch.randn(batch.shape, device=device)
    noisy = teacher_noise_scheduler.add_noise(batch, noise, timesteps)

    teacher_noise_pred1 = teacher_model(noisy, timesteps)
    teacher_sample1 = teacher_noise_scheduler.step(teacher_noise_pred1, timesteps, noisy)
    teacher_noise_pred2 = teacher_model(teacher_sample1, timesteps_next)
    teacher_sample2 = teacher_noise_scheduler.step(teacher_noise_pred2, timesteps_next, teacher_sample1)

    # student_noise_pred = student_model(noisy, timesteps)

    sigma_t = teacher_noise_scheduler.sqrt_one_minus_alphas_cumprod[timesteps].unsqueeze(1)
    sigma_tss = teacher_noise_scheduler.sqrt_one_minus_alphas_cumprod[timesteps_next].unsqueeze(1)
    alpha_t = teacher_noise_scheduler.sqrt_alphas_cumprod[timesteps].unsqueeze(1)
    alpha_tss = teacher_noise_scheduler.sqrt_alphas_cumprod[timesteps_next].unsqueeze(1)


    teacher_noise_total = (teacher_sample2 - (sigma_tss / sigma_t) * noisy) / ( alpha_tss - (sigma_tss / sigma_t) * alpha_t )

    teacher_sample_self_restored = teacher_noise_scheduler.step(teacher_noise_total, timesteps, noisy)

    loss = F.mse_loss(teacher_sample_self_restored, teacher_sample2)

    # breakpoint()
    print("loss.item()", loss.item())

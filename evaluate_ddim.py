import torch
import numpy as np

import os

from config import DDIMEvaluationConfig, DeviceEnum
import argparse
import wandb

import matplotlib.pyplot as plt


from torch.utils.data import DataLoader

# local imports
import datasets
from ddpm import eval_iteration
from noise_scheduler import DDIMSampler
from model import get_model
from metric import metric_nearest_distance

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    arguments = config = parser.parse_args()

    with open(arguments.config, 'r') as f:
        config_json_data = f.read()

    experiment_config = DDIMEvaluationConfig.model_validate_json(config_json_data)

    run = wandb.init(
        project="diffusion_sps",
        name="ddim.evaluate",
        notes=f"From config {arguments.config}",
        config=experiment_config.model_dump()
    )

    dataset = datasets.get_dataset(experiment_config.dataset, n=100000)
    dataset_frame_numpy: np.ndarray = np.vstack([ t.numpy() for t in dataset.tensors ])
    dataloader = DataLoader(
        dataset, batch_size=1000, shuffle=True, drop_last=True
    )

    device: DeviceEnum = experiment_config.nn_device # type: ignore
    model = get_model(experiment_config)
    model.load_state_dict(torch.load(experiment_config.checkpoint))

    ddim_sampler = DDIMSampler(
        model,
        num_timesteps=experiment_config.num_timesteps,
    )

    x_t = torch.randn([1000, 2], device=device)

    for steps in [ 128, 16 ]:
        for eta in [ 0.0, 1.0 ]:
            x_0 = ddim_sampler.forward(
                x_t,
                eta=eta,
                steps=steps,
                # steps=experiment_config.ddim_steps,
                # eta=experiment_config.ddim_eta,
            )

            frame = x_0.detach().cpu().numpy()
            plt.scatter(frame[:, 0], frame[:, 1], alpha=0.5)
            plt.title(f"DDIM steps={experiment_config.ddim_steps}, eta={experiment_config.ddim_eta}")

            os.makedirs("exps/evaluate_ddim", exist_ok=True)
            plt.savefig(f"exps/evaluate_ddim/imagesteps_steps-{steps}_eta-{eta}.png")
            plt.close()

            metric_values = metric_nearest_distance(frame, dataset_frame_numpy)
            print(f"steps={steps} eta={eta} metric_values", metric_values.value_mean, metric_values.value_external_mean)

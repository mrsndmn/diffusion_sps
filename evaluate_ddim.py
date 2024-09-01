import torch
import numpy as np

from config import DDIMEvaluationConfig
import argparse
import wandb


from torch.utils.data import DataLoader

# local imports
import datasets
from ddpm import eval_iteration
from model import get_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    arguments = config = parser.parse_args()

    with open(arguments.config, 'r') as f:
        config_json_data = f.read()

    experiment_config = DDIMEvaluationConfig.model_validate_json(config_json_data)

    run = wandb.init(
        project="diffusion_sps",
        name=experiment_config.experiment_name,
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


import torch
import pytest

from model import MLPSPS

def test_mlp_sps():
    num_timesteps = 10
    model = MLPSPS(num_timesteps=num_timesteps)
    device = 'cpu'

    model.eval()
    batch_sise = 3
    sample = torch.randn(batch_sise, 2, device=device)
    timesteps = torch.zeros(batch_sise).long().to(device)
    prediction = model(sample, timesteps)

    assert sample.shape == prediction.shape


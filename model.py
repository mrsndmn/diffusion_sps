from typing import Union
import torch
import torch.nn as nn
from positional_embeddings import PositionalEmbedding

from config import ExperimentConfig


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))

class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x

class MLPSPS(nn.Module):
    def __init__(
            self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
            time_emb: str = "sinusoidal", input_emb: str = "sinusoidal", num_timesteps=50,
        ):
        super().__init__()

        self.num_timesteps = num_timesteps

        mlps_for_timesteps = []
        for i in range(num_timesteps):
            mlp = MLP(
                hidden_size=hidden_size,
                hidden_layers=hidden_layers,
                emb_size=emb_size,
                time_emb=time_emb,
                input_emb=input_emb,
            )
            mlps_for_timesteps.append(mlp)

        self.mlp_sps = nn.ModuleList(mlps_for_timesteps)

    def init_from_checkpoint_weights(self, single_model_checkpoint):
        torch_state_dict = torch.load(single_model_checkpoint)
        for i in range(self.num_timesteps):
            self.mlp_sps[i].load_state_dict(torch_state_dict)

    def forward(self, x, t):
        prediction = []
        # print("self.mlp_sps", len(self.mlp_sps), "timesteps", t)
        for i in range(t.shape[0]):
            t_i = t[i]
            prediction.append(self.mlp_sps[t_i](x[i:i+1], t[i:i+1]))

        return torch.vstack(prediction)

    def forward_single_timestep(self, x, t):
        assert (t == t[0]).all(), 'all values of t is expected to be equal'

        predictions = self.mlp_sps[t[0]](x, t)
        return predictions



AnyModel = Union[MLP, MLPSPS]

def get_model(config: ExperimentConfig):

    model: Union[MLP | MLPSPS]

    if config.nn_model_type == 'mlp':
        model = MLP(
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
            emb_size=config.embedding_size,
            time_emb=config.time_embedding,
            input_emb=config.input_embedding
        )
    elif config.nn_model_type == 'mlp_sps':
        model = MLPSPS(
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
            emb_size=config.embedding_size,
            time_emb=config.time_embedding,
            input_emb=config.input_embedding,
            num_timesteps=config.num_timesteps,
        )
        if config.sps_checkpoint is not None:
            model.init_from_checkpoint_weights(config.sps_checkpoint)
    else:
        raise ValueError(f"unsupported nn_model_type {config.nn_model_type}")

    return model
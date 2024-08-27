from typing import Optional
from typing_extensions import Self

from pydantic import BaseModel, model_validator, computed_field
from enum import Enum

import os

class ModelTypeEnum(str, Enum):
    mlp         = "mlp"
    mlp_sps     = "mlp_sps"

class DatasetEnum(str, Enum):
    circle  = "circle"
    dino    = "dino"
    line    = "line"
    moons   = "moons"

class DeviceEnum(str, Enum):
    auto = "auto"
    mps = "mps"
    cuda = "cuda"
    cpu = "cpu"

class BetaScheduleEnum(str, Enum):
    linear = "linear"
    quadratic = "quadratic"

class InputEmbeddingEnum(str, Enum):
    sinusoidal = "sinusoidal"
    learnable  = "learnable"
    linear     = "linear"
    zero       = "zero"

class TimeEmbeddingEnum(str, Enum):
    sinusoidal = "sinusoidal"
    learnable  = "learnable"
    linear     = "linear"
    identity   = "identity"


class ExperimentConfig(BaseModel):
    experiment_name: str
    dataset: DatasetEnum
    nn_model_type: ModelTypeEnum

    # training params
    num_epochs: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float

    # evaluate params
    save_images_step: int

    sps_checkpoint: Optional[str]

    hidden_layers: int
    embedding_size: int
    num_timesteps: int
    hidden_size: int
    beta_schedule: BetaScheduleEnum
    time_embedding: TimeEmbeddingEnum
    input_embedding: InputEmbeddingEnum
    device: DeviceEnum

    @computed_field
    def outdir(self) -> str:
        return os.path.join("exps", self.experiment_name)

    @computed_field
    def imgdir(self) -> str:
        return os.path.join(self.outdir, "images") # type: ignore

    @model_validator(mode='after')
    def runtime_after_validator(self: Self) -> Self:

        if self.nn_model_type != ModelTypeEnum.mlp_sps and self.sps_checkpoint is not None:
            raise ValueError("`sps_checkpoint` can be used only with nn_model_type=mlp_sps")

        return self
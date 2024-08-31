from typing import Optional
from typing_extensions import Self

from dataclasses import dataclass

import torch
from pydantic import BaseModel, model_validator, computed_field, ConfigDict
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
    model_config = ConfigDict(extra="forbid")

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

    @computed_field
    def nn_device(self) -> DeviceEnum:
        if self.device == 'auto':
            device = DeviceEnum.cpu
            if torch.backends.mps.is_available():
                device = DeviceEnum.mps
            elif torch.cuda.is_available():
                device = DeviceEnum.cuda
        else:
            device = self.device

        return device


    @model_validator(mode='after')
    def runtime_after_validator(self: Self) -> Self:

        if self.nn_model_type != ModelTypeEnum.mlp_sps and self.sps_checkpoint is not None:
            raise ValueError("`sps_checkpoint` can be used only with nn_model_type=mlp_sps")

        return self

@dataclass
class ProgressiveDistillationExperimentConfig(ExperimentConfig):

    distillation_steps: int # сколько раз будет уменьшаться количество таймстепов?
    teacher_checkpoint: str

    student_scheduler_beta_correction: bool

    @model_validator(mode='after')
    def runtime_after_validate_progressive_distillation(self: Self) -> Self:
        max_divider: int = 2**self.distillation_steps
        if self.num_timesteps % max_divider != 0:
            raise ValueError("`num_timesteps` must be dividable by `2**distillation_steps`")

        if self.num_timesteps < max_divider:
            raise ValueError("`num_timesteps` must be less than `2**distillation_steps`")

        return self


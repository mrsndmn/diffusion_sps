from typing_extensions import Self
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from torch.nn import functional as F
import numpy as np


class DDPMScheduleConfig():
    def __init__(self,
            num_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            device=None
        ):

        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule

        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32, device=device) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_sqrt = torch.sqrt(self.alphas)

        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        self.inv_sqrt_alphas =  1 / torch.sqrt(self.alphas)

        self.device = device

        return


class RawNoiseScheduler():
    def __init__(self,
                 num_timesteps: int,
                 alphas_sqrt: torch.Tensor,
                 variances: torch.Tensor,
                 noise_multiplicator_k: torch.Tensor,
                 step_scaler: torch.Tensor,
                 step_model_output_scaler: torch.Tensor,
                 add_noise_noise_scaler: torch.Tensor,
                 add_noise_x_scaler: torch.Tensor,
                 device=None,
                 ):

        self.num_timesteps = num_timesteps

        # validate params
        self.validate_schedule_params(
            alphas_sqrt=alphas_sqrt,
            variances=variances,
            noise_multiplicator_k=noise_multiplicator_k,
            step_scaler=step_scaler,
            step_model_output_scaler=step_model_output_scaler,
            add_noise_noise_scaler=add_noise_noise_scaler,
            add_noise_x_scaler=add_noise_x_scaler,
        )

        self.alphas_sqrt = alphas_sqrt

        self.variances = variances.clip(1e-20)

        self.noise_multiplicator_k = noise_multiplicator_k

        # inv_sqrt_alphas_t
        self.step_scaler = step_scaler
        # betas_t / sqrt_one_minus_alphas_cumprod_t
        self.step_model_output_scaler = step_model_output_scaler

        # sqrt_alphas_cumprod
        self.add_noise_noise_scaler = add_noise_noise_scaler
        # sqrt_one_minus_alphas_cumprod
        self.add_noise_x_scaler = add_noise_x_scaler

    def validate_schedule_params(self, **kwargs):
        for k, v in kwargs.items():
            assert v.dim() == 2, f"{k}.dim != 2"
            assert v.shape == torch.Size([self.num_timesteps, 1]), f"{k}.shape[1] == 1"

    def get_variance(self, t):

        variance = self.variances[t]
        return variance

    def step(self, model_output: torch.Tensor, timesteps: torch.Tensor, sample, noise=None):
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        pred_prev_sample = self.step_scaler[timesteps] * (
            sample - self.step_model_output_scaler[timesteps] * model_output
        )

        variance = torch.zeros_like(pred_prev_sample)
        if (timesteps > 0).any():
            if noise is None:
                noise = torch.randn_like(model_output)

            timesteps_gt_0 = timesteps[timesteps > 0]
            variance[timesteps > 0] = (self.get_variance(timesteps_gt_0) ** 0.5) * noise[timesteps > 0]

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.add_noise_x_scaler[timesteps]
        s2 = self.add_noise_noise_scaler[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps

    def prepare_timesteps_for_sampling(self, step=1):
        return torch.tensor(list(range(self.num_timesteps))[::-step], dtype=torch.long)

    def progressive_distillation_teacher_total_noise(self, teacher_noise_pred1, teacher_noise_pred2, timesteps, timesteps_next, student_noise_scale):
        noise_mult_t = self.noise_multiplicator_k[timesteps]
        noise_mult_t_next = (self.noise_multiplicator_k[timesteps_next] * self.alphas_sqrt[timesteps])
        teacher_noise_total = (teacher_noise_pred1 * noise_mult_t + teacher_noise_pred2 * noise_mult_t_next) / student_noise_scale

        return teacher_noise_total

    def progressive_distillation_student_scale(self, timesteps):
        return self.noise_multiplicator_k[timesteps]

    @classmethod
    def from_ddpm_schedule_config(
            klass,
            schedule_config: DDPMScheduleConfig,
        ) -> Self:

        # used for progressive distillation
        noise_multiplicator_k = schedule_config.betas / schedule_config.sqrt_one_minus_alphas_cumprod

        # prepare scheduler precumputed params
        alphas_sqrt = schedule_config.alphas_sqrt
        variances = schedule_config.betas * (1. - schedule_config.alphas_cumprod_prev) / (1. - schedule_config.alphas_cumprod)
        step_scaler = schedule_config.inv_sqrt_alphas
        step_model_output_scaler = schedule_config.betas / schedule_config.sqrt_one_minus_alphas_cumprod
        add_noise_x_scaler = schedule_config.sqrt_alphas_cumprod
        add_noise_noise_scaler = schedule_config.sqrt_one_minus_alphas_cumprod

        return klass(
            num_timesteps=schedule_config.num_timesteps,
            alphas_sqrt=alphas_sqrt.unsqueeze(1),
            variances=variances.unsqueeze(1),
            noise_multiplicator_k=noise_multiplicator_k.unsqueeze(1),
            step_scaler=step_scaler.unsqueeze(1),
            step_model_output_scaler=step_model_output_scaler.unsqueeze(1),
            add_noise_noise_scaler=add_noise_noise_scaler.unsqueeze(1),
            add_noise_x_scaler=add_noise_x_scaler.unsqueeze(1),
            device=schedule_config.device
        )


# todo refactoring
def extract(v, i, shape):
    """
    Get the i-th number in v, and the shape of v is mostly (T, ), the shape of i is mostly (batch_size, ).
    equal to [v[index] for index in i]
    """
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out

class DDIMSampler(nn.Module):
    def __init__(
                self,
                model,
                num_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
            ):
        super().__init__()

        self.model = model

        self.T = num_timesteps
        # generate T steps of beta
        beta_t = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - beta_t
        self.register_buffer("alpha_t_bar", torch.cumprod(alpha_t, dim=0))

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step: int, prev_time_step: int, eta: float):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)

        # predict noise using model
        epsilon_theta_t = self.model(x_t, t)

        # calculate x_{t-1}
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_one = (
                torch.sqrt(alpha_t_prev / alpha_t) * x_t +
                (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                    (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                sigma_t * epsilon_t
        )
        return x_t_minus_one

    @torch.no_grad()
    def forward(self,
                x_t,
                steps: int = 1,
                method="linear",
                eta=0.0,
                only_return_x_0: bool = True,
                interval: int = 1
            ) -> torch.Tensor:
        """
        Parameters:
            x_t: Standard Gaussian noise. A tensor with shape (batch_size, channels, height, width).
            steps: Sampling steps.
            method: Sampling method, can be "linear" or "quadratic".
            eta: Coefficients of sigma parameters in the paper. The value 0 indicates DDIM, 1 indicates DDPM.
            only_return_x_0: Determines whether the image is saved during the sampling process. if True,
                intermediate pictures are not saved, and only return the final result $x_0$.
            interval: This parameter is valid only when `only_return_x_0 = False`. Decide the interval at which
                to save the intermediate process pictures, according to `step`.
                $x_t$ and $x_0$ will be included, no matter what the value of `interval` is.

        Returns:
            if `only_return_x_0 = True`, will return a tensor with shape (batch_size, channels, height, width),
            otherwise, return a tensor with shape (batch_size, sample, channels, height, width),
            include intermediate pictures.
        """
        if method == "linear":
            a = self.T // steps
            time_steps = np.asarray(list(range(0, self.T, a)))
        elif method == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).astype(np.int64)
        else:
            raise NotImplementedError(f"sampling method {method} is not implemented!")

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        # time_steps = time_steps + 1
        time_steps = time_steps
        # previous sequence
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        x = [x_t]
        with tqdm(reversed(range(0, steps)), total=steps) as sampling_steps:
            for i in sampling_steps:
                x_t = self.sample_one_step(x_t, time_steps[i], time_steps_prev[i], eta)

                if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                    x.append(torch.clip(x_t, -1.0, 1.0))

                sampling_steps.set_postfix(ordered_dict={"step": i + 1, "sample": len(x)})

        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]
        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]

class PDStudentNoiseScheduler():
    pass


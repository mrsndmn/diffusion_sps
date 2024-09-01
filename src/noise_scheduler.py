from typing_extensions import Self
import torch
from torch.nn import functional as F

class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear",
                 device=None):

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32, device=device) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_sqrt = torch.sqrt(self.alphas)

        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        self.inv_sqrt_alphas =  1 / torch.sqrt(self.alphas)

        # used for progressive distillation
        self.noise_multiplicator_k = self.betas / self.sqrt_one_minus_alphas_cumprod


    def get_variance(self, t):
        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance.unsqueeze(1)

    def step(self, model_output, timestep: torch.Tensor, sample, noise=None):
        t = timestep

        betas_t = self.betas[t].unsqueeze(1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        inv_sqrt_alphas_t = self.inv_sqrt_alphas[t].unsqueeze(1)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        pred_prev_sample = inv_sqrt_alphas_t * (
            sample - betas_t / sqrt_one_minus_alphas_cumprod_t * model_output
        )

        variance = torch.zeros_like(pred_prev_sample)
        if (t > 0).any():
            if noise is None:
                noise = torch.randn_like(model_output)

            t_gt_0 = t[t > 0]
            variance[t > 0] = (self.get_variance(t_gt_0) ** 0.5) * noise[t > 0]

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps

    def prepare_timesteps_for_sampling(self, step=1):
        return torch.tensor(list(range(self.num_timesteps))[::-step], dtype=torch.long)

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


# class StudentNoiseScheduler(RawNoiseScheduler):

#     @classmethod
#     def from_teacher_noise_scheduler(klass, teacher_noise_scheduler: RawNoiseScheduler) -> Self:

#         student_num_timesteps = teacher_noise_scheduler.num_timesteps // 2

#         alphas_sqrt = teacher_noise_scheduler.alphas_sqrt[::2] * teacher_noise_scheduler.alphas_sqrt[1::2]
#         variances = teacher_noise_scheduler.variances[::2] / teacher_noise_scheduler.alphas_sqrt[1::2] + teacher_noise_scheduler.variances[1::2]

#         # TODO copypaste from ddpm scheduler config
#         step_scaler = 1 / alphas_sqrt

#         noise_multiplicator_k

#         return klass(
#             num_timesteps=student_num_timesteps,
#             alphas_sqrt=alphas_sqrt.unsqueeze(1),
#             variances=variances.unsqueeze(1),
#             noise_multiplicator_k=noise_multiplicator_k.unsqueeze(1),
#             step_scaler=step_scaler.unsqueeze(1),
#             step_model_output_scaler=step_model_output_scaler.unsqueeze(1),
#             add_noise_noise_scaler=add_noise_noise_scaler.unsqueeze(1),
#             add_noise_x_scaler=add_noise_x_scaler.unsqueeze(1),
#             device=schedule_config.device
#         )


class PDStudentNoiseScheduler():
    pass


from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scheduler import DPMSolverScheduler
from tqdm import tqdm


class DiffusionModule(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: DPMSolverScheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    def get_loss(self, x0, class_label=None, noise=None):
        B = x0.shape[0]
        timestep = self.var_scheduler.uniform_sample_t(B, self.device)
        x_noisy, noise = self.var_scheduler.add_noise(x0, timestep)
        if class_label is not None:
            noise_pred = self.network(x_noisy, timestep, class_label=class_label)
        else:
            noise_pred = self.network(x_noisy, timestep)

        loss = F.mse_loss(noise_pred.flatten(), noise.flatten(), reduction="mean")
        return loss

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    def q_sample(self, x0, t, noise=None):
        t = t.long()
        if noise is None:
            noise = torch.randn_like(x0)
        return self.var_scheduler.add_noise(x0, t, noise)[0]

    @torch.no_grad()
    def sample(
        self,
        shape,
        return_traj=False,
    ):
        x_T = torch.randn(shape).to(self.device)

        traj = [x_T]
        for t in tqdm(self.var_scheduler.timesteps):
            x_t = traj[-1]
            noise_pred = self.network(x_t, t.to(self.device))
            x_t_prev = self.var_scheduler.step(x_t, t, noise_pred)
            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]

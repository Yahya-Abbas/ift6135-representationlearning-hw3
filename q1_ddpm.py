from typing import Optional, Tuple

import torch
from torch import nn


class DenoiseDiffusion:
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta

    def gather(self, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        c_ = c.gather(-1, t)
        return c_.reshape(-1, 1, 1, 1)

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q(x_t | x_0): return the closed-form mean and variance.
        # x0 shape: (batch_size, channels, height, width)
        # t shape: (batch_size,)
        # return shapes: both should broadcast correctly with x0
        # ==========================
        alpha_bar_t = self.gather(self.alpha_bar, t)
        mean = torch.sqrt(alpha_bar_t) * x0
        var = 1 - alpha_bar_t

        return mean, var

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(x0)
        # Sample x_t from the forward process using the reparameterization formula.
        # x0 shape: (batch_size, channels, height, width)
        # t shape: (batch_size,)
        # eps shape: same as x0
        # return shape: same as x0
        # ==========================
        mean, var = self.q_xt_x0(x0, t)
        sample = mean + torch.sqrt(var) * eps

        return sample

    def p_xt_prev_xt(self, xt: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reverse process p_theta(x_{t-1} | x_t).
        # xt shape: (batch_size, channels, height, width)
        # t shape: (batch_size,)
        # return mu_theta shape: same as xt
        # return var shape: broadcastable with xt
        # ==========================
        epsilon_theta = self.eps_model(xt, t)
        alpha_bar_t = self.gather(self.alpha_bar, t)
        beta_t = self.gather(self.beta, t)
        alpha_t = self.gather(self.alpha, t)

        rhs = (beta_t/torch.sqrt(1 - alpha_bar_t)) * epsilon_theta
        mu_theta = (1 / torch.sqrt(alpha_t)) * (xt - rhs)
        var = self.gather(self.sigma2, t)

        return mu_theta, var

    # Alias retained so newer wording can map to the W25-compatible implementation.
    def p_mean_variance(self, xt: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.p_xt_prev_xt(xt, t)

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, set_seed: bool = False) -> torch.Tensor:
        if set_seed:
            torch.manual_seed(42)
        # Draw one sample from p_theta(x_{t-1} | x_t).
        # xt shape: (batch_size, channels, height, width)
        # t shape: (batch_size,)
        # return shape: same as xt
        # ==========================
        mu_theta, var = self.p_mean_variance(xt, t)
        noise = torch.randn_like(xt)
        # mask for t = 0
        # mask = (t > 0).view(-1, 1, 1, 1) 
        # sample = mu_theta + torch.sqrt(var) * noise * mask
        sample = mu_theta + torch.sqrt(var) * noise
        
        return sample

    def loss(
        self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed: bool = False
    ) -> torch.Tensor:
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0)

        # Simplified DDPM denoising loss.
        # x0 shape: (batch_size, channels, height, width)
        # noise shape: same as x0
        # return: scalar loss tensor
        # ===========================
        
        xt = self.q_sample(x0, t, noise)
        eps_theta = self.eps_model(xt, t)
        loss = ((noise - eps_theta) ** 2).sum(dim=dim).mean()
        return loss

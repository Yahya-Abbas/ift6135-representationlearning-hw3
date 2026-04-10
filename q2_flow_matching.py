from typing import Optional

import torch
from torch import nn


class FlowMatching:
    def __init__(self, velocity_model: nn.Module, device: torch.device, time_scale: int = 1000):
        super().__init__()
        self.velocity_model = velocity_model
        self.device = device
        self.time_scale = time_scale

    def _model_time(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(batch_size)
        elif t.ndim > 1:
            t = t.reshape(batch_size)
        return (t * self.time_scale).long()

    def predict_velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_model = self._model_time(t.to(x_t.device), x_t.shape[0])
        return self.velocity_model(x_t, t_model)

    def sample_xt(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Straight-line interpolation path x_t = (1 - t) x_0 + t x_1.
        # x0 shape: (batch_size, channels, height, width)
        # x1 shape: same as x0
        # t shape: (batch_size, 1, 1, 1)
        # return shape: same as x0
        # ==========================
        x_t = (1 - t) * x0 + t * x1
        return x_t

    def compute_conditional_velocity(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # Conditional velocity for the straight-line path.
        # x0 shape: (batch_size, channels, height, width)
        # x1 shape: same as x0
        # return shape: same as x0
        # ==========================
        u_t = x1 - x0
        return u_t

    def loss(
        self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed: bool = False
    ) -> torch.Tensor:
        if set_seed:
            torch.manual_seed(42)

        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.rand((batch_size, 1, 1, 1), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)

        # Flow Matching training loss.
        # x0 shape: (batch_size, channels, height, width)
        # noise shape: same as x0
        # return: scalar loss tensor
        # ==========================
        xt = self.sample_xt(x0, noise, t)
        ut = self.compute_conditional_velocity(x0, noise)
        v_theta = self.predict_velocity(xt, t)
        loss = ((v_theta - ut) ** 2).sum(dim=dim).mean()
        
        return loss

    def euler_step(self, x_t: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        # One Euler ODE step.
        # x_t shape: (batch_size, channels, height, width)
        # t: current scalar time
        # dt: scalar step size
        # return shape: same as x_t
        # ==========================
        t = torch.tensor(t, device=x_t.device)
        v_theta = self.predict_velocity(x_t, t)
        x_next = x_t + dt * v_theta
        return x_next

    def euler_sample(self, noise: torch.Tensor, n_steps: int) -> torch.Tensor:
        dt = -1.0 / n_steps
        x_t = noise
        for step in range(n_steps):
            t = 1.0 - step / n_steps
            x_t = self.euler_step(x_t, t, dt)
        return x_t

    def midpoint_step(self, x_t: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        # One midpoint ODE step.
        # x_t shape: (batch_size, channels, height, width)
        # t: current scalar time
        # dt: scalar step size
        # return shape: same as x_t
        # ==========================
        x_next = self.euler_step(x_t, t, dt/2)
        t_mid = torch.tensor(t + dt/2, device=x_t.device)
        v_theta = self.predict_velocity(x_next, t_mid)
        x_next = x_t + dt * v_theta
        return x_next

    def midpoint_sample(self, noise: torch.Tensor, n_steps: int) -> torch.Tensor:
        dt = -1.0 / n_steps
        x_t = noise
        for step in range(n_steps):
            t = 1.0 - step / n_steps
            x_t = self.midpoint_step(x_t, t, dt)
        return x_t

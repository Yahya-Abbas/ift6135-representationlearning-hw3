"""Provided support code for Assignment 3. You do not need to modify this file."""

import copy
import os
import time

import torch
from matplotlib import pyplot as plt
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from ddpm_utils.args import args


torch.manual_seed(42)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class Trainer:
    def __init__(self, args, velocity_model, flow_matching):
        self.velocity_model = velocity_model.to(args.device)
        self.flow_matching = flow_matching
        self.optimizer = torch.optim.Adam(self.velocity_model.parameters(), lr=args.learning_rate)
        self.args = args
        self.current_epoch = 0
        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.velocity_model).eval().requires_grad_(False)

    def _use_amp(self) -> bool:
        return self.args.fp16_precision and str(self.args.device).startswith("cuda")

    def train_epoch(self, dataloader, scaler):
        current_lr = round(self.optimizer.param_groups[0]["lr"], 5)
        i = 0
        running_loss = 0.0
        device_type = "cuda" if str(self.args.device).startswith("cuda") else "cpu"
        with tqdm(range(len(dataloader)), desc="Epoch : - lr: - Loss :") as progress:
            for x0 in dataloader:
                i += 1
                x0 = x0.to(self.args.device)
                with autocast(device_type=device_type, enabled=self._use_amp()):
                    loss = self.flow_matching.loss(x0)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.ema.step_ema(self.ema_model, self.velocity_model)

                running_loss += loss.item()
                self.loss_per_iter.append(running_loss / i)
                progress.update()
                progress.set_description(
                    f"Epoch: {self.current_epoch}/{self.args.epochs} - lr: {current_lr} - Loss: {round(running_loss / i, 2)}"
                )
            progress.set_description(
                f"Epoch: {self.current_epoch}/{self.args.epochs} - lr: {current_lr} - Loss: {round(running_loss / len(dataloader), 2)}"
            )

            self.scheduler.step()

    def train(self, dataloader):
        scaler = GradScaler(device="cuda", enabled=self._use_amp())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.loss_per_iter = []
        for current_epoch in range(self.current_epoch, self.args.epochs):
            self.current_epoch = current_epoch
            self.train_epoch(dataloader, scaler)

            if current_epoch % self.args.show_every_n_epochs == 0:
                samples = self.sample(
                    method="midpoint",
                    n_steps=50,
                    n_samples=max(9, self.args.n_samples),
                    set_seed=True,
                )
                print(f"Showing/saving Flow Matching samples from epoch {self.current_epoch}")
                self.show_save(
                    samples,
                    show=True,
                    save=True,
                    file_name=f"flow_matching_epoch_{self.current_epoch}_midpoint_50.png",
                )

            if (current_epoch + 1) % self.args.save_every_n_epochs == 0:
                self.save_model()

    def sample(
        self,
        method: str = "euler",
        n_steps: int = 50,
        noise: torch.Tensor = None,
        n_samples: int = None,
        set_seed: bool = False,
    ) -> torch.Tensor:
        if set_seed:
            torch.manual_seed(42)

        if n_samples is None:
            n_samples = self.args.n_samples
        if noise is None:
            noise = torch.randn(
                [
                    n_samples,
                    self.args.image_channels,
                    self.args.image_size,
                    self.args.image_size,
                ],
                device=self.args.device,
            )
        else:
            noise = noise.to(self.args.device)

        with torch.no_grad():
            if method == "euler":
                samples = self.flow_matching.euler_sample(noise, n_steps)
            elif method == "midpoint":
                samples = self.flow_matching.midpoint_sample(noise, n_steps)
            else:
                raise ValueError(f"Unknown sampling method: {method}")
        return samples

    def generate_intermediate_samples(
        self,
        method: str = "euler",
        n_samples: int = 4,
        img_size: int = 32,
        steps_to_show=None,
        n_steps: int = 50,
        noise: torch.Tensor = None,
        set_seed: bool = False,
    ):
        if set_seed:
            torch.manual_seed(42)

        if steps_to_show is None:
            steps_to_show = [0, 10, 25, 50]

        if noise is None:
            x = torch.randn(
                n_samples, 1, img_size, img_size, device=self.args.device, requires_grad=False
            )
        else:
            x = noise.to(self.args.device)

        dt = -1.0 / n_steps
        images = [x.detach().cpu().numpy()]

        for step in tqdm(range(1, n_steps + 1, 1)):
            t = 1.0 - (step - 1) / n_steps
            with torch.no_grad():
                if method == "euler":
                    x = self.flow_matching.euler_step(x, t, dt)
                elif method == "midpoint":
                    x = self.flow_matching.midpoint_step(x, t, dt)
                else:
                    raise ValueError(f"Unknown sampling method: {method}")

            if step in steps_to_show:
                images.append(x.detach().cpu().numpy())

        return images

    def benchmark_sampling(self, configs, noise=None, set_seed: bool = False):
        if set_seed:
            torch.manual_seed(42)

        benchmark_rows = []
        base_noise = noise
        if base_noise is None:
            base_noise = torch.randn(
                [
                    self.args.n_samples,
                    self.args.image_channels,
                    self.args.image_size,
                    self.args.image_size,
                ],
                device=self.args.device,
            )

        for method, n_steps in configs:
            run_noise = base_noise.clone()
            start_time = time.perf_counter()
            samples = self.sample(method=method, n_steps=n_steps, noise=run_noise)
            runtime_seconds = time.perf_counter() - start_time
            nfe = n_steps if method == "euler" else 2 * n_steps
            benchmark_rows.append(
                {
                    "method": method,
                    "steps": n_steps,
                    "nfe": nfe,
                    "runtime_seconds": runtime_seconds,
                    "samples": samples.detach().cpu(),
                }
            )

        return benchmark_rows

    def save_model(self):
        model_dir = os.path.dirname(self.args.MODEL_PATH)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.velocity_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.args.MODEL_PATH,
        )

    def show_save(self, img_tensor, show=True, save=True, file_name="sample.png"):
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        assert img_tensor.shape[0] >= 9, "Number of images should be at least 9"
        img_tensor = img_tensor[:9]
        for i, ax in enumerate(axs.flat):
            img = img_tensor[i].squeeze().cpu().numpy()
            ax.imshow(img, cmap="gray")
            ax.axis("off")

        plt.tight_layout()
        if save:
            os.makedirs("images", exist_ok=True)
            plt.savefig(os.path.join("images", file_name))
        if show:
            plt.show()
        plt.close(fig)

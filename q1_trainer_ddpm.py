import copy
import os

import torch
from matplotlib import pyplot as plt
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from ddpm_utils.args import args


torch.manual_seed(42)


def one_param(m):
    return next(iter(m.parameters()))


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
    def __init__(self, args, eps_model, diffusion_model):
        self.eps_model = eps_model.to(args.device)
        self.diffusion = diffusion_model
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=args.learning_rate)
        self.args = args
        self.current_epoch = 0
        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.eps_model).eval().requires_grad_(False)

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
                    loss = self.diffusion.loss(x0)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.ema.step_ema(self.ema_model, self.eps_model)

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
        start_epoch = self.current_epoch
        self.loss_per_iter = []
        for current_epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = current_epoch
            self.train_epoch(dataloader, scaler)

            if current_epoch % self.args.show_every_n_epochs == 0:
                self.sample()

            if (current_epoch + 1) % self.args.save_every_n_epochs == 0:
                self.save_model()

    def sample(self, n_steps=None, noise=None, set_seed: bool = False):
        if set_seed:
            torch.manual_seed(42)
        if n_steps is None:
            n_steps = self.args.n_steps

        with torch.no_grad():
            if noise is None:
                x = torch.randn(
                    [
                        self.args.n_samples,
                        self.args.image_channels,
                        self.args.image_size,
                        self.args.image_size,
                    ],
                    device=self.args.device,
                )
            else:
                x = noise.to(self.args.device)
            if self.args.nb_save is not None:
                saving_steps = [self.args["n_steps"] - 1]
            for t_ in tqdm(range(n_steps)):
                # Reverse DDPM sampling loop.
                # At each iteration, build the current timestep tensor and update x with
                # self.diffusion.p_sample so that x goes from x_t to x_{t-1}.
                # ==========================
                t = n_steps - t_ - 1
                t = torch.full((x.shape[0],), t, device=self.args.device, dtype=torch.long)
                x = self.diffusion.p_sample(x, t)


                if self.args.nb_save is not None and t_ in saving_steps:
                    print(f"Showing/saving samples from epoch {self.current_epoch}")
                    self.show_save(
                        x,
                        show=True,
                        save=True,
                        file_name=f"epoch_{self.current_epoch}_sample_{t_}.png",
                    )
        return x

    def save_model(self):
        model_dir = os.path.dirname(self.args.MODEL_PATH)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.eps_model.state_dict(),
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

    def generate_intermediate_samples(
        self,
        n_samples=4,
        img_size=32,
        steps_to_show=[0, 999],
        n_steps=None,
        noise=None,
        set_seed: bool = False,
    ):
        if set_seed:
            torch.manual_seed(42)

        if n_steps is None:
            n_steps = self.args.n_steps

        if noise is None:
            x = torch.randn(
                n_samples, 1, img_size, img_size, device=self.args.device, requires_grad=False
            )
        else:
            x = noise.to(self.args.device)

        images = []
        images.append(x.detach().cpu().numpy())

        for step in tqdm(range(1, n_steps + 1, 1)):
            # Reverse DDPM loop: compute the current timestep and call
            # self.diffusion.p_sample to update x (same logic as Trainer.sample).
            # ==========================
            t = n_steps - step
            t = torch.full((x.shape[0],), t, device=self.args.device, dtype=torch.long)
            x = self.diffusion.p_sample(x, t)

            if step in steps_to_show:
                images.append(x.detach().cpu().numpy())

        return images

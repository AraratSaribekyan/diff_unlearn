import torch
from torch.nn import MSELoss
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from utils.data import MakeUnlearnLoader

from unet import MNIST_Unet


class MIA:
    def __init__(
            self,
            batch_size: int,
            models: list,
            label: int = 0,
            device: str ="cuda",
            forget: int = 0
    ):
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
        self.device = device
        self.model = MNIST_Unet()
        self.label = label
        self.models = models
        self.label_loader = MakeUnlearnLoader(
            ds_name = "mnist",
            batch_size=batch_size,
            shuffle=True,
            label=label
        )()[forget]
    
    @torch.no_grad()
    def __call__(self):
        losses = {}
        for model in self.models:
            self.model.load_state_dict(torch.load(model))
            self.model.to(self.device)
            tmp_losses = []
            x, y = next(iter(self.label_loader))
            x = x.to(self.device)
            y = y.to(self.device)
            noise = torch.randn_like(x)
            for t in tqdm(range(1000)):
                timesteps = t * torch.ones(x.shape[0]).long().to(self.device)
                noisy_x = self.scheduler.add_noise(x, noise, timesteps)
                pred_noise = self.model(noisy_x, timesteps, y)
                loss = MSELoss()(pred_noise, noise)
                tmp_losses.append(loss.cpu().item())
            losses[model] = tmp_losses

        return losses

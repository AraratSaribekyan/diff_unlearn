import os
import pathlib

import copy
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

ROOT_PATH = pathlib.Path(__file__).parent.parent.resolve()
CKP_PATH = os.path.join(ROOT_PATH, os.path.join("models", "checkpoints"))

class UnlearnLoopModified:
    def __init__(
            self,
            S = 1000,
            K = 10,
            lr = 1e-3,
            lamb = 0.1,
            data_loaders=None,
            model=None,
            ds_name="mnist",
            device="cpu",
            label_c=0,
            save_path = None,
            filters = None,
            img = None
    ):
        if data_loaders is None:
            raise ValueError("No loader is specified")
        if model is None:
            raise ValueError("No model is specified")
        if filters is None:
            raise ValueError("No filters are specified")
        
        self.S = S
        self.K = K
        self.lr = lr
        self.lamb = lamb
        self.forget_loader = data_loaders[0]
        self.remain_loader = data_loaders[1]
        self.model = model
        self.ds_name = ds_name
        self.device = device
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
        self.label_c = label_c
        self.save_path = save_path
        self.up = filters[1]
        self.down = filters[0]
        self.img = img.to(device)

    def __call__(self):
        opt = Adam(self.model.parameters(), lr=self.lr)
        loss_fn = MSELoss()
        losses = []

        self.model.to(self.device)
        self.up.to(self.device)
        self.down.to(self.device)
        
        forget_iterator = iter(self.forget_loader)
        remain_iterator = iter(self.remain_loader)

        for s in tqdm(range(self.S)):
            sub_model = copy.deepcopy(self.model)
            sub_opt = Adam(sub_model.parameters(), lr=self.lr)

            for _ in range(self.K):
                while True:
                    try:
                        x, y = next(forget_iterator)
                        break
                    except StopIteration:
                        forget_iterator = iter(self.forget_loader)
                        continue
                x = x.to(self.device)
                y = y.to(self.device)
                noise = torch.randn_like(x)
                # uniform_noise = torch.rand_like(x)
                timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(self.device)
                noisy_x = self.scheduler.add_noise(x, noise, timesteps)
                noisy_img = self.scheduler.add_noise(self.img.repeat(noisy_x.shape[0], 1, 1, 1), noise, timesteps)
                # noisy_x = self.up(self.down(noisy_img)) + noisy_x - self.up(self.down(noisy_x))
                noisy_x = noisy_img + noisy_x - self.up(self.down(noisy_x))
                pred_noise = sub_model(noisy_x, timesteps, y)
                # loss = loss_fn(pred_noise, uniform_noise)
                loss = loss_fn(pred_noise, noise)
                
                sub_opt.zero_grad()
                loss.backward()
                sub_opt.step()

            loss_cs = loss.item()
            while True:
                try:
                    x, y = next(forget_iterator)
                    break
                except StopIteration:
                    forget_iterator = iter(self.forget_loader)
                    continue
            x = x.to(self.device)
            y = y.to(self.device)
            noise = torch.randn_like(x)
            # uniform_noise = torch.rand_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(self.device)
            noisy_x = self.scheduler.add_noise(x, noise, timesteps)
            noisy_img = self.scheduler.add_noise(self.img.repeat(noisy_x.shape[0], 1, 1, 1), noise, timesteps)
            # noisy_x = self.up(self.down(noisy_img)) + noisy_x - self.up(self.down(noisy_x))
            noisy_x = noisy_img + noisy_x - self.up(self.down(noisy_x))
            pred_noise = self.model(noisy_x, timesteps, y)
            # loss_f = loss_fn(pred_noise, uniform_noise) - loss_cs
            loss_f = loss_fn(pred_noise, noise) - loss_cs

            while True:
                try:
                    x, y = next(remain_iterator)
                    break
                except StopIteration:
                    remain_iterator = iter(self.remain_loader)
                    continue
            x = x.to(self.device)
            y = y.to(self.device)
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(self.device)
            noisy_x = self.scheduler.add_noise(x, noise, timesteps)
            pred_noise = self.model(noisy_x, timesteps, y)
            loss = loss_fn(noise, pred_noise) + self.lamb*loss_f
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

            if s%50 == 0:
                avg_loss = sum(losses[-50:])/50
                print(f"Step {s}. Avarage of the last 50 loss values: {avg_loss:05f}")

        plt.plot(losses)
        torch.save(self.model.state_dict(), self.save_path)
        
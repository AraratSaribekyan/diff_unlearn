import os
import pathlib

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

ROOT_PATH = pathlib.Path(__file__).parent.parent.resolve()
CKP_PATH = os.path.join(ROOT_PATH, os.path.join("models", "checkpoints"))


class TrainLoop:
    def __init__(self, epochs=10, data_loader=None, resume=False, model=None, ds_name="mnist", device="cpu"):
        if data_loader is None:
            raise ValueError("No loader is specified")
        if model is None:
            raise ValueError("No model is specified")
    
        self.loader = data_loader
        self.resume = resume
        self.model = model
        self.ds_name = ds_name
        self.epochs = epochs
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
        self.device = device

    def __call__(self):
        ckpts = os.listdir(CKP_PATH)
        ckpt_file = self.ds_name+".pt"

        if self.resume:
            if ckpt_file in ckpts:
                self.model.load_state_dict(torch.load(os.path.join(CKP_PATH, ckpt_file)))
        
        opt = Adam(self.model.parameters(), lr=1e-3)
        loss_fn = MSELoss()
        losses = []

        self.model.to(self.device)
        for epoch in range(self.epochs):
            print(f"EPOCH: {epoch}")

            for x,y in tqdm(self.loader):
                x = x.to(self.device)
                y = y.to(self.device)
                noise = torch.randn_like(x)
                timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(self.device)
                noisy_x = self.scheduler.add_noise(x, noise, timesteps)

                pred_noise = self.model(noisy_x, timesteps, y)

                loss = loss_fn(pred_noise, noise)

                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(loss.item())

            avg_loss = sum(losses[-100:])/100
            print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}') 

        plt.plot(losses)

        torch.save(self.model.state_dict(), os.path.join(CKP_PATH, ckpt_file))

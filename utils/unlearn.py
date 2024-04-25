import os
import torch

from unet import MNIST_Unet
from .data import MakeUnlearnLoader
from .loops import UnlearnLoop


class UnlearnWrapper:
    def __init__(
            self,
            weights,
            ds_name,
            batch_size,
            shuffle,
            label,
            S,
            K,
            lr,
            lamb,
            device,
            save_path
    ):
        model = MNIST_Unet()
        model.load_state_dict(torch.load(weights))
        
        forget_loader, remain_loader = MakeUnlearnLoader(
            ds_name=ds_name,
            batch_size=batch_size,
            shuffle=shuffle,
            label=label
        )()

        self.loop = UnlearnLoop(
            data_loaders=(forget_loader, remain_loader),
            model=model,
            label_c=label,
            device=device,
            S=S,
            K=K,
            lr=lr,
            lamb=lamb,
            save_path=save_path
        )
    
    def __call__(self):
        self.loop()
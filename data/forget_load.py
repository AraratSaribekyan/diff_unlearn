import pathlib
import torch

from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import Sampler
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from matplotlib import pyplot as plt

AVAILABLE_DS = ["mnist"]
CUR_ABS_PATH = pathlib.Path(__file__).parent.resolve()


class MakeUnlearnLoader:
    def __init__(self, ds_name=None, batch_size=8, shuffle=True, label=None):
        if ds_name is None:
            raise ValueError("No dataset name was passed")
        if ds_name not in AVAILABLE_DS:
            raise ValueError(f"Invalid dataset name {ds_name}")
        if label is None:
            raise ValueError("No label was passed")
        
        self.ds_name = ds_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.label = label

    def __call__(self):
        if self.ds_name == "mnist":
            dataset = MNIST(root=CUR_ABS_PATH, train=True, download=True, transform=ToTensor())
            I_f = []
            I_r = []
            for i, label in enumerate(dataset.targets):
                if label == self.label:
                    I_f.append(i)
                else:
                    I_r.append(i)
            D_f = Subset(dataset, I_f)
            D_r = Subset(dataset, I_r)

        forget_loader = DataLoader(D_f, batch_size=self.batch_size, shuffle=True)
        remain_loader = DataLoader(D_r, batch_size=self.batch_size, shuffle=True)

        _, axs = plt.subplots(1,2, figsize=(10,6))
        imgs, _ = next(iter(forget_loader))
        axs[0].set_title("Forget set")
        axs[0].imshow(make_grid(imgs)[0], cmap="Greys")

        imgs, _ = next(iter(remain_loader))
        axs[1].set_title("Remain set")
        axs[1].imshow(make_grid(imgs)[0], cmap="Greys")


        return forget_loader, remain_loader

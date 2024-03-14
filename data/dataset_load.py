import pathlib

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from matplotlib import pyplot as plt

AVAILABLE_DS = ["mnist"]
CUR_ABS_PATH = pathlib.Path(__file__).parent.resolve()

class MakeDataLoader:
    def __init__(self, ds_name=None, batch_size=8, shuffle=True):
        if ds_name is None:
            raise ValueError("No dataset name was passed")
        if ds_name not in AVAILABLE_DS:
            raise ValueError(f"Invalid dataset name {ds_name}")
        
        self.ds_name = ds_name
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __call__(self):
        if self.ds_name == "mnist":
            dataset = MNIST(root=CUR_ABS_PATH, train=True, download=True, transform=ToTensor())
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

            imgs, lbls = next(iter(dataloader))
            print("Sample size: ", imgs.shape)
            print("Labels size: ", lbls)

            plt.imshow(make_grid(imgs)[0], cmap="Greys")

            return dataloader
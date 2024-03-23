import torch
from utils import Filter

DISTR_TYPES = ["uniform", "filtered"]


class distributionSampler:
    def __init__(self, distr_type, sample, device="cpu"):
        if distr_type not in DISTR_TYPES:
            raise ValueError(f"Invalid distribution type {distr_type}. Should be out of {DISTR_TYPES}")
        
        self.distr_type = distr_type
        if distr_type == "filtered":
            b, c, h, w = sample.shape
            if h!=w:
                raise ValueError("Image must be of square shape")
            if h%2 != 0:
                raise ValueError(f"Image size should be multiple to 2. Expected 2x got {h}")
            
            scale = 1
            size = h
            while size%2 == 0 and size!=2:
                size /= 2
                scale *= 2
            
            shape_down = (b,c,h,w)
            shape_up = (b,c,h//scale,w//scale)

            self.down = Filter(shape_down, 1/scale).to(device)
            self.up = Filter(shape_up, scale).to(device)
        print(self.distr_type)

    def __call__(self, noise):
        if self.distr_type == "uniform":
            return torch.rand_like(noise)
        if self.distr_type == "filtered":
            return self.up(self.down(noise))
            
            

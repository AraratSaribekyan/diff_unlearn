from data import MakeDataLoader
from models import MNIST_Unet
from loops import TrainLoop

class TrainWrapper:
    def __init__(
            self,
            epochs,
            ds_name,
            device,
            resume,
            batch_size,
            shuffle,
            model_save
    ):
        loader = MakeDataLoader(
            ds_name=ds_name,
            batch_size=batch_size,
            shuffle=shuffle
        )()
        model = MNIST_Unet()
        self.train = TrainLoop(
            epochs=epochs,
            data_loader=loader,
            resume=resume,
            model=model,
            ds_name=ds_name,
            device=device,
            model_save=model_save
        )
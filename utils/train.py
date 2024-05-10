from unet import MNIST_Unet
from .data import MakeDataLoader, MakeUnlearnLoader
from .loops import TrainLoop

class TrainWrapper:
    def __init__(
            self,
            epochs,
            ds_name,
            device,
            resume,
            batch_size,
            shuffle,
            model_save,
            unlearn_label = None
    ):
        if unlearn_label is None:
            loader = MakeDataLoader(
                ds_name=ds_name,
                batch_size=batch_size,
                shuffle=shuffle
            )()
        else:
            loader = MakeUnlearnLoader(
                ds_name = ds_name,
                batch_size=batch_size,
                shuffle=shuffle,
                label=unlearn_label
            )()[1]
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
from typing import Any, Iterable
import torch

import albumentations as A
import kornia.augmentation as K  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential
from terratorch.datasets import MNeonTreeNonGeo
from terratorch.datamodules.utils import wrap_in_compose_is_list


MEANS = {
    "BLUE": 108.44846666666666,
    "CANOPY_HEIGHT_MODEL": 9.886005401611328,
    "GREEN": 131.00455555555556,
    "NEON": 1130.868248148148,
    "RED": 122.29733333333333
}

STDS =  {
    "BLUE": 33.09221632578563,
    "CANOPY_HEIGHT_MODEL": 7.75406551361084,
    "GREEN": 51.442224204429245,
    "NEON": 1285.299937507298,
    "RED": 54.053087350753124
}

class MNeonTreeNonGeoDataModule(NonGeoDataModule):
    def __init__(
        self, 
        batch_size: int = 8, 
        num_workers: int = 0, 
        data_root: str = "./",
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        aug: AugmentationSequential = None,
        **kwargs: Any
    ) -> None:

        super().__init__(MNeonTreeNonGeo, batch_size, num_workers, **kwargs)
        
        bands = kwargs.get("bands", MNeonTreeNonGeo.all_band_names)
        self.means = torch.tensor([MEANS[b] for b in bands])
        self.stds = torch.tensor([STDS[b] for b in bands])
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.data_root = data_root
        self.aug = AugmentationSequential(K.Normalize(self.means, self.stds), data_keys=["image", "mask"]) if aug is None else aug
    
    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(  
                split="train", data_root=self.data_root, transform=self.train_transform, **self.kwargs
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val", data_root=self.data_root, transform=self.val_transform, **self.kwargs
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test",data_root=self.data_root, transform=self.test_transform, **self.kwargs
            )


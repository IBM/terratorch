from typing import Any
import torch

import albumentations as A
import kornia.augmentation as K  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential
from terratorch.datasets import MBigEarthNonGeo
from terratorch.datamodules.utils import wrap_in_compose_is_list

MEANS = {
    "COASTAL_AEROSOL": 378.4027,
    "BLUE": 482.2730,
    "GREEN": 706.5345,
    "RED": 720.9285,
    "RED_EDGE_1": 1100.6688,
    "RED_EDGE_2": 1909.2914,
    "RED_EDGE_3": 2191.6985,
    "NIR_BROAD": 2336.8706,
    "NIR_NARROW": 2394.7449,
    "WATER_VAPOR": 2368.3127,
    "SWIR_1": 1875.2487,
    "SWIR_2": 1229.3818,
}

STDS = {
    "COASTAL_AEROSOL": 157.5666,
    "BLUE": 255.0429,
    "GREEN": 303.1750,
    "RED": 391.2943,
    "RED_EDGE_1": 380.7916,
    "RED_EDGE_2": 551.6558,
    "RED_EDGE_3": 638.8196,
    "NIR_BROAD": 744.2009,
    "NIR_NARROW": 675.4041,
    "WATER_VAPOR": 561.0154,
    "SWIR_1": 563.4095,
    "SWIR_2": 479.1786,
}


class MBigEarthNonGeoDataModule(NonGeoDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        data_root: str = "./",
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        aug: AugmentationSequential = None,
        **kwargs: Any,
    ) -> None:

        super().__init__(MBigEarthNonGeo, batch_size, num_workers, **kwargs)

        bands = kwargs.get("bands", MBigEarthNonGeo.all_band_names)
        self.means = torch.tensor([MEANS[b] for b in bands])
        self.stds = torch.tensor([STDS[b] for b in bands])
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.data_root = data_root
        self.aug = (
            AugmentationSequential(K.Normalize(self.means, self.stds), data_keys=["image"]) if aug is None else aug
        )

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
                split="test", data_root=self.data_root, transform=self.test_transform, **self.kwargs
            )

from typing import Any

import albumentations as A
from torchgeo.transforms import AugmentationSequential

from terratorch.datamodules.geobench_data_module import GeobenchDataModule
from terratorch.datasets import MPv4gerNonGeo

MEANS = {"BLUE": 116.628328, "GREEN": 119.65935, "RED": 113.385309}

STDS = {
    "BLUE": 44.668890717415586,
    "GREEN": 48.282311849967364,
    "RED": 54.19692448815262,
}


class MPv4gerNonGeoDataModule(GeobenchDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        data_root: str = "./",
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        aug: AugmentationSequential = None,
        partition: str = "default",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            MPv4gerNonGeo,
            MEANS,
            STDS,
            batch_size=batch_size,
            num_workers=num_workers,
            data_root=data_root,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            aug=aug,
            partition=partition,
            **kwargs,
        )

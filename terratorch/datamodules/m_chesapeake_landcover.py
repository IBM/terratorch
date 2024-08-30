from typing import Any

import albumentations as A
from torchgeo.transforms import AugmentationSequential

from terratorch.datamodules.geobench_data_module import GeobenchDataModule
from terratorch.datasets import MChesapeakeLandcoverNonGeo

MEANS = {"BLUE": 0.4807923436164856, "GREEN": 0.5200885534286499, "NIR": 0.569856584072113, "RED": 0.4570387601852417}

STDS = {"BLUE": 0.17441707849502563, "GREEN": 0.1976749747991562, "NIR": 0.2831788957118988, "RED": 0.21191735565662384}


class MChesapeakeLandcoverNonGeoDataModule(GeobenchDataModule):
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
            MChesapeakeLandcoverNonGeo,
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

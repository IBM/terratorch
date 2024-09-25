from typing import Any, Iterable

import albumentations as A
import kornia.augmentation as K  # noqa: N812
import torch
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential

from terratorch.datamodules.geobench_data_module import GeobenchDataModule
from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import MForestNetNonGeo

MEANS = {
    "BLUE": 72.852258,
    "GREEN": 83.677155,
    "RED": 77.58181,
    "NIR": 123.987442,
    "SWIR_1": 91.536942,
    "SWIR_2": 74.719202,
}

STDS = {
    "BLUE": 15.837172547567825,
    "GREEN": 14.788812599596188,
    "RED": 16.100543441881086,
    "NIR": 16.35234883118129,
    "SWIR_1": 13.7882739778638,
    "SWIR_2": 12.69131413539181,
}


class MForestNetNonGeoDataModule(GeobenchDataModule):
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
        use_metadata: bool = False,  # noqa: FBT002, FBT001
        **kwargs: Any,
    ) -> None:
        super().__init__(
            MForestNetNonGeo,
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
            use_metadata=use_metadata,
            **kwargs,
        )

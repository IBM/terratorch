from typing import Any

import albumentations as A
from torchgeo.transforms import AugmentationSequential

from terratorch.datamodules.geobench_data_module import GeobenchDataModule
from terratorch.datasets import MPv4gerSegNonGeo

MEANS = {"BLUE": 139.761751, "GREEN": 137.354091, "RED": 131.102356}

STDS = {"BLUE": 48.29800594656056, "GREEN": 50.86544377633718, "RED": 54.52768048660482}


class MPv4gerSegNonGeoDataModule(GeobenchDataModule):
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
            MPv4gerSegNonGeo,
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

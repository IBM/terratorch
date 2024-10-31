from collections.abc import Sequence
from typing import Any

import albumentations as A
from torchgeo.transforms import AugmentationSequential

from terratorch.datamodules.geobench_data_module import GeobenchDataModule
from terratorch.datasets import MNzCattleNonGeo

MEANS = {"BLUE": 106.51769083969465, "GREEN": 130.09102671755724, "RED": 126.31354389312978}

STDS = {"BLUE": 18.991624007810348, "GREEN": 19.92435830412983, "RED": 23.38495539231075}


class MNzCattleNonGeoDataModule(GeobenchDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        data_root: str = "./",
        bands: Sequence[str] | None = None,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        aug: AugmentationSequential = None,
        partition: str = "default",
        use_metadata: bool = False,  # noqa: FBT002, FBT001
        **kwargs: Any,
    ) -> None:
        super().__init__(
            MNzCattleNonGeo,
            MEANS,
            STDS,
            batch_size=batch_size,
            num_workers=num_workers,
            data_root=data_root,
            bands=bands,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            aug=aug,
            partition=partition,
            use_metadata=use_metadata,
            **kwargs,
        )

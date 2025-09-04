from collections.abc import Sequence
from typing import Any

import albumentations as A
from kornia.augmentation import AugmentationSequential

from terratorch.datamodules.geobench_data_module import GeobenchDataModule
from terratorch.datasets import MBigEarthNonGeo

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


class MBigEarthNonGeoDataModule(GeobenchDataModule):
    """NonGeo LightningDataModule implementation for M-BigEarthNet dataset."""

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
        **kwargs: Any,
    ) -> None:
        """
        Initializes the MBigEarthNonGeoDataModule for the M-BigEarthNet dataset.

        Args:
            batch_size (int, optional): Batch size for DataLoaders. Defaults to 8.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
            data_root (str, optional): Root directory of the dataset. Defaults to "./".
            bands (Sequence[str] | None, optional): List of bands to use. Defaults to None.
            train_transform (A.Compose | None | list[A.BasicTransform], optional): Transformations for training.
            val_transform (A.Compose | None | list[A.BasicTransform], optional): Transformations for validation.
            test_transform (A.Compose | None | list[A.BasicTransform], optional): Transformations for testing.
            aug (AugmentationSequential, optional): Augmentation/normalization pipeline. Defaults to None.
            partition (str, optional): Partition size. Defaults to "default".
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(
            MBigEarthNonGeo,
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
            **kwargs,
        )

from collections.abc import Sequence
from typing import Any

import albumentations as A
from kornia.augmentation import AugmentationSequential

from terratorch.datamodules.geobench_data_module import GeobenchDataModule
from terratorch.datasets import MEuroSATNonGeo

MEANS = {
    "COASTAL_AEROSOL": 1355.5426,
    "BLUE": 1113.8855,
    "GREEN": 1035.7394,
    "RED": 928.2619,
    "RED_EDGE_1": 1188.2629,
    "RED_EDGE_2": 2032.7325,
    "RED_EDGE_3": 2416.5286,
    "NIR_BROAD": 2342.5396,
    "NIR_NARROW": 748.9036,
    "WATER_VAPOR": 12.0419,
    "CIRRUS": 1810.1284,
    "SWIR_1": 1101.3801,
    "SWIR_2": 2644.5996,
}

STDS = {
    "COASTAL_AEROSOL": 68.9288,
    "BLUE": 160.0012,
    "GREEN": 194.6687,
    "RED": 286.8012,
    "RED_EDGE_1": 236.6991,
    "RED_EDGE_2": 372.3853,
    "RED_EDGE_3": 478.1329,
    "NIR_BROAD": 556.7527,
    "NIR_NARROW": 102.5583,
    "WATER_VAPOR": 1.2167,
    "CIRRUS": 392.9388,
    "SWIR_1": 313.7339,
    "SWIR_2": 526.7788,
}


class MEuroSATNonGeoDataModule(GeobenchDataModule):
    """NonGeo LightningDataModule implementation for M-EuroSAT dataset."""

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
        Initializes the MEuroSATNonGeoDataModule for the MEuroSATNonGeo dataset.

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
            MEuroSATNonGeo,
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

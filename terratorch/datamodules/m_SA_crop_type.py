from collections.abc import Sequence
from typing import Any

import albumentations as A
from kornia.augmentation import AugmentationSequential

from terratorch.datamodules.geobench_data_module import GeobenchDataModule
from terratorch.datasets import MSACropTypeNonGeo

MEANS = {
    "COASTAL_AEROSOL": 12.739611,
    "BLUE": 16.526744,
    "GREEN": 26.636417,
    "RED": 36.696639,
    "RED_EDGE_1": 46.388679,
    "RED_EDGE_2": 58.281453,
    "RED_EDGE_3": 63.575819,
    "NIR_BROAD": 68.1836,
    "NIR_NARROW": 69.142591,
    "WATER_VAPOR": 69.904566,
    "SWIR_1": 83.626811,
    "SWIR_2": 65.767679,
    "CLOUD_PROBABILITY": 0.0,
}

STDS = {
    "COASTAL_AEROSOL": 7.492811526301659,
    "BLUE": 9.329547939662671,
    "GREEN": 12.674537246073758,
    "RED": 19.421922023931593,
    "RED_EDGE_1": 19.487411106531287,
    "RED_EDGE_2": 19.959174612412983,
    "RED_EDGE_3": 21.53805760692545,
    "NIR_BROAD": 23.05077775347288,
    "NIR_NARROW": 22.329695761624677,
    "WATER_VAPOR": 21.877766438821954,
    "SWIR_1": 28.14418826277069,
    "SWIR_2": 27.2346215312965,
    "CLOUD_PROBABILITY": 0.0,
}


class MSACropTypeNonGeoDataModule(GeobenchDataModule):
    """NonGeo LightningDataModule implementation for M-SA-CropType dataset."""

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
        Initializes the MSACropTypeNonGeoDataModule for the MSACropTypeNonGeo dataset.

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
            MSACropTypeNonGeo,
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

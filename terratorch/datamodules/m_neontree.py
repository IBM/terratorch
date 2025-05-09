from collections.abc import Sequence
from typing import Any

import albumentations as A
from kornia.augmentation import AugmentationSequential

from terratorch.datamodules.geobench_data_module import GeobenchDataModule
from terratorch.datasets import MNeonTreeNonGeo

MEANS = {
    "BLUE": 108.44846666666666,
    "CANOPY_HEIGHT_MODEL": 9.886005401611328,
    "GREEN": 131.00455555555556,
    "NEON": 1130.868248148148,
    "RED": 122.29733333333333,
}

STDS = {
    "BLUE": 33.09221632578563,
    "CANOPY_HEIGHT_MODEL": 7.75406551361084,
    "GREEN": 51.442224204429245,
    "NEON": 1285.299937507298,
    "RED": 54.053087350753124,
}


class MNeonTreeNonGeoDataModule(GeobenchDataModule):
    """NonGeo LightningDataModule implementation for M-NeonTree dataset."""

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
        Initializes the MNeonTreeNonGeoDataModule for the MNeonTreeNonGeo dataset.

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
            MNeonTreeNonGeo,
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

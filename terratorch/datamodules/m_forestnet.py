from collections.abc import Sequence
from typing import Any

import albumentations as A
from kornia.augmentation import AugmentationSequential

from terratorch.datamodules.geobench_data_module import GeobenchDataModule
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
    """NonGeo LightningDataModule implementation for M-ForestNet dataset."""

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
        """
        Initializes the MForestNetNonGeoDataModule for the MForestNetNonGeo dataset.

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
            use_metadata (bool): Whether to return metadata info.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(
            MForestNetNonGeo,
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

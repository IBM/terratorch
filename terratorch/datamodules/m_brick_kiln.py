from collections.abc import Sequence
from typing import Any

import albumentations as A
from kornia.augmentation import AugmentationSequential

from terratorch.datamodules.geobench_data_module import GeobenchDataModule
from terratorch.datasets import MBrickKilnNonGeo

MEANS = {
    "COASTAL_AEROSOL": 574.7587880700896,
    "BLUE": 674.3473615470523,
    "GREEN": 886.3656479311578,
    "RED": 815.0945462528913,
    "RED_EDGE_1": 1128.8088426870465,
    "RED_EDGE_2": 1934.450471876027,
    "RED_EDGE_3": 2045.7652282437202,
    "NIR_BROAD": 2012.744587807115,
    "NIR_NARROW": 1608.6255233989034,
    "WATER_VAPOR": 1129.8171906000355,
    "CIRRUS": 83.27188605598549,
    "SWIR_1": 90.54924599052214,
    "SWIR_2": 68.98768652434848,
}

STDS = {
    "COASTAL_AEROSOL": 193.60631504991184,
    "BLUE": 238.75447480113132,
    "GREEN": 276.9631260242207,
    "RED": 361.15060137326634,
    "RED_EDGE_1": 364.5888078793488,
    "RED_EDGE_2": 724.2707123576525,
    "RED_EDGE_3": 819.653063972575,
    "NIR_BROAD": 794.3652427593881,
    "NIR_NARROW": 800.8538290702304,
    "WATER_VAPOR": 704.0219637458916,
    "CIRRUS": 36.355745901131705,
    "SWIR_1": 28.004671947623894,
    "SWIR_2": 24.268892726362033,
}


class MBrickKilnNonGeoDataModule(GeobenchDataModule):
    """NonGeo LightningDataModule implementation for M-BrickKiln dataset."""

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
        Initializes the MBrickKilnNonGeoDataModule for the M-BrickKilnNonGeo dataset.

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
            MBrickKilnNonGeo,
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

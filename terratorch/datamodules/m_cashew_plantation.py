from collections.abc import Sequence
from typing import Any

import albumentations as A
from kornia.augmentation import AugmentationSequential

from terratorch.datamodules.geobench_data_module import GeobenchDataModule
from terratorch.datasets import MBeninSmallHolderCashewsNonGeo

MEANS = {
    "COASTAL_AEROSOL": 520.1185302734375,
    "BLUE": 634.7583618164062,
    "GREEN": 892.461181640625,
    "RED": 880.7075805664062,
    "RED_EDGE_1": 1380.6409912109375,
    "RED_EDGE_2": 2233.432373046875,
    "RED_EDGE_3": 2549.379638671875,
    "NIR_BROAD": 2643.248046875,
    "NIR_NARROW": 2643.531982421875,
    "WATER_VAPOR": 2852.87451171875,
    "SWIR_1": 2463.933349609375,
    "SWIR_2": 1600.9207763671875,
    "CLOUD_PROBABILITY": 0.010281000286340714,
}

STDS = {
    "COASTAL_AEROSOL": 204.2023468017578,
    "BLUE": 227.25344848632812,
    "GREEN": 222.32545471191406,
    "RED": 350.47235107421875,
    "RED_EDGE_1": 280.6436767578125,
    "RED_EDGE_2": 373.7521057128906,
    "RED_EDGE_3": 449.9236145019531,
    "NIR_BROAD": 414.6498107910156,
    "NIR_NARROW": 415.1019592285156,
    "WATER_VAPOR": 413.8980407714844,
    "SWIR_1": 494.97430419921875,
    "SWIR_2": 514.4229736328125,
    "CLOUD_PROBABILITY": 0.3447800576686859,
}


class MBeninSmallHolderCashewsNonGeoDataModule(GeobenchDataModule):
    """NonGeo LightningDataModule implementation for M-Cashew Plantation dataset."""

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
        Initializes the MBeninSmallHolderCashewsNonGeoDataModule for the M-BeninSmallHolderCashewsNonGeo dataset.

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
            MBeninSmallHolderCashewsNonGeo,
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

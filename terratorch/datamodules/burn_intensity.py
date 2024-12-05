from collections.abc import Sequence
from typing import Any

import albumentations as A

from terratorch.datamodules.utils import NormalizeWithTimesteps, wrap_in_compose_is_list
from terratorch.datasets import BurnIntensityNonGeo
from torchgeo.datamodules import NonGeoDataModule

MEANS = {
    "BLUE": [331.6921, 896.8024, 348.8031],
    "GREEN": [555.1077, 1093.9736, 500.2181],
    "RED": [605.2513, 1142.7225, 597.9034],
    "NIR": [1761.3884, 1890.2156, 1552.0403],
    "SWIR_1": [1117.1825, 1408.0839, 1293.0919],
    "SWIR_2": [2168.0090, 2270.9753, 1362.1312],
}

STDS = {
    "BLUE": [213.0656, 1620.4131, 314.7517],
    "GREEN": [273.0910, 1628.4181, 365.6746],
    "RED": [414.8322, 1600.7698, 424.8185],
    "NIR": [818.7486, 1236.8453, 804.9058],
    "SWIR_1": [677.2739, 1153.7432, 795.4156],
    "SWIR_2": [612.9131, 1495.8365, 661.6196],
}

class BurnIntensityNonGeoDataModule(NonGeoDataModule):
    """NonGeo datamodule implementation for BurnIntensity."""

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 0,
        bands: Sequence[str] = BurnIntensityNonGeo.all_band_names,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        use_full_data: bool = True,
        no_data_replace: float | None = 0.0001,
        no_label_replace: int | None = -1,
        use_metadata: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(BurnIntensityNonGeo, batch_size, num_workers, **kwargs)
        self.data_root = data_root

        means = [MEANS[b] for b in bands]
        stds = [STDS[b] for b in bands]
        self.bands = bands
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.aug = NormalizeWithTimesteps(means, stds)
        self.use_full_data = use_full_data
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.use_metadata = use_metadata

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train",
                data_root=self.data_root,
                transform=self.train_transform,
                bands=self.bands,
                use_full_data=self.use_full_data,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                use_metadata=self.use_metadata,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                transform=self.val_transform,
                bands=self.bands,
                use_full_data=self.use_full_data,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                use_metadata=self.use_metadata,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                transform=self.test_transform,
                bands=self.bands,
                use_full_data=self.use_full_data,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                use_metadata=self.use_metadata,
            )

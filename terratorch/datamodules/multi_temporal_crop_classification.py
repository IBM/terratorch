from collections.abc import Sequence
from typing import Any

import albumentations as A
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule

from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import MultiTemporalCropClassification

MEANS_PER_VERSION = {
    '1': {
        "BLUE": 830.5397,
        "GREEN": 2427.1667,
        "RED": 760.6795,
        "NIR_NARROW": 2575.2020,
        "SWIR_1": 649.9128,
        "SWIR_2": 2344.4357,
    },
    '2': {
        "BLUE": 829.5907,
        "GREEN": 2437.3473,
        "RED": 748.6308,
        "NIR_NARROW": 2568.9369,
        "SWIR_1": 638.9926,
        "SWIR_2": 2336.4087,
    }
}

STDS_PER_VERSION = {
    '1': {
        "BLUE": 447.9155,
        "GREEN": 910.8289,
        "RED": 490.9398,
        "NIR_NARROW": 1142.5207,
        "SWIR_1": 430.9440,
        "SWIR_2": 1094.0881,
    },
    '2': {
        "BLUE": 447.1192,
        "GREEN": 913.5633,
        "RED": 480.5570,
        "NIR_NARROW": 1140.6160,
        "SWIR_1": 418.6212,
        "SWIR_2": 1091.6073,
    }
}


class MultiTemporalCropClassificationDataModule(NonGeoDataModule):
    """NonGeo datamodule implementation for multi-temporal crop classification."""

    def __init__(
        self,
        data_root: str,
        version: str = '2',
        batch_size: int = 4,
        num_workers: int = 0,
        bands: Sequence[str] = MultiTemporalCropClassification.all_band_names,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        drop_last: bool = True,
        no_data_replace: float | None = 0,
        no_label_replace: int | None = -1,
        expand_temporal_dimension: bool = True,
        reduce_zero_label: bool = True,
        use_metadata: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(MultiTemporalCropClassification, batch_size, num_workers, **kwargs)
        self.data_root = data_root
        means = MEANS_PER_VERSION[version]
        stds = STDS_PER_VERSION[version]    
        self.means = [means[b] for b in bands]
        self.stds = [stds[b] for b in bands]
        self.version = version
        self.bands = bands
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.aug = Normalize(self.means, self.stds)
        self.drop_last = drop_last
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.expand_temporal_dimension = expand_temporal_dimension
        self.reduce_zero_label = reduce_zero_label
        self.use_metadata = use_metadata

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train",
                version=self.version,
                data_root=self.data_root,
                transform=self.train_transform,
                bands=self.bands,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension = self.expand_temporal_dimension,
                reduce_zero_label = self.reduce_zero_label,
                use_metadata=self.use_metadata,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val",
                version=self.version,
                data_root=self.data_root,
                transform=self.val_transform,
                bands=self.bands,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension = self.expand_temporal_dimension,
                reduce_zero_label = self.reduce_zero_label,
                use_metadata=self.use_metadata,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test",
                version=self.version,
                data_root=self.data_root,
                transform=self.test_transform,
                bands=self.bands,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension = self.expand_temporal_dimension,
                reduce_zero_label = self.reduce_zero_label,
                use_metadata=self.use_metadata,
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
        )

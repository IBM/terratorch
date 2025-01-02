# Copyright contributors to the Terratorch project

import os
from collections.abc import Sequence
from typing import Any

import albumentations as A
import kornia.augmentation as K  # noqa: N812
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import GeoDataModule, NonGeoDataModule
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler
from torchgeo.transforms import AugmentationSequential

from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import FireScarsHLS, FireScarsNonGeo, FireScarsSegmentationMask

MEANS_PER_VERSION = {
    '1': {
        "BLUE": 0.0535,
        "GREEN": 0.0788,
        "RED": 0.0963,
        "NIR_NARROW": 0.2119,
        "SWIR_1": 0.2360,
        "SWIR_2": 0.1731,
    },
    '2': {
        "BLUE": 0.0535,
        "GREEN": 0.0788,
        "RED": 0.0963,
        "NIR_NARROW": 0.2119,
        "SWIR_1": 0.2360,
        "SWIR_2": 0.1731,
    }
}

STDS_PER_VERSION = {
    '1': {
        "BLUE": 0.0308,
        "GREEN": 0.0378,
        "RED": 0.0550,
        "NIR_NARROW": 0.0707,
        "SWIR_1": 0.0919,
        "SWIR_2": 0.0841,
    },
    '2': {
        "BLUE": 0.0308,
        "GREEN": 0.0378,
        "RED": 0.0550,
        "NIR_NARROW": 0.0707,
        "SWIR_1": 0.0919,
        "SWIR_2": 0.0841,
    }
}


class FireScarsNonGeoDataModule(NonGeoDataModule):
    """NonGeo datamodule implementation for Fire Scars"""

    def __init__(
        self,
        data_root: str,
        version: str = '2',
        batch_size: int = 4,
        num_workers: int = 0,
        bands: Sequence[str] = FireScarsNonGeo.all_band_names,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        drop_last: bool = True,
        no_data_replace: float | None = 0,
        no_label_replace: int | None = -1,
        use_metadata: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(FireScarsNonGeo, batch_size, num_workers, **kwargs)
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
        self.aug = AugmentationSequential(K.Normalize(self.means, self.stds), data_keys=["image"])
        self.drop_last = drop_last
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
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


class FireScarsDataModule(GeoDataModule):
    """Geo Fire Scars data module implementation that merges input data with ground truth segmentation masks."""

    def __init__(self, data_root: str, **kwargs: Any) -> None:
        super().__init__(FireScarsSegmentationMask, 4, 224, 100, 0, **kwargs)
        means = list(MEANS.values())
        stds = list(STDS.values())
        self.train_aug = AugmentationSequential(K.RandomCrop(224, 224), K.Normalize(means, stds))
        self.aug = AugmentationSequential(K.Normalize(means, stds))
        self.data_root = data_root

    def setup(self, stage: str) -> None:
        self.images = FireScarsHLS(
            os.path.join(self.data_root, "training/")
        )
        self.labels = FireScarsSegmentationMask(
            os.path.join(self.data_root, "training/")
        )
        self.dataset = self.images & self.labels
        self.train_aug = AugmentationSequential(K.RandomCrop(224, 224), K.normalize())

        self.images_test = FireScarsHLS(
            os.path.join(self.data_root, "validation/")
        )
        self.labels_test = FireScarsSegmentationMask(
            os.path.join(self.data_root, "validation/")
        )
        self.val_dataset = self.images_test & self.labels_test

        if stage in ["fit"]:
            self.train_batch_sampler = RandomBatchGeoSampler(self.dataset, self.patch_size, self.batch_size, None)
        if stage in ["fit", "validate"]:
            self.val_sampler = GridGeoSampler(self.val_dataset, self.patch_size, self.patch_size)
        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(self.val_dataset, self.patch_size, self.patch_size)

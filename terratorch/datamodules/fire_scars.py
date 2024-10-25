# Copyright contributors to the Terratorch project

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


MEANS = {
    "BLUE": 0.033349706741586264,
    "GREEN": 0.05701185520536176,
    "RED": 0.05889748132001316,
    "NIR_NARROW": 0.2323245113436119,
    "SWIR_1": 0.1972854853760658,
    "SWIR_2": 0.11944914225186566,
}

STDS = {
    "BLUE": 0.02269135568823774,
    "GREEN": 0.026807560223070237,
    "RED": 0.04004109844362779,
    "NIR_NARROW": 0.07791732423672691,
    "SWIR_1": 0.08708738838140137,
    "SWIR_2": 0.07241979477437814,
}


class FireScarsNonGeoDataModule(NonGeoDataModule):
    """NonGeo datamodule implementation for Fire Scars"""

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 0,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        drop_last: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(FireScarsNonGeo, batch_size, num_workers, **kwargs)
        self.data_root = data_root

        bands = kwargs.get("bands", FireScarsNonGeo.all_band_names)
        means = [MEANS[b] for b in bands]
        stds = [STDS[b] for b in bands]

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.aug = AugmentationSequential(K.Normalize(means, stds), data_keys=["image"])
        self.drop_last = drop_last

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train",
                data_root=self.data_root,
                transform=self.train_transform,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                transform=self.val_transform,
                **self.kwargs,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                transform=self.test_transform,
                **self.kwargs,
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(FireScarsSegmentationMask, 4, 224, 100, 0, **kwargs)
        means = list(MEANS.values())
        stds = list(STDS.values())
        self.train_aug = AugmentationSequential(K.RandomCrop(224, 224), K.Normalize(means, stds))
        self.aug = AugmentationSequential(K.Normalize(means, stds))

    def setup(self, stage: str) -> None:
        self.images = FireScarsHLS(
            "/dccstor/geofm-finetuning/fire-scars/finetune-data/6_bands_no_replant_extended/training/"
        )
        self.labels = FireScarsSegmentationMask(
            "/dccstor/geofm-finetuning/fire-scars/finetune-data/6_bands_no_replant_extended/training/"
        )
        self.dataset = self.images & self.labels
        self.train_aug = AugmentationSequential(K.RandomCrop(224, 224), K.normalize())

        self.images_test = FireScarsHLS(
            "/dccstor/geofm-finetuning/fire-scars/finetune-data/6_bands_no_replant_extended/validation/"
        )
        self.labels_test = FireScarsSegmentationMask(
            "/dccstor/geofm-finetuning/fire-scars/finetune-data/6_bands_no_replant_extended/validation/"
        )
        self.val_dataset = self.images_test & self.labels_test

        if stage in ["fit"]:
            self.train_batch_sampler = RandomBatchGeoSampler(self.dataset, self.patch_size, self.batch_size, None)
        if stage in ["fit", "validate"]:
            self.val_sampler = GridGeoSampler(self.val_dataset, self.patch_size, self.patch_size)
        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(self.val_dataset, self.patch_size, self.patch_size)

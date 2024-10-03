# Copyright contributors to the Terratorch project

from typing import Any

import albumentations as A
import kornia.augmentation as K  # noqa: N812
from torchgeo.datamodules import GeoDataModule, NonGeoDataModule
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler
from torchgeo.transforms import AugmentationSequential

from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import FireScarsHLS, FireScarsNonGeo, FireScarsSegmentationMask

MEANS = [
    0.033349706741586264,
    0.05701185520536176,
    0.05889748132001316,
    0.2323245113436119,
    0.1972854853760658,
    0.11944914225186566,
]

STDS = [
    0.02269135568823774,
    0.026807560223070237,
    0.04004109844362779,
    0.07791732423672691,
    0.08708738838140137,
    0.07241979477437814,
]


class FireScarsNonGeoDataModule(NonGeoDataModule):
    """NonGeo Fire Scars data module implementation"""

    def __init__(
        self,
        data_root: str,
        bands: list[int],
        batch_size: int = 4,
        num_workers: int = 0,
        use_metadata: bool = False,  # noqa: FBT001, FBT002
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(FireScarsNonGeo, batch_size, num_workers, **kwargs)

        self.bands = bands
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.use_metadata = use_metadata
        self.data_root = data_root

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                data_root=self.data_root + "/" + "training",
                use_metadata=self.use_metadata,
                transform = self.train_transform,
                bands=self.bands
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                data_root=self.data_root + "/" + "validation",
                use_metadata=self.use_metadata,
                transform = self.val_transform,
                bands=self.bands
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                data_root=self.data_root + "/" + "validation",
                use_metadata=self.use_metadata,
                transform = self.test_transform,
                bands=self.bands
            )


class FireScarsDataModule(GeoDataModule):
    """Geo Fire Scars data module implementation that merges input data with ground truth segmentation masks."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(FireScarsSegmentationMask, 4, 224, 100, 0, **kwargs)
        self.train_aug = AugmentationSequential(K.RandomCrop(224, 224), K.Normalize(MEANS, STDS))
        self.aug = AugmentationSequential(K.Normalize(MEANS, STDS))

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

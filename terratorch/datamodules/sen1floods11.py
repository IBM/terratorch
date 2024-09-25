# Copyright contributors to the Terratorch project

from pathlib import Path
from typing import Any

import albumentations as A
import kornia.augmentation as K  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential

from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import Sen1Floods11NonGeo

MEANS = [
    0.16450718,
    0.1412956,
    0.13795798,
    0.12353792,
    0.1481099,
    0.23991728,
    0.28587557,
    0.26345379,
    0.30902815,
    0.04911151,
    0.00652506,
    0.2044958,
    0.11912015,
]
STDS = [
    0.06977374,
    0.07406382,
    0.07370365,
    0.08692279,
    0.07778555,
    0.09105416,
    0.10690993,
    0.10096586,
    0.11798815,
    0.03380113,
    0.01463465,
    0.09772074,
    0.07659938,
]


class Sen1Floods11NonGeoDataModule(NonGeoDataModule):
    """NonGeo Fire Scars data module implementation"""

    def __init__(
        self,
        batch_size: int = 4,
        num_workers: int = 0,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        use_metadata: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> None:
        super().__init__(Sen1Floods11NonGeo, batch_size, num_workers, **kwargs)
        bands = kwargs["bands"]
        if bands is not None:
            means = [MEANS[b] for b in bands]
            stds = [STDS[b] for b in bands]
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.aug = AugmentationSequential(K.Normalize(means, stds), data_keys=["image", "mask"])
        self.use_metadata = use_metadata

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train",
                data_root=self.data_root,
                bands=self.bands,
                transform=self.train_transform,
                use_metadata=self.use_metadata,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                transform=self.val_transform,
                partition=self.partition,
                use_metadata=self.use_metadata,
                **self.kwargs,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test",
                data_root=self.data_root,
                transform=self.test_transform,
                partition=self.partition,
                use_metadata=self.use_metadata,
                **self.kwargs,
            )


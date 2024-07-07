from typing import Any, Iterable
import torch

import albumentations as A
import kornia.augmentation as K  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential
from terratorch.datasets import MNzCattleNonGeo
from terratorch.datamodules.utils import wrap_in_compose_is_list


MEANS = {"BLUE": 106.51769083969465, "GREEN": 130.09102671755724, "RED": 126.31354389312978}

STDS = {"BLUE": 18.991624007810348, "GREEN": 19.92435830412983, "RED": 23.38495539231075}


class MNzCattleNonGeoDataModule(NonGeoDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        data_root: str = "./",
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        aug: AugmentationSequential = None,
        **kwargs: Any,
    ) -> None:

        super().__init__(MNzCattleNonGeo, batch_size, num_workers, **kwargs)

        bands = kwargs.get("bands", MNzCattleNonGeo.all_band_names)
        self.means = torch.tensor([MEANS[b] for b in bands])
        self.stds = torch.tensor([STDS[b] for b in bands])
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.data_root = data_root
        self.aug = (
            AugmentationSequential(K.Normalize(self.means, self.stds), data_keys=["image", "mask"])
            if aug is None
            else aug
        )

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train", data_root=self.data_root, transform=self.train_transform, **self.kwargs
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val", data_root=self.data_root, transform=self.val_transform, **self.kwargs
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test", data_root=self.data_root, transform=self.test_transform, **self.kwargs
            )

from typing import Any, Iterable
import torch

import albumentations as A
import kornia.augmentation as K  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential
from terratorch.datasets import MForestNetNonGeo
from terratorch.datamodules.utils import wrap_in_compose_is_list


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


class MForestNetNonGeoDataModule(NonGeoDataModule):
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

        super().__init__(MForestNetNonGeo, batch_size, num_workers, **kwargs)

        bands = kwargs.get("bands", MForestNetNonGeo.all_band_names)
        self.means = torch.tensor([MEANS[b] for b in bands])
        self.stds = torch.tensor([STDS[b] for b in bands])
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.data_root = data_root
        self.aug = (
            AugmentationSequential(K.Normalize(self.means, self.stds), data_keys=["image"]) if aug is None else aug
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

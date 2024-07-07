from typing import Any
import torch

import albumentations as A
import kornia.augmentation as K  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential
from terratorch.datasets import MBrickKilnNonGeo
from terratorch.datamodules.utils import wrap_in_compose_is_list


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


class MBrickKilnNonGeoDataModule(NonGeoDataModule):
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

        super().__init__(MBrickKilnNonGeo, batch_size, num_workers, **kwargs)

        bands = kwargs.get("bands", MBrickKilnNonGeo.all_band_names)
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

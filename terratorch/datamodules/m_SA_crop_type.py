from typing import Any, Iterable
import torch

import albumentations as A
import kornia.augmentation as K  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential
from terratorch.datasets import MSACropTypeNonGeo
from terratorch.datamodules.utils import wrap_in_compose_is_list


MEANS = {
    "COASTAL_AEROSOL": 12.739611,
    "BLUE": 16.526744,
    "GREEN": 26.636417,
    "RED": 36.696639,
    "RED_EDGE_1": 46.388679,
    "RED_EDGE_2": 58.281453,
    "RED_EDGE_3": 63.575819,
    "NIR_BROAD": 68.1836,
    "NIR_NARROW": 69.142591,
    "WATER_VAPOR": 69.904566,
    "SWIR_1": 83.626811,
    "SWIR_2": 65.767679,
    "CLOUD_PROBABILITY": 0.0
}

STDS = {
    "COASTAL_AEROSOL": 7.492811526301659,
    "BLUE": 9.329547939662671,
    "GREEN": 12.674537246073758,
    "RED": 19.421922023931593,
    "RED_EDGE_1": 19.487411106531287,
    "RED_EDGE_2": 19.959174612412983,
    "RED_EDGE_3": 21.53805760692545,
    "NIR_BROAD": 23.05077775347288,
    "NIR_NARROW": 22.329695761624677,
    "WATER_VAPOR": 21.877766438821954,
    "SWIR_1": 28.14418826277069,
    "SWIR_2": 27.2346215312965,
    "CLOUD_PROBABILITY": 0.0
}

class MSACropTypeNonGeoDataModule(NonGeoDataModule):
    def __init__(
        self, 
        batch_size: int = 8, 
        num_workers: int = 0, 
        data_root: str = "./",
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        aug: AugmentationSequential = None,
        **kwargs: Any
    ) -> None:

        super().__init__(MSACropTypeNonGeo, batch_size, num_workers, **kwargs)
        
        bands = kwargs.get("bands", MSACropTypeNonGeo.all_band_names)
        self.means = torch.tensor([MEANS[b] for b in bands])
        self.stds = torch.tensor([STDS[b] for b in bands])
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.data_root = data_root
        self.aug = AugmentationSequential(K.Normalize(self.means, self.stds), data_keys=["image", "mask"]) if aug is None else aug
    
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
                split="test",data_root=self.data_root, transform=self.test_transform, **self.kwargs
            )

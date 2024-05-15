from typing import Any
import torch

import albumentations as A
import kornia.augmentation as K  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential
from terratorch.datasets import MBeninSmallHolderCashewsNonGeo
from terratorch.datamodules.utils import wrap_in_compose_is_list


MEANS = {
    "COASTAL_AEROSOL": 520.1185302734375,
    "BLUE": 634.7583618164062,
    "GREEN": 892.461181640625,
    "RED": 880.7075805664062,
    "RED_EDGE_1": 1380.6409912109375,
    "RED_EDGE_2": 2233.432373046875,
    "RED_EDGE_3": 2549.379638671875,
    "NIR_BROAD": 2643.248046875,
    "NIR_NARROW": 2643.531982421875,
    "WATER_VAPOR": 2852.87451171875,
    "SWIR_1": 2463.933349609375,
    "SWIR_2": 1600.9207763671875,
    "CLOUD_PROBABILITY": 0.010281000286340714
}

STDS = {
    "COASTAL_AEROSOL": 204.2023468017578,
    "BLUE": 227.25344848632812,
    "GREEN": 222.32545471191406,
    "RED": 350.47235107421875,
    "RED_EDGE_1": 280.6436767578125,
    "RED_EDGE_2": 373.7521057128906,
    "RED_EDGE_3": 449.9236145019531,
    "NIR_BROAD": 414.6498107910156,
    "NIR_NARROW": 415.1019592285156,
    "WATER_VAPOR": 413.8980407714844,
    "SWIR_1": 494.97430419921875,
    "SWIR_2": 514.4229736328125,
    "CLOUD_PROBABILITY": 0.3447800576686859
}

class MBeninSmallHolderCashewsNonGeoDataModule(NonGeoDataModule):

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

        super().__init__(MBeninSmallHolderCashewsNonGeo, batch_size, num_workers, **kwargs)
        
        bands = kwargs.get("bands", MBeninSmallHolderCashewsNonGeo.all_band_names)
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
from typing import Any, Iterable
import torch

import albumentations as A
import kornia.augmentation as K  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential
from terratorch.datasets import MEuroSATNonGeo
from terratorch.datamodules.utils import wrap_in_compose_is_list

MEANS = {
    "COASTAL_AEROSOL": 1355.5426, 
    "BLUE": 1113.8855, 
    "GREEN": 1035.7394,  
    "RED": 928.2619, 
    "RED_EDGE_1": 1188.2629, 
    "RED_EDGE_2": 2032.7325,
    "RED_EDGE_3": 2416.5286, 
    "NIR_BROAD": 2342.5396,  
    "NIR_NARROW": 748.9036,   
    "WATER_VAPOR": 12.0419, 
    "CIRRUS": 1810.1284, 
    "SWIR_1": 1101.3801,
    "SWIR_2": 2644.5996
}

STDS = {
    "COASTAL_AEROSOL": 68.9288, 
    "BLUE": 160.0012, 
    "GREEN": 194.6687, 
    "RED": 286.8012, 
    "RED_EDGE_1": 236.6991, 
    "RED_EDGE_2": 372.3853, 
    "RED_EDGE_3": 478.1329,
    "NIR_BROAD": 556.7527, 
    "NIR_NARROW": 102.5583,   
    "WATER_VAPOR": 1.2167, 
    "CIRRUS": 392.9388, 
    "SWIR_1": 313.7339, 
    "SWIR_2": 526.7788
}

class MEuroSATNonGeoDataModule(NonGeoDataModule):
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

        super().__init__(MEuroSATNonGeo, batch_size, num_workers, **kwargs)
        
        bands = kwargs.get("bands", MEuroSATNonGeo.all_band_names)
        self.means = torch.tensor([MEANS[b] for b in bands])
        self.stds = torch.tensor([STDS[b] for b in bands])
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.data_root = data_root
        self.aug = AugmentationSequential(K.Normalize(self.means, self.stds), data_keys=["image"]) if aug is None else aug
    
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

from typing import Any, Iterable
import torch

import albumentations as A
import kornia.augmentation as K  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential
from terratorch.datasets import MSo2SatNonGeo
from terratorch.datamodules.utils import wrap_in_compose_is_list

MEANS = {
    "VH_REAL": 0.00030114364926703274,
    "BLUE": 0.12951652705669403,
    "VH_IMAGINARY": -5.6475887504348066e-06,
    "GREEN": 0.11734361201524734,
    "VV_REAL": 0.000289927760604769,
    "RED": 0.11374464631080627,
    "VV_IMAGINARY": -0.0005758664919994771,
    "VH_LEE_FILTERED": 0.056090500205755234,
    "RED_EDGE_1": 0.12693354487419128,
    "VV_LEE_FILTERED": 0.33695536851882935,
    "RED_EDGE_2": 0.16917912662029266,
    "VH_LEE_FILTERED_REAL": -0.0026130808982998133,
    "RED_EDGE_3": 0.19080990552902222,
    "NIR_BROAD": 0.18381330370903015,
    "VV_LEE_FILTERED_IMAGINARY": -0.00021384705905802548,
    "NIR_NARROW": 0.20517952740192413,
    "SWIR_1": 0.1762811541557312,
    "SWIR_2": 0.1286638230085373,
}

STDS = {
    "VH_REAL": 0.20626230537891388,
    "BLUE": 0.040680479258298874,
    "VH_IMAGINARY": 0.19834314286708832,
    "GREEN": 0.05125178396701813,
    "VV_REAL": 0.5187134146690369,
    "RED": 0.07254913449287415,
    "VV_IMAGINARY": 0.519291877746582,
    "VH_LEE_FILTERED": 1.8058000802993774,
    "RED_EDGE_1": 0.06872648745775223,
    "VV_LEE_FILTERED": 4.893222808837891,
    "RED_EDGE_2": 0.07402216643095016,
    "VH_LEE_FILTERED_REAL": 0.8903943300247192,
    "RED_EDGE_3": 0.08412779122591019,
    "NIR_BROAD": 0.08534552156925201,
    "VV_LEE_FILTERED_IMAGINARY": 1.2786953449249268,
    "NIR_NARROW": 0.09248979389667511,
    "SWIR_1": 0.10270608961582184,
    "SWIR_2": 0.09284552931785583,
}


class MSo2SatNonGeoDataModule(NonGeoDataModule):
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

        super().__init__(MSo2SatNonGeo, batch_size, num_workers, **kwargs)

        bands = kwargs.get("bands", MSo2SatNonGeo.all_band_names)
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

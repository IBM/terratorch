from collections.abc import Sequence
from typing import Any

import albumentations as A
from kornia.augmentation import AugmentationSequential

from terratorch.datamodules.geobench_data_module import GeobenchDataModule
from terratorch.datasets import MSo2SatNonGeo

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


class MSo2SatNonGeoDataModule(GeobenchDataModule):
    """NonGeo LightningDataModule implementation for M-So2Sat dataset."""

    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        data_root: str = "./",
        bands: Sequence[str] | None = None,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        aug: AugmentationSequential = None,
        partition: str = "default",
        **kwargs: Any,
    ) -> None:
        """
        Initializes the MSo2SatNonGeoDataModule for the MSo2SatNonGeo dataset.

        Args:
            batch_size (int, optional): Batch size for DataLoaders. Defaults to 8.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
            data_root (str, optional): Root directory of the dataset. Defaults to "./".
            bands (Sequence[str] | None, optional): List of bands to use. Defaults to None.
            train_transform (A.Compose | None | list[A.BasicTransform], optional): Transformations for training.
            val_transform (A.Compose | None | list[A.BasicTransform], optional): Transformations for validation.
            test_transform (A.Compose | None | list[A.BasicTransform], optional): Transformations for testing.
            aug (AugmentationSequential, optional): Augmentation/normalization pipeline. Defaults to None.
            partition (str, optional): Partition size. Defaults to "default".
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(
            MSo2SatNonGeo,
            MEANS,
            STDS,
            batch_size=batch_size,
            num_workers=num_workers,
            data_root=data_root,
            bands=bands,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            aug=aug,
            partition=partition,
            **kwargs,
        )

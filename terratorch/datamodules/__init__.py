# Copyright contributors to the Terratorch project

from terratorch.datamodules.fire_scars import FireScarsNonGeoDataModule
from terratorch.datamodules.generic_pixel_wise_data_module import (
    GenericNonGeoPixelwiseRegressionDataModule,
    GenericNonGeoSegmentationDataModule,
)
from terratorch.datamodules.generic_scalar_label_data_module import (
    GenericNonGeoClassificationDataModule,
)
from terratorch.datamodules.m_bigearthnet import MBigEarthNonGeoDataModule
from terratorch.datamodules.m_brick_kiln import MBrickKilnNonGeoDataModule

# geobench segmentation datamodules
from terratorch.datamodules.m_cashew_plantation import MBeninSmallHolderCashewsNonGeoDataModule
from terratorch.datamodules.m_chesapeake_landcover import MChesapeakeLandcoverNonGeoDataModule

# geobench classification datamodules
from terratorch.datamodules.m_eurosat import MEuroSATNonGeoDataModule
from terratorch.datamodules.m_forestnet import MForestNetNonGeoDataModule
from terratorch.datamodules.m_neontree import MNeonTreeNonGeoDataModule
from terratorch.datamodules.m_nz_cattle import MNzCattleNonGeoDataModule
from terratorch.datamodules.m_pv4ger import MPv4gerNonGeoDataModule
from terratorch.datamodules.m_pv4ger_seg import MPv4gerSegNonGeoDataModule
from terratorch.datamodules.m_SA_crop_type import MSACropTypeNonGeoDataModule
from terratorch.datamodules.m_so2sat import MSo2SatNonGeoDataModule

# GenericNonGeoRegressionDataModule,
from terratorch.datamodules.sen1floods11 import Sen1Floods11NonGeoDataModule
from terratorch.datamodules.torchgeo_data_module import TorchGeoDataModule, TorchNonGeoDataModule

__all__ = (
    "GenericNonGeoSegmentationDataModule",
    "GenericNonGeoPixelwiseRegressionDataModule",
    "GenericNonGeoSegmentationDataModule",
    "GenericNonGeoClassificationDataModule",
    # "GenericNonGeoRegressionDataModule",
    "Sen1Floods11NonGeoDataModule",
    "FireScarsNonGeoDataModule",
    "TorchGeoDataModule",
    "TorchNonGeoDataModule",
    "MEuroSATNonGeoDataModule",
    "MBigEarthNonGeoDataModule",
    "MBrickKilnNonGeoDataModule",
    "MForestNetNonGeoDataModule",
    "MSo2SatNonGeoDataModule",
    "MPv4gerNonGeoDataModule",
    "MBeninSmallHolderCashewsNonGeoDataModule",
    "MNzCattleNonGeoDataModule",
    "MChesapeakeLandcoverNonGeoDataModule",
    "MPv4gerSegNonGeoDataModule",
    "MSACropTypeNonGeoDataModule",
    "MNeonTreeNonGeoDataModule"
)

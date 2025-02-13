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
from terratorch.datamodules.multi_temporal_crop_classification import MultiTemporalCropClassificationDataModule
from terratorch.datamodules.open_sentinel_map import OpenSentinelMapDataModule
from terratorch.datamodules.pastis import PASTISDataModule
from terratorch.datamodules.era5 import ERA5DataModule

try:
    wxc_present = True
    from terratorch.datamodules.merra2_downscale import Merra2DownscaleNonGeoDataModule 
except ImportError as e:
    import logging
    logging.getLogger("terratorch").debug("wxc_downscaling not installed")
    wxc_present = False

# GenericNonGeoRegressionDataModule,
from terratorch.datamodules.sen1floods11 import Sen1Floods11NonGeoDataModule
from terratorch.datamodules.sen4agrinet import Sen4AgriNetDataModule
from terratorch.datamodules.torchgeo_data_module import TorchGeoDataModule, TorchNonGeoDataModule
from terratorch.datamodules.generic_multimodal_data_module import GenericMultiModalDataModule


# miscellaneous datamodules
from terratorch.datamodules.openearthmap import OpenEarthMapNonGeoDataModule

from terratorch.datamodules.burn_intensity import BurnIntensityNonGeoDataModule
from terratorch.datamodules.carbonflux import CarbonFluxNonGeoDataModule
from terratorch.datamodules.landslide4sense import Landslide4SenseNonGeoDataModule
from terratorch.datamodules.biomassters import BioMasstersNonGeoDataModule
from terratorch.datamodules.forestnet import ForestNetNonGeoDataModule

# miscellaneous datamodules
from terratorch.datamodules.openearthmap import OpenEarthMapNonGeoDataModule

# Generic classification datamodule
from terratorch.datamodules.sen4map import Sen4MapLucasDataModule

__all__ = (
    "GenericNonGeoSegmentationDataModule",
    "GenericNonGeoPixelwiseRegressionDataModule",
    "GenericNonGeoSegmentationDataModule",
    "GenericNonGeoClassificationDataModule",
    # "GenericNonGeoRegressionDataModule",
    "BurnIntensityNonGeoDataModule",
    "CarbonFluxNonGeoDataModule",
    "Landslide4SenseNonGeoDataModule",
    "ForestNetNonGeoDataModule",
    "BioMasstersNonGeoDataModule"
    "Sen1Floods11NonGeoDataModule",
    "Sen4MapLucasDataModule",
    "FireScarsNonGeoDataModule",
    "MultiTemporalCropClassificationDataModule",
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
    "MNeonTreeNonGeoDataModule",
    "OpenEarthMapModule"
    "OpenSentinelMapDataModule",
    "PASTISDataModule",
    "Sen4AgriNetDataModule",
    "GenericMultiModalDataModule",
)

if wxc_present:
    __all__.__add__(("Merra2DownscaleNonGeoDataModule", ))
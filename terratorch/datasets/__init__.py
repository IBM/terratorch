# Copyright contributors to the Terratorch project

from terratorch.datasets.fire_scars import FireScarsHLS, FireScarsNonGeo, FireScarsSegmentationMask
from terratorch.datasets.generic_pixel_wise_dataset import (
    GenericNonGeoPixelwiseRegressionDataset,
    GenericNonGeoSegmentationDataset,
)
from terratorch.datasets.generic_scalar_label_dataset import (
    GenericNonGeoClassificationDataset,
)
from terratorch.datasets.hls import HLSL30, HLSS30
from terratorch.datasets.m_bigearthnet import MBigEarthNonGeo
from terratorch.datasets.m_brick_kiln import MBrickKilnNonGeo

# geobench datasets segmentation
from terratorch.datasets.m_cashew_plantation import MBeninSmallHolderCashewsNonGeo
from terratorch.datasets.m_chesapeake_landcover import MChesapeakeLandcoverNonGeo

# geobench datasets classification
from terratorch.datasets.m_eurosat import MEuroSATNonGeo
from terratorch.datasets.m_forestnet import MForestNetNonGeo
from terratorch.datasets.m_neontree import MNeonTreeNonGeo
from terratorch.datasets.m_nz_cattle import MNzCattleNonGeo
from terratorch.datasets.m_pv4ger import MPv4gerNonGeo
from terratorch.datasets.m_pv4ger_seg import MPv4gerSegNonGeo
from terratorch.datasets.m_SA_crop_type import MSACropTypeNonGeo
from terratorch.datasets.m_so2sat import MSo2SatNonGeo
from terratorch.datasets.open_sentinel_map import OpenSentinelMap
from terratorch.datasets.pastis import PASTIS
from terratorch.datasets.sen4agrinet import Sen4AgriNet

# GenericNonGeoRegressionDataset,
from terratorch.datasets.sen1floods11 import Sen1Floods11NonGeo
from terratorch.datasets.utils import HLSBands

# TorchGeo RasterDatasets
from terratorch.datasets.wsf import WSF2019, WSFEvolution

__all__ = (
    "GenericNonGeoSegmentationDataset",
    "GenericNonGeoPixelwiseRegressionDataset",
    "GenericNonGeoClassificationDataset",
    "GenericNonGeoRegressionDataset",
    "FireScarsNonGeo",
    "FireScarsHLS",
    "FireScarsSegmentationMask",
    "Sen1Floods11NonGeo",
    "HLSBands",
    "MEuroSATNonGeo",
    "MBigEarthNonGeo",
    "MBrickKilnNonGeo",
    "MForestNetNonGeo",
    "MSo2SatNonGeo",
    "MPv4gerNonGeo",
    "MBeninSmallHolderCashewsNonGeo",
    "MNzCattleNonGeo",
    "MChesapeakeLandcoverNonGeo",
    "MPv4gerSegNonGeo",
    "MSACropTypeNonGeo",
    "MNeonTreeNonGeo",
    "OpenSentinelMap",
    "PASTIS",
    "Sen4AgriNet",
    "WSF2019",
    "WSFEvolution",
    "HLSL30",
    "HLSS30",
)

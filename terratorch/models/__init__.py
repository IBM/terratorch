# Copyright contributors to the Terratorch project

from terratorch.models.prithvi_model_factory import PrithviModelFactory
from terratorch.models.satmae_model_factory import SatMAEModelFactory
from terratorch.models.scalemae_model_factory import ScaleMAEModelFactory
from terratorch.models.smp_model_factory import SMPModelFactory
from terratorch.models.timm_model_factory import TimmModelFactory
from terratorch.models.generic_unet_model_factory import GenericUnetModelFactory
try:
    from terratorch.models.wxc_model_factory import WxCModelFactory
except:
    print("granitewcx is not installed.")

__all__ = (
    "PrithviModelFactory",
    "ClayModelFactory",
    "SatMAEModelFactory",
    "ScaleMAEModelFactory",
    "SMPModelFactory",
    "GenericUnetModelFactory",
    "TimmModelFactory",
    "AuxiliaryHead",
    "AuxiliaryHeadWithDecoderWithoutInstantiatedHead",
    "WxCModelFactory",
)

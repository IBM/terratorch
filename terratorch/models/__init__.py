# Copyright contributors to the Terratorch project


import logging
import terratorch.models.necks  # register necks  # noqa: F401
from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory
from terratorch.models.generic_unet_model_factory import GenericUnetModelFactory
from terratorch.models.generic_model_factory import GenericModelFactory
from terratorch.models.prithvi_model_factory import PrithviModelFactory
from terratorch.models.clay_model_factory import ClayModelFactory
from terratorch.models.satmae_model_factory import SatMAEModelFactory
from terratorch.models.smp_model_factory import SMPModelFactory
from terratorch.models.timm_model_factory import TimmModelFactory
from terratorch.models.full_model_factory import FullModelFactory

try:
    granitewcx = True
    from terratorch.models.wxc_model_factory import WxCModelFactory
except ImportError:
    logging.getLogger("terratorch").debug("granitewxc not installed, please use pip install granitewxc")
    granitewcx = False

__all__ = (
    "PrithviModelFactory",
    "ClayModelFactory",
    "SatMAEModelFactory",
    "SMPModelFactory",
    "GenericUnetModelFactory",
    "GenericModelFactory",
    "TimmModelFactory",
    "WxCModelFactory",
    "EncoderDecoderFactory",
    "FullModelFactory",
)

if granitewcx:
    __all__.__add__((WxCModelFactory,))

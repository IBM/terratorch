# Copyright contributors to the Terratorch project


import logging

import terratorch.models.necks  # register necks  # noqa: F401
from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory
from terratorch.models.generic_unet_model_factory import GenericUnetModelFactory
from terratorch.models.prithvi_model_factory import PrithviModelFactory
from terratorch.models.clay_model_factory import ClayModelFactory
from terratorch.models.satmae_model_factory import SatMAEModelFactory
from terratorch.models.smp_model_factory import SMPModelFactory
from terratorch.models.timm_model_factory import TimmModelFactory

try:
    from terratorch.models.wxc_model_factory import WxCModelFactory
except ImportError:
    import logging
    logging.getLogger("terratorch").debug("granitewcx not installed")

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
    "UNet",
    "WxCModelFactory",
    "EncoderDecoderFactory"
)

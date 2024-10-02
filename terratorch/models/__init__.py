# Copyright contributors to the Terratorch project

import logging

import terratorch.models.post_backbone_ops
from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory
from terratorch.models.generic_unet_model_factory import GenericUnetModelFactory
from terratorch.models.prithvi_model_factory import PrithviModelFactory
from terratorch.models.registry import BACKBONE_REGISTRY, DECODER_REGISTRY
from terratorch.models.satmae_model_factory import SatMAEModelFactory
from terratorch.models.scalemae_model_factory import ScaleMAEModelFactory
from terratorch.models.smp_model_factory import SMPModelFactory
from terratorch.models.timm_model_factory import TimmModelFactory

try:
    from terratorch.models.wxc_model_factory import WxCModelFactory
except ImportError:
    logging.debug("granitewcx is not installed.")

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
    "EncoderDecoderFactory"
)

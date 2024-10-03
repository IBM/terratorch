# Copyright contributors to the Terratorch project

from terratorch.models.decoders.fcn_decoder import FCNDecoder
from terratorch.models.decoders.identity_decoder import IdentityDecoder
from terratorch.models.decoders.satmae_head import SatMAEHead, SatMAEHeadViT
from terratorch.models.decoders.smp_decoders import (  # register smp decoder registry
    SMPDecoderWrapper,
)
from terratorch.models.decoders.upernet_decoder import UperNetDecoder

__all__ = ["FCNDecoder", "UperNetDecoder", "IdentityDecoder", "SatMAEHead", "SatMAEHeadViT", "SMPDecoderWrapper"]

import importlib

if importlib.util.find_spec("segmentation_models_pytorch"):
    from terratorch.models.decoders.smp_decoders import SMP_DECODER_REGISTRY
__all__ += ["SMP_DECODER_REGISTRY"]

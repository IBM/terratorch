# reference torchgeo https://torchgeo.readthedocs.io/en/stable/_modules/torchgeo/models/swin.html

import torchgeo.models.swin as swin
from torchgeo.models.swin import Swin_V2_T_Weights, Swin_V2_B_Weights
import logging
from collections.abc import Callable
from functools import partial
import huggingface_hub
import torch.nn as nn
from typing import List
import huggingface_hub
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models import swin_v2_t, swin_v2_b
import torch

torchgeo_swin_model_registry: dict[str, Callable] = {}

def register_swin_model(constructor: Callable):
    torchgeo_swin_model_registry[constructor.__name__] = constructor
    return constructor

class SwinEncoderWrapper(nn.Module):

    """
    A wrapper for Satlas models from torchgeo to return only the forward pass of the encoder 
    Attributes:
        satlas_model (SwinTransformer): The instantiated dofa model
        weights
    Methods:
        forward(x: List[torch.Tensor], wavelengths: list[float]) -> torch.Tensor:
            Forward pass for embeddings with specified indices.
    """

    def __init__(self, swin_model, weights=None) -> None:
        """
        Args:
            dofa_model (DOFA): The decoder module to be wrapped.
            weights ()
        """
        super().__init__()
        self.swin_model = swin_model
        self.weights = weights

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.swin_model(x)


@register_swin_model
def satlas_swin_t_sentinel2_mi_ms(**kwargs):
    model = swin_v2_t(**kwargs)
    return SatlasEncoderWrapper(model, Swin_V2_T_Weights.SENTINEL2_MI_MS_SATLAS)

@register_swin_model
def satlas_swin_t_sentinel2_mi_rgb(**kwargs):
    model = swin_v2_t(**kwargs)
    return SatlasEncoderWrapper(model, Swin_V2_T_Weights.SENTINEL2_MI_RGB_SATLAS)

@register_swin_model
def satlas_swin_t_sentinel2_si_ms(**kwargs):
    model = swin_v2_t(**kwargs)
    return SatlasEncoderWrapper(model, Swin_V2_T_Weights.SENTINEL2_si_MS_SATLAS)

@register_swin_model
def satlas_swin_t_sentinel2_si_rgb(**kwargs):
    model = swin_v2_t(**kwargs)
    return SatlasEncoderWrapper(model, Swin_V2_T_Weights.SENTINEL2_si_RGB_SATLAS)

@register_swin_model
def satlas_swin_b_sentinel2_mi_ms(**kwargs):
    model = swin_v2_b(**kwargs)
    return SatlasEncoderWrapper(model, Swin_V2_B_Weights.SENTINEL2_MI_MS_SATLAS)

@register_swin_model
def satlas_swin_b_sentinel2_mi_rgb(**kwargs):
    model = swin_v2_b(**kwargs)
    return SatlasEncoderWrapper(model, Swin_V2_B_Weights.SENTINEL2_MI_RGB_SATLAS)

@register_swin_model
def satlas_swin_b_sentinel2_si_ms(**kwargs):
    model = swin_v2_b(**kwargs)
    return SatlasEncoderWrapper(model, Swin_V2_B_Weights.SENTINEL2_si_MS_SATLAS)

@register_swin_model
def satlas_swin_b_sentinel2_si_rgb(**kwargs):
    model = swin_v2_b(**kwargs)
    return SatlasEncoderWrapper(model, Swin_V2_B_Weights.SENTINEL2_si_RGB_SATLAS)

@register_swin_model
def satlas_swin_b_naip_mi_rgb(**kwargs):
    model = swin_v2_b(**kwargs)
    return SatlasEncoderWrapper(model, Swin_V2_B_Weights.NAIP_MI_RGB_SATLAS)

@register_swin_model
def satlas_swin_b_naip_si_rgb(**kwargs):
    model = swin_v2_b(**kwargs)
    return SatlasEncoderWrapper(model, Swin_V2_B_Weights.NAIP_SI_RGB_SATLAS)
   
@register_swin_model
def satlas_swin_b_landsat_mi_ms(**kwargs):
    model = swin_v2_b(**kwargs)
    return SatlasEncoderWrapper(model, Swin_V2_B_Weights.LANDSAT_MI_MS_SATLAS)

@register_swin_model
def satlas_swin_b_landsat_mi_rgb(**kwargs):
    model = swin_v2_b(**kwargs)
    return SatlasEncoderWrapper(model, Swin_V2_B_Weights.LANDSAT_MI_RGB_SATLAS)


# reference torchgeo https://torchgeo.readthedocs.io/en/stable/_modules/torchgeo/models/vit.html

import torchgeo.models.vit as vit
from torchgeo.models.vit import ViTSmall16_Weights
import logging
from collections.abc import Callable
from functools import partial
import huggingface_hub
import torch.nn as nn
from typing import List
import huggingface_hub
from torchvision.models._api import Weights, WeightsEnum
import torch

torchgeo_vit_model_registry: dict[str, Callable] = {}

def register_vit_model(constructor: Callable):
    torchgeo_vit_model_registry[constructor.__name__] = constructor
    return constructor

class ViTEncoderWrapper(nn.Module):

    """
    A wrapper for ViT models from torchgeo to return only the forward pass of the encoder 
    Attributes:
        satlas_model (VisionTransformer): The instantiated dofa model
        weights
    Methods:
        forward(x: List[torch.Tensor], wavelengths: list[float]) -> torch.Tensor:
            Forward pass for embeddings with specified indices.
    """

    def __init__(self, vit_model, weights=None) -> None:
        """
        Args:
            dofa_model (DOFA): The decoder module to be wrapped.
            weights ()
        """
        super().__init__()
        self.vit_model = vit_model
        self.weights = weights
        self.out_channels = self.vit_model.feature_info

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.vit_model(x)


@register_vit_model
def ssl4eol_vit_small_patch16_224_landsat_tm_toa_moco(**kwargs):
    model = vit_small_patch16_224(**kwargs)
    return ViTEncoderWrapper(model, ViTSmall16_Weights.LANDSAT_TM_TOA_MOCO)

@register_vit_model
def ssl4eol_vit_small_patch16_224_landsat_tm_toa_simclr(**kwargs):
    model = vit_small_patch16_224(**kwargs)
    return ViTEncoderWrapper(model, ViTSmall16_Weights.LANDSAT_TM_TOA_SIMCLR)

@register_vit_model
def ssl4eol_vit_small_patch16_224_landsat_etm_toa_moco(**kwargs):
    model = vit_small_patch16_224(**kwargs)
    return ViTEncoderWrapper(model, ViTSmall16_Weights.LANDSAT_ETM_TOA_MOCO)

@register_vit_model
def ssl4eol_vit_small_patch16_224_landsat_etm_toa_simclr(**kwargs):
    model = vit_small_patch16_224(**kwargs)
    return ViTEncoderWrapper(model, ViTSmall16_Weights.LANDSAT_ETM_TOA_SIMCLR)

@register_vit_model
def ssl4eol_vit_small_patch16_224_landsat_etm_sr_moco(**kwargs):
    model = vit_small_patch16_224(**kwargs)
    return ViTEncoderWrapper(model, ViTSmall16_Weights.LANDSAT_ETM_SR_MOCO)

@register_vit_model
def ssl4eol_vit_small_patch16_224_landsat_etm_sr_simclr(**kwargs):
    model = vit_small_patch16_224(**kwargs)
    return ViTEncoderWrapper(model, ViTSmall16_Weights.LANDSAT_ETM_SR_SIMCLR)

@register_vit_model
def ssl4eol_vit_small_patch16_224_landsat_oli_tirs_toa_simclr(**kwargs):
    model = vit_small_patch16_224(**kwargs)
    return ViTEncoderWrapper(model, ViTSmall16_Weights.LANDSAT_OLI_TIRS_TOA_SIMCLR)

@register_vit_model
def ssl4eol_vit_small_patch16_224_landsat_oli_sr_moco(**kwargs):
    model = vit_small_patch16_224(**kwargs)
    return ViTEncoderWrapper(model, ViTSmall16_Weights.LANDSAT_OLI_SR_MOCO)

@register_vit_model
def ssl4eol_vit_small_patch16_224_landsat_oli_sr_simclr(**kwargs):
    model = vit_small_patch16_224(**kwargs)
    return ViTEncoderWrapper(model, ViTSmall16_Weights.LANDSAT_OLI_SR_SIMCLR)

@register_vit_model
def ssl4eos12_vit_small_patch16_224_sentinel2_all_dino(**kwargs):
    model = vit_small_patch16_224(**kwargs)
    return ViTEncoderWrapper(model, ViTSmall16_Weights.SENTINEL2_ALL_DINO)

@register_vit_model
def ssl4eos12_vit_small_patch16_224_sentinel2_all_moco(**kwargs):
    model = vit_small_patch16_224(**kwargs)
    return ViTEncoderWrapper(model, ViTSmall16_Weights.SENTINEL2_ALL_MOCO)

#### to add build model and load weights
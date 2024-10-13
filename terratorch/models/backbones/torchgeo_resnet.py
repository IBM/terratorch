# reference torchgeo https://torchgeo.readthedocs.io/en/stable/_modules/torchgeo/models/resnet.html

import torchgeo.models.resnet as resnet
from torchgeo.models.resnet import ResNet, ResNet18_Weights, ResNet50_Weights, ResNet152_Weights, resnet18, resnet50, resnet152
import logging
from collections.abc import Callable
from functools import partial
import huggingface_hub
import torch.nn as nn
from typing import List
import huggingface_hub
from torchvision.models._api import Weights, WeightsEnum
import torch

torchgeo_resnet_model_registry: dict[str, Callable] = {}

def register_resnet_model(constructor: Callable):
    torchgeo_resnet_model_registry[constructor.__name__] = constructor
    return constructor

class ResNetEncoderWrapper(nn.Module):

    """
    A wrapper for ViT models from torchgeo to return only the forward pass of the encoder 
    Attributes:
        satlas_model (VisionTransformer): The instantiated dofa model
        weights
    Methods:
        forward(x: List[torch.Tensor], wavelengths: list[float]) -> torch.Tensor:
            Forward pass for embeddings with specified indices.
    """

    def __init__(self, resnet_model, weights=None) -> None:
        """
        Args:
            dofa_model (DOFA): The decoder module to be wrapped.
            weights ()
        """
        super().__init__()
        self.resnet_model = resnet_model
        self.weights = weights
        self.out_channels = resnet_model.feature_info

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.resnet_model(x)

#### resnet 18
@register_resnet_model
def ssl4eol_resnet18_landsat_tm_toa_moco(**kwargs):
    model = resnet18(**kwargs)
    return ResNetEncoderWrapper(model, ResNet18_Weights.LANDSAT_TM_TOA_MOCO)

@register_resnet_model
def ssl4eol_resnet18_landsat_tm_toa_simclr(**kwargs):
    model = resnet18(**kwargs)
    return ResNetEncoderWrapper(model, ResNet18_Weights.LANDSAT_TM_TOA_SIMCLR)

@register_resnet_model
def ssl4eol_resnet18_landsat_etm_toa_moco(**kwargs):
    model = resnet18(**kwargs)
    return ResNetEncoderWrapper(model, ResNet18_Weights.LANDSAT_ETM_TOA_MOCO)

@register_resnet_model
def ssl4eol_resnet18_landsat_etm_toa_simclr(**kwargs):
    model = resnet18(**kwargs)
    return ResNetEncoderWrapper(model, ResNet18_Weights.LANDSAT_ETM_TOA_SIMCLR)

@register_resnet_model
def ssl4eol_resnet18_landsat_etm_sr_moco(**kwargs):
    model = resnet18(**kwargs)
    return ResNetEncoderWrapper(model, ResNet18_Weights.LANDSAT_ETM_SR_MOCO)

@register_resnet_model
def ssl4eol_resnet18_landsat_etm_sr_simclr(**kwargs):
    model = resnet18(**kwargs)
    return ResNetEncoderWrapper(model, ResNet18_Weights.LANDSAT_ETM_SR_SIMCLR)

@register_resnet_model
def ssl4eol_resnet18_landsat_oli_tirs_toa_moco(**kwargs):
    model = resnet18(**kwargs)
    return ResNetEncoderWrapper(model, ResNet18_Weights.LANDSAT_OLI_TIRS_TOA_MOCO)

@register_resnet_model
def ssl4eol_resnet18_landsat_oli_tirs_toa_simclr(**kwargs):
    model = resnet18(**kwargs)
    return ResNetEncoderWrapper(model, ResNet18_Weights.LANDSAT_OLI_TIRS_TOA_SIMCLR)

@register_resnet_model
def ssl4eol_resnet18_landsat_oli_sr_moco(**kwargs):
    model = resnet18(**kwargs)
    return ResNetEncoderWrapper(model, ResNet18_Weights.LANDSAT_OLI_SR_MOCO)

@register_resnet_model
def ssl4eol_resnet18_landsat_oli_sr_simclr(**kwargs):
    model = resnet18(**kwargs)
    return ResNetEncoderWrapper(model, ResNet18_Weights.LANDSAT_OLI_SR_SIMCLR)

@register_resnet_model
def ssl4eos12_resnet18_sentinel2_all_moco(**kwargs):
    model = resnet18(**kwargs)
    return ResNetEncoderWrapper(model, ResNet18_Weights.SENTINEL2_ALL_MOCO)

@register_resnet_model
def ssl4eos12_resnet18_sentinel2_rgb_moco(**kwargs):
    model = resnet18(**kwargs)
    return ResNetEncoderWrapper(model, ResNet18_Weights.SENTINEL2_RGB_MOCO)

@register_resnet_model
def seco_resnet18_sentinel2_rgb_seco(**kwargs):
    model = resnet18(**kwargs)
    return ResNetEncoderWrapper(model, ResNet18_Weights.SENTINEL2_RGB_SECO)

#### resnet 50
@register_resnet_model
def fmow_resnet50_fmow_rgb_gassl(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.FMOW_RGB_GASSL)

@register_resnet_model
def ssl4eol_resnet50_landsat_tm_toa_moco(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.LANDSAT_TM_TOA_MOCO)

@register_resnet_model
def ssl4eol_resnet50_landsat_tm_toa_simclr(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.LANDSAT_TM_TOA_SIMCLR)

@register_resnet_model
def ssl4eol_resnet50_landsat_etm_toa_moco(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.LANDSAT_ETM_TOA_MOCO)

@register_resnet_model
def ssl4eol_resnet50_landsat_etm_toa_simclr(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.LANDSAT_ETM_TOA_SIMCLR)

@register_resnet_model
def ssl4eol_resnet50_landsat_etm_sr_moco(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.LANDSAT_ETM_SR_MOCO)

@register_resnet_model
def ssl4eol_resnet50_landsat_etm_sr_simclr(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.LANDSAT_ETM_SR_SIMCLR)

@register_resnet_model
def ssl4eol_resnet50_landsat_oli_tirs_toa_moco(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.LANDSAT_OLI_TIRS_TOA_MOCO)

@register_resnet_model
def ssl4eol_resnet50_landsat_oli_tirs_toa_simclr(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.LANDSAT_OLI_TIRS_TOA_SIMCLR)

@register_resnet_model
def ssl4eol_resnet50_landsat_oli_sr_moco(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.LANDSAT_OLI_SR_MOCO)

@register_resnet_model
def ssl4eol_resnet50_landsat_oli_sr_simclr(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.LANDSAT_OLI_SR_SIMCLR)

@register_resnet_model
def ssl4eos12_resnet50_sentinel1_all_decur(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.SENTINEL1_ALL_DECUR)

@register_resnet_model
def ssl4eos12_resnet50_sentinel1_all_moco(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.SENTINEL1_ALL_MOCO)

@register_resnet_model
def ssl4eos12_resnet50_sentinel2_all_decur(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.SENTINEL2_ALL_DECUR)

@register_resnet_model
def ssl4eos12_resnet50_sentinel2_all_dino(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.SENTINEL2_ALL_DINO)

@register_resnet_model
def ssl4eos12_resnet50_sentinel2_all_moco(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.SENTINEL2_ALL_MOCO)

@register_resnet_model
def ssl4eos12_resnet50_sentinel2_rgb_moco(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.SENTINEL2_RGB_MOCO)

@register_resnet_model
def seco_resnet50_sentinel2_rgb_seco(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.SENTINEL2_RGB_SECO)

@register_resnet_model
def satlas_resnet50_sentinel2_mi_ms_satlas(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.SENTINEL2_MI_MS_SATLAS)

@register_resnet_model
def satlas_resnet50_sentinel2_mi_rgb_satlas(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.SENTINEL2_MI_RGB_SATLAS)

@register_resnet_model
def satlas_resnet50_sentinel2_si_ms_satlas(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.SENTINEL2_SI_MS_SATLAS)

@register_resnet_model
def satlas_resnet50_sentinel2_si_rgb_satlas(**kwargs):
    model = resnet50(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.SENTINEL2_SI_RGB_SATLAS)

#### resnet152
@register_resnet_model
def satlas_resnet152_sentinel2_mi_ms(**kwargs):
    model = resnet152(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.SENTINEL2_MI_MS_SATLAS)

@register_resnet_model
def satlas_resnet152_sentinel2_mi_rgb(**kwargs):
    model = resnet152(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.SENTINEL2_MI_RGB_SATLAS)

@register_resnet_model
def satlas_resnet152_sentinel2_si_ms_satlas(**kwargs):
    model = resnet152(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.SENTINEL2_SI_MS_SATLAS)

@register_resnet_model
def satlas_resnet152_sentinel2_si_rgb_satlas(**kwargs):
    model = resnet152(**kwargs)
    return ResNetEncoderWrapper(model, ResNet50_Weights.SENTINEL2_SI_RGB_SATLAS)


#### to add build model and load weights
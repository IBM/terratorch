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
from terratorch.datasets.utils import OpticalBands, SARBands
from terratorch.models.backbones.select_patch_embed_weights import select_patch_embed_weights

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
import torch
import pdb

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

    def __init__(self, resnet_model, resnet_meta, weights=None, out_indices=None) -> None:
        """
        Args:
            dofa_model (DOFA): The decoder module to be wrapped.
            weights ()
        """
        super().__init__()
        self.resnet_model = resnet_model
        self.resnet_meta = resnet_meta
        self.weights = weights
        self.out_indices = out_indices if out_indices else [-1]
        self.out_channels = [x['num_chs'] for x in self.resnet_model.feature_info]
        self.resnet_meta['original_out_channels'] = self.out_channels
        self.out_channels = [x for i, x in enumerate(self.out_channels) if (i in self.out_indices) | (i == (len(self.out_channels)-1)) & (-1 in self.out_indices)]
        
    
    def forward(self, x: List[torch.Tensor], **kwargs) -> torch.Tensor:
        
        features = self.resnet_model.forward_intermediates(x, intermediates_only=True)

        outs = []
        for i, feature in enumerate(features):
            if i in self.out_indices:
                outs.append(feature)
            elif (i == (len(self.resnet_meta["original_out_channels"])-1)) & (-1 in self.out_indices):
                outs.append(feature)

        return outs
        

look_up_table = {
    "B01": "COASTAL_AEROSOL",
    "B02": "BLUE",
    "B03": "GREEN",
    "B04": "RED",
    "B05": "RED_EDGE_1",
    "B06": "RED_EDGE_2",
    "B07": "RED_EDGE_3",
    "B08": "NIR_BROAD",
    "B8A": "NIR_NARROW",
    "B09": "WATER_VAPOR",
    "B10": "CIRRUS",
    "B11": "SWIR_1",
    "B12": "SWIR_2",
    "VV": "VV",
    "VH": "VH",
    "R": "RED",
    "G": "GREEN",
    "B": "BLUE"
}

resnet18_meta = {
    "layers": (2, 2, 2, 2)
    }

resnet50_meta = {
    "layers": (3, 4, 6, 3)
}

resnet152_meta = {
     "layers": (3, 8, 36, 3)
}


def get_pretrained_bands(model_bands):

    model_bands = [look_up_table[x.split('.')[-1]] for x in model_bands]

    return model_bands    


#### resnet 18
@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet18_landsat_tm_toa_moco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet18_Weights.LANDSAT_TM_TOA_MOCO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet18(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet18_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet18_landsat_tm_toa_simclr(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet18_Weights.LANDSAT_TM_TOA_SIMCLR, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet18(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet18_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet18_landsat_etm_toa_moco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet18_Weights.LANDSAT_ETM_TOA_MOCO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet18(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet18_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet18_landsat_etm_toa_simclr(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet18_Weights.LANDSAT_ETM_TOA_SIMCLR, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet18(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet18_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet18_landsat_etm_sr_moco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet18_Weights.LANDSAT_ETM_SR_MOCO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet18(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet18_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet18_landsat_etm_sr_simclr(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet18_Weights.LANDSAT_ETM_SR_SIMCLR, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet18(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet18_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet18_landsat_oli_tirs_toa_moco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet18_Weights.LANDSAT_OLI_TIRS_TOA_MOCO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet18(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet18_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet18_landsat_oli_tirs_toa_simclr(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet18_Weights.LANDSAT_OLI_TIRS_TOA_SIMCLR, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet18(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet18_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet18_landsat_oli_sr_moco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet18_Weights.LANDSAT_OLI_SR_MOCO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet18(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet18_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet18_landsat_oli_sr_simclr(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet18_Weights.LANDSAT_OLI_SR_SIMCLR, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet18(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet18_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eos12_resnet18_sentinel2_all_moco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None =  ResNet18_Weights.SENTINEL2_ALL_MOCO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet18(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet18_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eos12_resnet18_sentinel2_rgb_moco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet18_Weights.SENTINEL2_RGB_MOCO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet18(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet18_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def seco_resnet18_sentinel2_rgb_seco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet18_Weights.SENTINEL2_RGB_SECO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet18(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet18_meta, weights, out_indices)


#### resnet 50
@TERRATORCH_BACKBONE_REGISTRY.register
def fmow_resnet50_fmow_rgb_gassl(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.FMOW_RGB_GASSL, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet50_landsat_tm_toa_moco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.LANDSAT_TM_TOA_MOCO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet50_landsat_tm_toa_simclr(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.LANDSAT_TM_TOA_SIMCLR, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet50_landsat_etm_toa_moco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.LANDSAT_ETM_TOA_MOCO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet50_landsat_etm_toa_simclr(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.LANDSAT_ETM_TOA_SIMCLR, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet50_landsat_etm_sr_moco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.LANDSAT_ETM_SR_MOCO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet50_landsat_etm_sr_simclr(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.LANDSAT_ETM_SR_SIMCLR, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet50_landsat_oli_tirs_toa_moco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.LANDSAT_OLI_TIRS_TOA_MOCO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet50_landsat_oli_tirs_toa_simclr(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.LANDSAT_OLI_TIRS_TOA_SIMCLR, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet50_landsat_oli_sr_moco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.LANDSAT_OLI_SR_MOCO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_resnet50_landsat_oli_sr_simclr(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.LANDSAT_OLI_SR_SIMCLR, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eos12_resnet50_sentinel1_all_decur(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.SENTINEL1_ALL_DECUR, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        if weights is not None:
            weights.meta['bands'] = ['VV', 'VH']
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eos12_resnet50_sentinel1_all_moco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None =  ResNet50_Weights.SENTINEL1_ALL_MOCO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        if weights is not None:
            weights.meta['bands'] = ['VV', 'VH']
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eos12_resnet50_sentinel2_all_decur(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.SENTINEL2_ALL_DECUR, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        if weights is not None:
            weights.meta['bands'] = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eos12_resnet50_sentinel2_all_dino(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.SENTINEL2_ALL_DINO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        if weights is not None:
            weights.meta['bands'] = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eos12_resnet50_sentinel2_all_moco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.SENTINEL2_ALL_MOCO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        if weights is not None:
            weights.meta['bands'] = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eos12_resnet50_sentinel2_rgb_moco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.SENTINEL2_RGB_MOCO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def seco_resnet50_sentinel2_rgb_seco(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.SENTINEL2_RGB_SECO, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_resnet50_sentinel2_mi_ms_satlas(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.SENTINEL2_MI_MS_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_resnet50_sentinel2_mi_rgb_satlas(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.SENTINEL2_MI_RGB_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_resnet50_sentinel2_si_ms_satlas(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.SENTINEL2_SI_MS_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_resnet50_sentinel2_si_rgb_satlas(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet50_Weights.SENTINEL2_SI_RGB_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet50(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet50_meta, weights, out_indices)

#### resnet152
@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_resnet152_sentinel2_mi_ms(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None =  ResNet152_Weights.SENTINEL2_MI_MS_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet152(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet152_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_resnet152_sentinel2_mi_rgb(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet152_Weights.SENTINEL2_MI_RGB_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet152(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet152_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_resnet152_sentinel2_si_ms_satlas(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = ResNet152_Weights.SENTINEL2_SI_MS_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet152(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet152_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_resnet152_sentinel2_si_rgb_satlas(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None =  ResNet152_Weights.SENTINEL2_SI_RGB_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = resnet152(**kwargs)
    if pretrained:
        model = load_resnet_weights(model, model_bands, ckpt_data, weights)
    return ResNetEncoderWrapper(model, resnet152_meta, weights, out_indices)


#### to add build model and load weights
def load_resnet_weights(model: nn.Module, model_bands, ckpt_data: str, weights: Weights, input_size: int = 224, custom_weight_proj: str = "conv1.weight") -> nn.Module:
    
    pretrained_bands = get_pretrained_bands(weights.meta["bands"]) if "bands" in weights.meta else []
    if ckpt_data is not None:
        if ckpt_data.find("https://hf.co/") > -1:
            repo_id = ckpt_data.split("/resolve/")[0].replace("https://hf.co/", '')
            filename = ckpt_data.split("/")[-1]
            ckpt_data = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)

        checkpoint_model = torch.load(ckpt_data, map_location="cpu", weights_only=True)
        state_dict = model.state_dict()
        
        for k in ["fc.weight", "fc.bias"]:
                if (
                    k in checkpoint_model
                    and checkpoint_model[k].shape != state_dict[k].shape
                ):
                    logging.info(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
    
        checkpoint_model = select_patch_embed_weights(checkpoint_model, model, pretrained_bands, model_bands, custom_weight_proj)
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
    
        logging.info(msg)
    else:
        if weights is not None:
            checkpoint_model = weights.get_state_dict(progress=True)
            checkpoint_model = select_patch_embed_weights(checkpoint_model, model, pretrained_bands, model_bands, custom_weight_proj)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
            assert set(missing_keys) <= {'fc.weight', 'fc.bias'}
            assert not unexpected_keys
    
    return model

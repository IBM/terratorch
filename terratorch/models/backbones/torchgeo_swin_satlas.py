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
from terratorch.datasets.utils import OpticalBands, SARBands
from terratorch.models.backbones.select_patch_embed_weights import select_patch_embed_weights

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

class SwinEncoderWrapper(nn.Module):

    """
    A wrapper for Satlas models from torchgeo to return only the forward pass of the encoder 
    Attributes:
        swin_model (SwinTransformer): The instantiated dofa model
        weights
    Methods:
        forward(x: List[torch.Tensor], wavelengths: list[float]) -> torch.Tensor:
            Forward pass for embeddings with specified indices.
    """

    def __init__(self, swin_model, swin_meta, weights=None, out_indices=None) -> None:
        """
        Args:
            swin_model (SwinTransformer): The backbone module to be wrapped.
            swin_meta (dict): dict containing the metadata for swin.
            weights (Weights): Weights class for the swin model to be wrapped.
            out_indices (list): List containing the feature indices to be returned.
        """
        super().__init__()
        self.swin_model = swin_model
        self.weights = weights
        self.out_indices = out_indices if out_indices else [-1]

        self.out_channels = []
        for i in range(len(swin_meta["depths"])):
            self.out_channels.append(swin_meta["embed_dim"] * 2**i)
        self.out_channels = [elem for elem in self.out_channels for _ in range(2)]
        self.out_channels = [x for i, x in enumerate(self.out_channels) if (i in self.out_indices) | (i == (len(self.out_channels)-1)) & (-1 in self.out_indices)]

    def forward(self, x: List[torch.Tensor], **kwargs) -> torch.Tensor:

        outs = []
        for i, layer in enumerate(self.swin_model.features):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
            elif (i == (len(self.swin_model.features)-1)) & (-1 in self.out_indices):
                outs.append(x)
        
        return tuple(outs)

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

swin_v2_t_meta = {
    "patch_size":[4, 4],
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": [8, 8],
    "stochastic_depth_prob": 0.2
}

swin_v2_b_meta = {
    "patch_size":[4, 4],
    "embed_dim": 128,
    "depths": [2, 2, 18, 2],
    "num_heads": [4, 8, 16, 32],
    "window_size": [8, 8],
    "stochastic_depth_prob": 0.5
}

def get_pretrained_bands(model_bands):

    model_bands = [look_up_table[x.split('.')[-1]] for x in model_bands]

    return model_bands

def load_model(load_function, swin_meta, **kwargs):

    in_chans = kwargs['in_chans']
    del kwargs['in_chans']
    model = load_function(**kwargs)
    model.features[0][0] = torch.nn.Conv2d(in_chans, swin_meta["embed_dim"], kernel_size=swin_meta["patch_size"], stride=swin_meta["patch_size"])
    return model
    

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_swin_t_sentinel2_mi_ms(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = Swin_V2_T_Weights.SENTINEL2_MI_MS_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = load_model(swin_v2_t, swin_v2_t_meta, **kwargs)
    if pretrained:
        model = load_swin_weights(model, model_bands, ckpt_data, weights)
    return SwinEncoderWrapper(model, swin_v2_t_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_swin_t_sentinel2_mi_rgb(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = Swin_V2_T_Weights.SENTINEL2_MI_RGB_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = load_model(swin_v2_t, swin_v2_t_meta, **kwargs)
    if pretrained:
        model = load_swin_weights(model, model_bands, ckpt_data, weights)
    return SwinEncoderWrapper(model, swin_v2_t_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_swin_t_sentinel2_si_ms(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = Swin_V2_T_Weights.SENTINEL2_SI_MS_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = load_model(swin_v2_t, swin_v2_t_meta, **kwargs)
    if pretrained:
        model = load_swin_weights(model, model_bands, ckpt_data, weights)
    return SwinEncoderWrapper(model, swin_v2_t_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_swin_t_sentinel2_si_rgb(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = Swin_V2_T_Weights.SENTINEL2_SI_RGB_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = load_model(swin_v2_t, swin_v2_t_meta, **kwargs)
    if pretrained:
        model = load_swin_weights(model, model_bands, ckpt_data, weights)
    return SwinEncoderWrapper(model, swin_v2_t_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_swin_b_sentinel2_mi_ms(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = Swin_V2_B_Weights.SENTINEL2_MI_MS_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = load_model(swin_v2_b, swin_v2_b_meta, **kwargs)
    if pretrained:
        model = load_swin_weights(model, model_bands, ckpt_data, weights)
    return SwinEncoderWrapper(model, swin_v2_b_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_swin_b_sentinel2_mi_rgb(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = Swin_V2_B_Weights.SENTINEL2_MI_RGB_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = load_model(swin_v2_b, swin_v2_b_meta, **kwargs)
    if pretrained:
        model = load_swin_weights(model, model_bands, ckpt_data, weights)
    return SwinEncoderWrapper(model, swin_v2_b_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_swin_b_sentinel2_si_ms(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = Swin_V2_B_Weights.SENTINEL2_SI_MS_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = load_model(swin_v2_b, swin_v2_b_meta, **kwargs)
    if pretrained:
        model = load_swin_weights(model, model_bands, ckpt_data, weights)
    return SwinEncoderWrapper(model, swin_v2_b_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_swin_b_sentinel2_si_rgb(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = Swin_V2_B_Weights.SENTINEL2_SI_RGB_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = load_model(swin_v2_b, swin_v2_b_meta, **kwargs)
    if pretrained:
        model = load_swin_weights(model, model_bands, ckpt_data, weights)
    return SwinEncoderWrapper(model, swin_v2_b_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_swin_b_naip_mi_rgb(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = Swin_V2_B_Weights.NAIP_RGB_MI_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = load_model(swin_v2_b, swin_v2_b_meta, **kwargs)
    if pretrained:
        model = load_swin_weights(model, model_bands, ckpt_data, weights)
    return SwinEncoderWrapper(model, swin_v2_b_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_swin_b_naip_si_rgb(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = Swin_V2_B_Weights.NAIP_RGB_SI_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = load_model(swin_v2_b, swin_v2_b_meta, **kwargs)
    if pretrained:
        model = load_swin_weights(model, model_bands, ckpt_data, weights)
    return SwinEncoderWrapper(model, swin_v2_b_meta, weights, out_indices)
   
@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_swin_b_landsat_mi_ms(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = Swin_V2_B_Weights.LANDSAT_MI_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = load_model(swin_v2_b, swin_v2_b_meta, **kwargs)
    if pretrained:
        model = load_swin_weights(model, model_bands, ckpt_data, weights)
    return SwinEncoderWrapper(model, swin_v2_b_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_swin_b_landsat_mi_rgb(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = Swin_V2_B_Weights.LANDSAT_SI_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = load_model(swin_v2_b, swin_v2_b_meta, **kwargs)
    if pretrained:
        model = load_swin_weights(model, model_bands, ckpt_data, weights)
    return SwinEncoderWrapper(model, swin_v2_b_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_swin_b_sentinel1_mi(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = Swin_V2_B_Weights.SENTINEL1_MI_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = load_model(swin_v2_b, swin_v2_b_meta, **kwargs)
    if pretrained:
        model = load_swin_weights(model, model_bands, ckpt_data, weights)
    return SwinEncoderWrapper(model, swin_v2_b_meta, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def satlas_swin_b_sentinel1_si(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = Swin_V2_B_Weights.SENTINEL1_SI_SATLAS, out_indices: list | None = None, **kwargs):
    if "in_chans" not in kwargs: kwargs["in_chans"] = len(model_bands)
    model = load_model(swin_v2_b, swin_v2_b_meta, **kwargs)
    if pretrained:
        model = load_swin_weights(model, model_bands, ckpt_data, weights)
    return SwinEncoderWrapper(model, swin_v2_b_meta, weights, out_indices)


def load_swin_weights(model: nn.Module, model_bands, ckpt_data: str, weights: Weights, input_size: int = 224, custom_weight_proj: str = "features.0.0.weight") -> nn.Module:
    
    pretrained_bands = get_pretrained_bands(weights.meta["bands"])
    
    if ckpt_data is not None:
        if ckpt_data.find("https://hf.co/") > -1:
            repo_id = ckpt_data.split("/resolve/")[0].replace("https://hf.co/", '')
            filename = ckpt_data.split("/")[-1]
            ckpt_data = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)

        checkpoint_model = torch.load(ckpt_data, map_location="cpu", weights_only=True)
        state_dict = model.state_dict()
        
        for k in ["head.weight", "head.bias"]:
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
            assert set(missing_keys) <= set()
            assert not unexpected_keys

    
    return model

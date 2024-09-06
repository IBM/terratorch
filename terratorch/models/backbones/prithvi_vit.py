# Copyright contributors to the Terratorch project


import logging
from functools import partial
from pathlib import Path
from collections import defaultdict

import torch
from timm.models import FeatureInfo
from timm.models._builder import build_model_with_cfg
from timm.models._registry import generate_default_cfgs, register_model
from torch import nn

from terratorch.datasets import HLSBands
from terratorch.models.backbones.prithvi_select_patch_embed_weights import prithvi_select_patch_embed_weights
from terratorch.models.backbones.vit_encoder_decoder import TemporalViTEncoder
from terratorch.datasets.utils import generate_bands_intervals

PRETRAINED_BANDS = [
    HLSBands.BLUE,
    HLSBands.GREEN,
    HLSBands.RED,
    HLSBands.NIR_NARROW,
    HLSBands.SWIR_1,
    HLSBands.SWIR_2,
]

def _cfg(file: Path = "", **kwargs) -> dict:
    return {
        "file": file,
        "source": "file",
        "input_size": (6, 224, 224),
        "license": "mit",
        # "first_conv": "patch_embed.proj",
        **kwargs,
    }

default_cfgs = generate_default_cfgs(
    {
        "prithvi_vit_100": {
            "hf_hub_id": "ibm-nasa-geospatial/Prithvi-100M",
            "hf_hub_filename": "Prithvi_100M.pt"
        }
    }
)

def checkpoint_filter_fn(
    state_dict, model: TemporalViTEncoder, pretrained_bands: list[HLSBands | int], model_bands: list[HLSBands | int]
) -> dict:
    if "pos_embed" in state_dict:
        del state_dict["pos_embed"]
    if "decoder_pos_embed" in state_dict:
        del state_dict["decoder_pos_embed"]

    if model.encoder_only:
        encoder_only_dict = {}
        for k, v in state_dict.items():
            if "decoder" in k:
                continue
            if "mask_token" in k:
                continue
            encoder_only_dict[k] = v
        state_dict = encoder_only_dict

    state_dict = prithvi_select_patch_embed_weights(state_dict, model, pretrained_bands, model_bands)

    return state_dict

def _create_prithvi(
    variant: str,
    pretrained: bool = False,  # noqa: FBT001, FBT002
    pretrained_bands: list[HLSBands] | None = None,
    model_bands: list[HLSBands | int] | None = None,
    **kwargs,
) -> TemporalViTEncoder:
    if pretrained_bands is None:
        pretrained_bands = PRETRAINED_BANDS

    # Little hack because VIT does not support timm's features_only
    # so we do it ourselves
    encoder_only = kwargs.get("features_only", False)
    if "features_only" in kwargs:
        kwargs = {k: v for k, v in kwargs.items() if k != "features_only"}

    model_bands = generate_bands_intervals(model_bands)

    kwargs["in_chans"] = len(model_bands)

    def checkpoint_filter_wrapper_fn(state_dict, model):
        return checkpoint_filter_fn(state_dict, model, pretrained_bands, model_bands)

    model = build_model_with_cfg(
        TemporalViTEncoder,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_wrapper_fn,
        pretrained_strict=True,
        encoder_only=encoder_only,
        **kwargs,
    )

    if encoder_only:
        default_out_indices = list(range(len(model.blocks)))
        out_indices = kwargs.get("out_indices", default_out_indices)
        if "out_indices" in kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k != "out_indices"}
        model.feature_info = FeatureInfo(model.feature_info, out_indices)
        model.encode_decode_forward = model.forward
        def forward_filter_indices(*args, **kwargs):
            features = model.forward_features(*args, **kwargs)
            return [features[i] for i in out_indices]
        model.forward = forward_filter_indices
        model.model_bands = model_bands
        model.pretrained_bands = pretrained_bands

    return model

def create_prithvi_vit_100(
    model_name: str,
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> TemporalViTEncoder:
    """Prithvi ViT 100M"""
    pretrained_bands = PRETRAINED_BANDS
    if bands is None:
        bands = pretrained_bands
        logging.info(
            f"Model bands not passed. Assuming bands are ordered in the same way as {PRETRAINED_BANDS}.\
            Pretrained patch_embed layer may be misaligned with current bands"
        )

    model_args = {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "decoder_embed_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "mlp_ratio": 4,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "num_frames": 1,
    }

    model = _create_prithvi(
        model_name,
        pretrained=pretrained,
        model_bands=bands,
        pretrained_bands=pretrained_bands,
        **dict(model_args,**kwargs),
    )
    
    return model


def create_prithvi_vit_300(
    model_name: str,
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> TemporalViTEncoder:
    """Prithvi ViT 300M"""
    pretrained_bands = PRETRAINED_BANDS
    if bands is None:
        bands = pretrained_bands
        logging.info(
            f"Model bands not passed. Assuming bands are ordered in the same way as {PRETRAINED_BANDS}.\
            Pretrained patch_embed layer may be misaligned with current bands"
        )
    model_args = {
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "decoder_embed_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "mlp_ratio": 4,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "num_frames": 1,
    }
    model = _create_prithvi(
        model_name,
        pretrained=pretrained,
        pretrained_bands=pretrained_bands,
        model_bands=bands,
        **dict(model_args, **kwargs),
    )
    return model


@register_model
def prithvi_vit_tiny(
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> TemporalViTEncoder:
    """Prithvi ViT tiny"""
    pretrained_bands = PRETRAINED_BANDS
    if bands is None:
        bands = pretrained_bands
    model_args = {
        "patch_size": 16,
        "embed_dim": 256,
        "depth": 4,
        "num_heads": 4,
        "decoder_embed_dim": 128,
        "decoder_depth": 4,
        "decoder_num_heads": 4,
        "mlp_ratio": 4,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "num_frames": 1,
        "model_bands": bands,
    }
    model = _create_prithvi("prithvi_vit_tiny", **dict(model_args, **kwargs))
    return model

@register_model
def prithvi_vit_100(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> TemporalViTEncoder:
    return create_prithvi_vit_100("prithvi_vit_100", pretrained, bands, **kwargs)


@register_model
def prithvi_vit_300(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> TemporalViTEncoder:
    return create_prithvi_vit_300("prithvi_vit_300", pretrained, bands, **kwargs)

"""This module handles registering prithvi_swin models into timm.
"""
from typing import List
import logging
import math
import warnings
from collections import OrderedDict
from pathlib import Path

import torch
from timm.models import SwinTransformer
from timm.models._builder import build_model_with_cfg
from timm.models._registry import generate_default_cfgs, register_model
from timm.models.swin_transformer import checkpoint_filter_fn as timm_swin_checkpoint_filter_fn

from terratorch.datasets.utils import HLSBands
from terratorch.models.backbones.prithvi_select_patch_embed_weights import prithvi_select_patch_embed_weights
from terratorch.models.backbones.hiera_encoder_decoder import HieraMaxViTEncoderDecoder

def _create_hiera_transformer(
    variant: str,
    pretrained_bands: List[int],
    model_bands: List[int],
    pretrained: bool = False,  # noqa: FBT002, FBT001
    **kwargs,
):
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get("depths", (1, 1, 3, 1))))
    out_indices = kwargs.pop("out_indices", default_out_indices)

    # the current swin model is not multitemporal
    if "num_frames" in kwargs:
        kwargs = {k: v for k, v in kwargs.items() if k != "num_frames"}
    kwargs["in_chans"] = len(model_bands)

    def checkpoint_filter_wrapper_fn(state_dict, model):
        return checkpoint_filter_fn(state_dict, model, pretrained_bands, model_bands)

    model: HieraMaxViTEncoderDecoder = build_model_with_cfg(
        HieraMaxViTEncoderDecoder,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_wrapper_fn,
        pretrained_strict=False,
        #feature_cfg={"flatten_sequential": True, "out_indices": out_indices},
        **kwargs,
    )
    model.pretrained_bands = pretrained_bands
    model.model_bands = model_bands

    def prepare_features_for_image_model(x):
        return [
            # layer_output.reshape(
            #     -1,
            #     int(math.sqrt(layer_output.shape[1])),
            #     int(math.sqrt(layer_output.shape[1])),
            #     layer_output.shape[2],
            # )
            layer_output.permute(0, 3, 1, 2).contiguous()
            for layer_output in x
        ]

    # add permuting here
    model.prepare_features_for_image_model = prepare_features_for_image_model
    return model

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
        # Hiera model trained for Weather forecasting
        "hiera_weather": _cfg(),
    }
)


@register_model
def hiera_weather(
    pretrained: bool = False,  # noqa: FBT002, FBT001
    pretrained_bands: List[int] | None = None,
    bands: List[int] | None = None,
    **kwargs,
) -> HieraMaxViTEncoderDecoder:

    """Hiera Weather"""
    if pretrained_bands is None:
        pretrained_bands = PRETRAINED_BANDS
    if bands is None:
        bands = pretrained_bands
        logging.info(
            f"Model bands not passed. Assuming bands are ordered in the same way as {PRETRAINED_BANDS}.\
            Pretrained patch_embed layer may be misaligned with current bands"
        )

    model_args = {
        "embed_dim": 128,
        "mlp_multiplier": 4,
        "depths": (2, 2, 18, 2),
        "dropout": 0.4, 
        "n_heads": (4, 8, 16, 32),
        "drop_path": 0.1,
    }
    transformer = _create_hiera_transformer(
        "hiera_weather", pretrained_bands, bands, pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return transformer



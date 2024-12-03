# Copyright contributors to the Terratorch project


import torch
import logging
from functools import partial

from timm.models import FeatureInfo
from timm.models._builder import build_model_with_cfg
from timm.models._registry import generate_default_cfgs, register_model
from torch import nn, Tensor

from terratorch.datasets import HLSBands
from terratorch.models.backbones.select_patch_embed_weights import select_patch_embed_weights
from terratorch.datasets.utils import generate_bands_intervals
from terratorch.models.backbones.prithvi_mae import PrithviViT, PrithviMAE

PRETRAINED_BANDS = [
    HLSBands.BLUE,
    HLSBands.GREEN,
    HLSBands.RED,
    HLSBands.NIR_NARROW,
    HLSBands.SWIR_1,
    HLSBands.SWIR_2,
]

default_cfgs = generate_default_cfgs(
    {
        "prithvi_vit_100": {
            "hf_hub_id": "ibm-nasa-geospatial/Prithvi-100M",
            "hf_hub_filename": "Prithvi_EO_V1_100M.pt",
        },
        "prithvi_eo_v2_300": {
            "hf_hub_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
            "hf_hub_filename": "Prithvi_EO_V2_300M.pt",
        },
        "prithvi_eo_v2_300_tl": {
            "hf_hub_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
            "hf_hub_filename": "Prithvi_EO_V2_300M_TL.pt",
        },
        "prithvi_eo_v2_600": {
            "hf_hub_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
            "hf_hub_filename": "Prithvi_EO_V2_600M.pt",
        },
        "prithvi_eo_v2_600_tl": {
            "hf_hub_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL",
            "hf_hub_filename": "Prithvi_EO_V2_600M_TL.pt",
        },
        "prithvi_vit_tiny": {}
    }
)

def checkpoint_filter_fn_vit(
    state_dict, model: PrithviViT, pretrained_bands: list[HLSBands | int], model_bands: list[HLSBands | int]
) -> dict:
    """Encoder only model"""

    clean_dict = {}
    for k, v in state_dict.items():
        if "pos_embed" in k:
            v = model.pos_embed  # pos_embed depends on num_frames and is fixed.
        if "decoder" in k or "_dec" in k or k == "mask_token":
            continue  # Drop decoder weights

        if not model.temporal_encoding and "temporal_embed" in k:
            continue
        if not model.location_encoding and "location_embed" in k:
            continue

        if k.startswith("encoder."):
            clean_dict[k.replace("encoder.", "")] = v  # Convert Prithvi MAE to Prithvi ViT
        else:
            clean_dict[k] = v

    state_dict = clean_dict

    state_dict = select_patch_embed_weights(state_dict, model, pretrained_bands, model_bands)

    return state_dict


def checkpoint_filter_fn_mae(
    state_dict, model: PrithviMAE, pretrained_bands: list[HLSBands | int], model_bands: list[HLSBands | int]
) -> dict:
    """Encoder-decoder model"""

    clean_dict = {}
    for k, v in state_dict.items():
        # pos_embed depends on num_frames and is fixed.
        if "decoder_pos_embed" in k:
            v = model.decoder.decoder_pos_embed
        elif "pos_embed" in k:
            v = model.encoder.pos_embed

        if not model.encoder.temporal_encoding and "temporal_embed" in k:
            continue
        if not model.encoder.location_encoding and "location_embed" in k:
            continue

        if k.startswith("encoder.") or k.startswith("decoder."):
            clean_dict[k] = v  # Weights in Prithvi MAE format
        # Convert Prithvi V1 weights
        elif "decoder" in k or "_dec" in k or k == "mask_token":
            clean_dict["decoder." + k] = v
        else:
            clean_dict["encoder." + k] = v

    state_dict = clean_dict

    state_dict = select_patch_embed_weights(state_dict, model, pretrained_bands, model_bands)

    return state_dict


def pad_images(imgs: Tensor,patch_size: int, padding:str) -> Tensor:
    p = patch_size
    # h, w = imgs.shape[3], imgs.shape[4]
    t, h, w = imgs.shape[-3:]
    h_pad, w_pad = (p - h % p) % p, (p - w % p) % p  # Ensure padding is within bounds
    if h_pad > 0 or w_pad > 0:
        imgs = torch.stack([
            nn.functional.pad(img, (0, w_pad, 0, h_pad), mode=padding)
            for img in imgs  # Apply per image to avoid NotImplementedError from torch.nn.functional.pad
        ])
    return imgs

def _create_prithvi(
    variant: str,
    pretrained: bool = False,  # noqa: FBT001, FBT002
    pretrained_bands: list[HLSBands] | None = None,
    model_bands: list[HLSBands | int] | None = None,
    **kwargs,
) -> PrithviViT:
    if pretrained_bands is None:
        pretrained_bands = PRETRAINED_BANDS


    if model_bands is None:
        model_bands: list[HLSBands | int] = pretrained_bands
        logging.info(
            f"Model bands not passed. Assuming bands are ordered in the same way as {PRETRAINED_BANDS}.\
            Pretrained patch_embed layer may be misaligned with current bands"
        )
    else:
        model_bands = [HLSBands.try_convert_to_hls_bands_enum(b) for b in model_bands]

    padding = kwargs.get("padding", "none")
    patch_size = kwargs.get("patch_size", 16)

    # Little hack because VIT does not support timm's features_only
    encoder_only = kwargs.pop("features_only", False)

    model_bands = generate_bands_intervals(model_bands)

    kwargs["in_chans"] = len(model_bands)

    if encoder_only:
        prithvi_model_class = PrithviViT
        def checkpoint_filter_wrapper_fn(state_dict, model):
            return checkpoint_filter_fn_vit(state_dict, model, pretrained_bands, model_bands)
    else:
        prithvi_model_class = PrithviMAE
        def checkpoint_filter_wrapper_fn(state_dict, model):
            return checkpoint_filter_fn_mae(state_dict, model, pretrained_bands, model_bands)

    # When the pretrained configuration is not available in HF, we shift to
    # pretrained=False
    try:
        model = build_model_with_cfg(
            prithvi_model_class,
            variant,
            pretrained,
            pretrained_filter_fn=checkpoint_filter_wrapper_fn,
            pretrained_strict=True,
            **kwargs,
        )
    except RuntimeError:
        print(f"No pretrained configuration was found for the model {variant}.")
        model = build_model_with_cfg(
            prithvi_model_class,
            variant,
            False,
            pretrained_filter_fn=checkpoint_filter_wrapper_fn,
            pretrained_strict=True,
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

    if padding != "none":
        original_forward = model.forward
        original_forward_features = model.forward_features

        def pad_and_forward(forward_fn, patch_size, padding, *args, **kwargs):
            inputs = pad_images(args[0], patch_size, padding)
            return forward_fn(inputs, **kwargs)

        def forward_pad_images(*args, **kwargs):
            return pad_and_forward(original_forward, patch_size, padding, *args, **kwargs)

        def forward_features_pad_images(*args, **kwargs):
            return pad_and_forward(original_forward_features, patch_size, padding, *args, **kwargs)

        model.forward = forward_pad_images
        model.forward_features = forward_features_pad_images


    return model

def create_prithvi_vit_100(
    model_name: str,
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:
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
    bands: list[HLSBands | int] | None = None,
    **kwargs,
) -> PrithviViT:
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


def create_prithvi_vit_600(
    model_name: str,
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:
    """Prithvi ViT 600M"""
    pretrained_bands = PRETRAINED_BANDS
    if bands is None:
        bands = pretrained_bands
        logging.info(
            f"Model bands not passed. Assuming bands are ordered in the same way as {PRETRAINED_BANDS}.\
            Pretrained patch_embed layer may be misaligned with current bands"
        )
    model_args = {
        "patch_size": 14,
        "embed_dim": 1280,
        "depth": 32,
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
    bands: list[HLSBands | int] | None = None,
    **kwargs,
) -> PrithviViT:
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
) -> PrithviViT:
    return create_prithvi_vit_100("prithvi_vit_100", pretrained, bands, **kwargs)


@register_model
def prithvi_eo_v2_300(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:
    return create_prithvi_vit_300("prithvi_eo_v2_300", pretrained, bands, **kwargs)


@register_model
def prithvi_eo_v2_600(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:
    return create_prithvi_vit_600("prithvi_eo_v2_600", pretrained, bands, **kwargs)


@register_model
def prithvi_eo_v2_300_tl(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:
    return create_prithvi_vit_300("prithvi_eo_v2_300_tl", pretrained, bands, **kwargs)


@register_model
def prithvi_eo_v2_600_tl(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:
    return create_prithvi_vit_600("prithvi_eo_v2_600_tl", pretrained, bands, **kwargs)
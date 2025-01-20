# Copyright contributors to the Terratorch project

import torch
import logging
from torch import nn, Tensor
from timm.models import (FeatureInfo, load_model_config_from_hf, build_model_with_cfg, generate_default_cfgs,
                         register_model)

from terratorch.datasets import HLSBands
from terratorch.models.backbones.select_patch_embed_weights import select_patch_embed_weights
from terratorch.datasets.utils import generate_bands_intervals
from terratorch.models.backbones.prithvi_mae import PrithviViT, PrithviMAE

logger = logging.getLogger(__name__)

PRETRAINED_BANDS = [
    HLSBands.BLUE,
    HLSBands.GREEN,
    HLSBands.RED,
    HLSBands.NIR_NARROW,
    HLSBands.SWIR_1,
    HLSBands.SWIR_2,
]

PRITHVI_V1_MEAN = [775.0, 1081.0, 1229.0, 2497.0, 2204.0, 1611.0]
PRITHVI_V1_STD = [1282.0, 1270.0, 1399.0, 1368.0, 1292.0, 1155.0]
PRITHVI_V2_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
PRITHVI_V2_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]


def _cfg(**kwargs):
    return {
        "img_size": 224,
        "num_frames": 4,
        "patch_size": [1, 16, 16],
        "in_chans": 6,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "decoder_embed_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "mlp_ratio": 4,
        'mean': PRITHVI_V2_MEAN,
        'std': PRITHVI_V2_STD,
        "coords_encoding": [],
        "coords_scale_learn": False,
        "bands": PRETRAINED_BANDS,
        "mask_ratio": 0.75,
        "norm_pix_loss": False,
        **kwargs
    }


prithvi_cfgs = {
    "prithvi_eo_tiny": _cfg(num_frames=1, embed_dim=256, depth=4, num_heads=4,
                            decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4),
    "prithvi_eo_v1_100": _cfg(num_frames=3, mean=PRITHVI_V1_MEAN, std=PRITHVI_V1_STD),
    "prithvi_eo_v2_100": _cfg(),
    "prithvi_eo_v2_300": _cfg(embed_dim=1024, depth=24, num_heads=16),
    "prithvi_eo_v2_300_tl": _cfg(embed_dim=1024, depth=24, num_heads=16,
                                 coords_encoding=["time", "location"], coords_scale_learn=True),
    "prithvi_eo_v2_600": _cfg(embed_dim=1280, depth=32, num_heads=16, patch_size=[1, 14, 14]),
    "prithvi_eo_v2_600_tl": _cfg(embed_dim=1280, depth=32, num_heads=16, patch_size=[1, 14, 14],
                                 coords_encoding=["time", "location"], coords_scale_learn=True),
}

# Timm pretrained configs
default_cfgs = generate_default_cfgs(
    {
        "prithvi_eo_v1_100": {
            "hf_hub_id": "ibm-nasa-geospatial/Prithvi-EO-1.0-100M",
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
        logger.info(
            f"Model bands not passed. Assuming bands are ordered in the same way as {PRETRAINED_BANDS}.\
            Pretrained patch_embed layer may be misaligned with current bands"
        )
    else:
        model_bands = [HLSBands.try_convert_to_hls_bands_enum(b) for b in model_bands]

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

    if pretrained:
        assert variant in default_cfgs, (f"No pre-trained model found for variant {variant} "
                                            f"(pretrained models: {default_cfgs.keys()})")
        # Load pre-trained config from hf
        try:
            model_args = load_model_config_from_hf(default_cfgs[variant].default.hf_hub_id)[0]
            model_args.update(kwargs)
        except:
            logger.warning(f"No pretrained configuration was found on HuggingFace for the model {variant}."
                           f"Using random initialization.")
            model_args = prithvi_cfgs[variant].copy()
            model_args.update(kwargs)
    else:
        # Load default config
        model_args = prithvi_cfgs[variant].copy()
        model_args.update(kwargs)

    try:
        model = build_model_with_cfg(
            prithvi_model_class,
            variant,
            pretrained,
            pretrained_filter_fn=checkpoint_filter_wrapper_fn,
            pretrained_strict=True,
            **model_args,
        )
    except RuntimeError as e:
        if pretrained:
            logger.error(f"Failed to initialize the pre-trained model {variant} via timm, "
                         f"consider running the code with pretrained=False.")
        else:
            logger.error(f"Failed to initialize the model {variant} via timm.")
        raise e

    if encoder_only:
        default_out_indices = list(range(len(model.blocks)))
        out_indices = kwargs.pop("out_indices", default_out_indices)
        model.feature_info = FeatureInfo(model.feature_info, out_indices)
        model.encode_decode_forward = model.forward
        def forward_filter_indices(*args, **kwargs):
            features = model.forward_features(*args, **kwargs)
            return [features[i] for i in out_indices]
        model.forward = forward_filter_indices
        model.model_bands = model_bands
        model.pretrained_bands = pretrained_bands

    return model


def create_prithvi_from_config(
    model_name: str,
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:
    pretrained_bands = PRETRAINED_BANDS
    if bands is None:
        bands = pretrained_bands
        logger.info(
            f"Model bands not passed. Assuming bands are ordered in the same way as {PRETRAINED_BANDS}.\
            Pretrained patch_embed layer may be misaligned with current bands"
        )

    kwargs['num_frames'] = kwargs.pop('num_frames', 1)  # Set num frames to 1 if not present

    model = _create_prithvi(
        model_name,
        pretrained=pretrained,
        model_bands=bands,
        pretrained_bands=pretrained_bands,
        **kwargs,
    )
    
    return model


@register_model
def prithvi_vit_tiny(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:

    logger.warning(f'The model prithvi_vit_tiny was renamed to prithvi_eo_tiny. '
                    f'prithvi_vit_tiny will be removed in a future version.')

    return prithvi_eo_tiny(pretrained=pretrained, bands=bands, **kwargs)


@register_model
def prithvi_eo_tiny(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:

    return create_prithvi_from_config("prithvi_eo_tiny", pretrained, bands, **kwargs)


@register_model
def prithvi_vit_100(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:

    logger.warning(f'The model prithvi_vit_100 was renamed to prithvi_eo_v1_100. '
                    f'prithvi_vit_100 will be removed in a future version.')

    return prithvi_eo_v1_100(pretrained=pretrained, bands=bands, **kwargs)


@register_model
def prithvi_eo_v1_100(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:

    return create_prithvi_from_config("prithvi_eo_v1_100", pretrained, bands, **kwargs)


@register_model
def prithvi_eo_v2_300(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:

    return create_prithvi_from_config("prithvi_eo_v2_300", pretrained, bands, **kwargs)


@register_model
def prithvi_eo_v2_600(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:


    return create_prithvi_from_config("prithvi_eo_v2_600", pretrained, bands, **kwargs)


@register_model
def prithvi_eo_v2_300_tl(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:

    return create_prithvi_from_config("prithvi_eo_v2_300_tl", pretrained, bands, **kwargs)


@register_model
def prithvi_eo_v2_600_tl(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:

    return create_prithvi_from_config("prithvi_eo_v2_600_tl", pretrained, bands, **kwargs)

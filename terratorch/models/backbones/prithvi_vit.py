# Copyright contributors to the Terratorch project
import warnings

import torch
import logging
from torch import nn, Tensor
from terratorch.datasets import HLSBands
from terratorch.models.backbones.select_patch_embed_weights import select_patch_embed_weights
from terratorch.datasets.utils import generate_bands_intervals
from terratorch.models.backbones.prithvi_mae import PrithviViT, PrithviMAE
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
from huggingface_hub import hf_hub_download

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
pretrained_weights = {
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


def checkpoint_filter_fn_vit(
    state_dict, model: PrithviViT, pretrained_bands: list[HLSBands | int], model_bands: list[HLSBands | int]
) -> dict:
    """Encoder only model"""

    clean_dict = {}
    for k, v in state_dict.items():
        if "_timm_module." in k:  # Backwards compatibility for old model checkpoints
            k = k.replace("_timm_module.", "")

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

    state_dict = select_patch_embed_weights(state_dict, model, pretrained_bands, model_bands, encoder_only=True)

    return state_dict


def checkpoint_filter_fn_mae(
    state_dict, model: PrithviMAE, pretrained_bands: list[HLSBands | int], model_bands: list[HLSBands | int]
) -> dict:
    """Encoder-decoder model"""

    clean_dict = {}
    for k, v in state_dict.items():
        if "_timm_module." in k:  # Backwards compatibility for old model checkpoints
            k = k.replace("_timm_module.", "")

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

    state_dict = select_patch_embed_weights(state_dict, model, pretrained_bands, model_bands, encoder_only=False)

    return state_dict

def _create_prithvi(
    variant: str,
    pretrained: bool = False,  # noqa: FBT001, FBT002    
    model_bands: list[HLSBands | int] | None = None,
    ckpt_path: str = None,
    pretrained_bands: list[HLSBands | str | int] | None = None,
    num_frames: int = 1,
    encoder_only: bool = True,
    **kwargs,
) -> PrithviViT | PrithviMAE:
    """
    Build PrithviViT and PrithviMAE models.
    By default, encoder_only is set to True and a ViT is returned.
    """

    # Load default config
    model_args = prithvi_cfgs[variant].copy()

    # Backwards compatibility from timm (pretrained_cfg_overlay={"file": "<path to weights>"}) TODO: Remove before v1.0
    if "pretrained_cfg_overlay" in kwargs:
        warnings.warn(f"pretrained_cfg_overlay is deprecated and will be removed in a future version, "
                      f"use ckpt_path=<file path> instead.", DeprecationWarning, stacklevel=2)
        if ckpt_path is not None:
            warnings.warn(f"pretrained_cfg_overlay and ckpt_path are provided, ignoring pretrained_cfg_overlay.")
        elif "file" not in kwargs["pretrained_cfg_overlay"]:
            warnings.warn("pretrained_cfg_overlay does not include 'file path', ignoring pretrained_cfg_overlay.")
        else:
            ckpt_path = kwargs.pop("pretrained_cfg_overlay")["file"]

    pretrained_bands = pretrained_bands or model_args.get("bands", PRETRAINED_BANDS)

    if model_bands is None:
        model_bands: list[HLSBands | int] = pretrained_bands
        logger.info(f"Model bands not passed. Assuming bands are ordered in the same way as {pretrained_bands}."
                    f"Pretrained patch_embed layer may be misaligned with current bands")
    else:
        model_bands = [HLSBands.try_convert_to_hls_bands_enum(b) for b in model_bands]
        model_bands = generate_bands_intervals(model_bands)

    kwargs["in_chans"] = len(model_bands)
    kwargs["num_frames"] = num_frames
    model_args.update(kwargs)

    if encoder_only:
        prithvi_model_class = PrithviViT
        checkpoint_filter_wrapper_fn = checkpoint_filter_fn_vit
    else:
        prithvi_model_class = PrithviMAE
        checkpoint_filter_wrapper_fn = checkpoint_filter_fn_mae

    model = prithvi_model_class(**model_args)

    if pretrained:
        if ckpt_path is not None:
            # Load model from checkpoint
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint_filter_wrapper_fn(state_dict, model, pretrained_bands, model_bands)

            loaded_keys = model.load_state_dict(state_dict, strict=False)
            if loaded_keys.missing_keys:
                logger.warning(f"Missing keys in ckpt_path {ckpt_path}: {loaded_keys.missing_keys}")
            if loaded_keys.unexpected_keys:
                logger.warning(f"Unexpected keys in ckpt_path {ckpt_path}: {loaded_keys.unexpected_keys}")
        else:
            assert variant in pretrained_weights, (f"No pre-trained model found for variant {variant} "
                                                   f"(pretrained models: {pretrained_weights.keys()})")

            try:
                # Download config.json to count model downloads
                _ = hf_hub_download(repo_id=pretrained_weights[variant]["hf_hub_id"], filename="config.json")
                # Load model from Hugging Face
                pretrained_path = hf_hub_download(repo_id=pretrained_weights[variant]["hf_hub_id"],
                                                  filename=pretrained_weights[variant]["hf_hub_filename"])
                state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
                state_dict = checkpoint_filter_wrapper_fn(state_dict, model, pretrained_bands, model_bands)
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                logger.error(f"Failed to load the pre-trained weights for {variant}.")
                raise e
    elif ckpt_path is not None:
        logger.warning(f"ckpt_path is provided but pretrained is set to False, ignoring ckpt_path {ckpt_path}.")

    model.model_bands = model_bands
    model.pretrained_bands = pretrained_bands

    assert encoder_only or "out_indices" not in kwargs, "out_indices provided for a MAE model."
    if encoder_only:
        default_out_indices = list(range(len(model.blocks)))
        out_indices = kwargs.pop("out_indices", default_out_indices)

        def forward_filter_indices(*args, **kwargs):
            features = model.forward_features(*args, **kwargs)
            return [features[i] for i in out_indices]

        model.forward = forward_filter_indices
        model.out_indices = out_indices

    return model


@ TERRATORCH_BACKBONE_REGISTRY.register
def prithvi_eo_tiny(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:

    return _create_prithvi("prithvi_eo_tiny", pretrained=pretrained, **dict({"model_bands": bands}, **kwargs))


@ TERRATORCH_BACKBONE_REGISTRY.register
def prithvi_eo_v1_100(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:

    return _create_prithvi("prithvi_eo_v1_100", pretrained=pretrained, **dict({"model_bands": bands}, **kwargs))


@ TERRATORCH_BACKBONE_REGISTRY.register
def prithvi_eo_v2_300(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:

    return _create_prithvi("prithvi_eo_v2_300", pretrained=pretrained, **dict({"model_bands": bands}, **kwargs))


@ TERRATORCH_BACKBONE_REGISTRY.register
def prithvi_eo_v2_600(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:

    return _create_prithvi("prithvi_eo_v2_600", pretrained=pretrained, **dict({"model_bands": bands}, **kwargs))


@ TERRATORCH_BACKBONE_REGISTRY.register
def prithvi_eo_v2_300_tl(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:

    return _create_prithvi("prithvi_eo_v2_300_tl", pretrained=pretrained, **dict({"model_bands": bands}, **kwargs))


@ TERRATORCH_BACKBONE_REGISTRY.register
def prithvi_eo_v2_600_tl(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:

    return _create_prithvi("prithvi_eo_v2_600_tl", pretrained=pretrained, **dict({"model_bands": bands}, **kwargs))


# TODO: Remove prithvi_vit_tiny and prithvi_vit_100 before version 1.0.
@ TERRATORCH_BACKBONE_REGISTRY.register
def prithvi_vit_tiny(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:

    warnings.warn(f"The model prithvi_vit_tiny was renamed to prithvi_eo_tiny. "
                  f"prithvi_vit_tiny will be removed in a future version.", FutureWarning)

    return prithvi_eo_tiny(pretrained=pretrained, **dict({"model_bands": bands}, **kwargs))


@ TERRATORCH_BACKBONE_REGISTRY.register
def prithvi_vit_100(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:
    warnings.warn("The model prithvi_vit_100 was renamed to prithvi_eo_v1_100. "
                  "prithvi_vit_100 will be removed in a future version.", FutureWarning)

    return prithvi_eo_v1_100(pretrained=pretrained, **dict({"model_bands": bands}, **kwargs))


# TODO: Remove timm_ errors before version v1.0.
@ TERRATORCH_BACKBONE_REGISTRY.register
def timm_prithvi_eo_v1_100(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> None:
    raise ValueError("The Prithvi models were moved to the terratorch registry. "
                     "Please remove the timm_ prefix from the model name.")


@ TERRATORCH_BACKBONE_REGISTRY.register
def timm_prithvi_eo_v2_300(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> None:
    raise ValueError("The Prithvi models were moved to the terratorch registry. "
                     "Please remove the timm_ prefix from the model name.")


@ TERRATORCH_BACKBONE_REGISTRY.register
def timm_prithvi_eo_v2_600(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> None:
    raise ValueError("The Prithvi models were moved to the terratorch registry. "
                     "Please remove the timm_ prefix from the model name.")

@ TERRATORCH_BACKBONE_REGISTRY.register
def timm_prithvi_eo_v2_300_tl(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> None:
    raise ValueError("The Prithvi models were moved to the terratorch registry. "
                     "Please remove the timm_ prefix from the model name.")


@ TERRATORCH_BACKBONE_REGISTRY.register
def timm_prithvi_eo_v2_600_tl(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> None:
    raise ValueError("The Prithvi models were moved to the terratorch registry. "
                     "Please remove the timm_ prefix from the model name.")

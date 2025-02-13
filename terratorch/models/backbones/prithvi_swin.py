# Copyright contributors to the Terratorch project

"""This module handles registering prithvi_swin models into timm.
"""

import logging
import warnings
from collections import OrderedDict

import torch
from timm.models import SwinTransformer
from timm.models._builder import build_model_with_cfg
from timm.models._registry import generate_default_cfgs, register_model

from terratorch.datasets.utils import HLSBands
from terratorch.models.backbones.select_patch_embed_weights import select_patch_embed_weights
from terratorch.models.backbones.swin_encoder_decoder import MMSegSwinTransformer
from terratorch.datasets.utils import generate_bands_intervals

logger = logging.getLogger(__name__)

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
        "prithvi_swin_B": {},
        "prithvi_swin_L": {},
    }
)

def convert_weights_swin2mmseg(ckpt):
    # from https://github.com/open-mmlab/mmsegmentation/blob/main/tools/model_converters/swin2mmseg.py
    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith("head"):
            continue
        elif k.startswith("layers"):
            new_v = v
            if "attn." in k:
                new_k = k.replace("attn.", "attn.w_msa.")
            elif "mlp." in k:
                if "mlp.fc1." in k:
                    new_k = k.replace("mlp.fc1.", "ffn.layers.0.0.")
                elif "mlp.fc2." in k:
                    new_k = k.replace("mlp.fc2.", "ffn.layers.1.")
                else:
                    new_k = k.replace("mlp.", "ffn.")
            elif "downsample" in k:
                new_k = k
                if "reduction." in k:
                    new_v = correct_unfold_reduction_order(v)
                elif "norm." in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace("layers", "stages", 1)
        elif k.startswith("patch_embed"):
            new_v = v
            if "proj" in k:
                new_k = k.replace("proj", "projection")
            else:
                new_k = k
        else:
            new_v = v
            new_k = k

        new_ckpt[new_k] = new_v

    return new_ckpt


def weights_are_swin_implementation(state_dict: dict[str, torch.Tensor]):
    # if keys start with 'encoder', treat it as the swin implementation
    for k in state_dict.keys():
        if k.startswith("encoder."):
            return True
    return False

# Identifying when a prefix is being used in the checkpoints
# it will identify it. 
def identify_prefix(state_dict, model):

    state_dict_ = model.state_dict()

    prefix = list(state_dict.keys())[0].replace(list(state_dict_.keys())[0], "")

    return prefix 

# Replacing "_" with "." when necessary.
def adapt_prefix(key):
    if key.startswith("stages_"):
        key = key.replace("stages_", "stages.")
    return key 

def checkpoint_filter_fn(state_dict: dict[str, torch.Tensor], model: torch.nn.Module, pretrained_bands, model_bands):
    """convert patch embedding weight from manual patchify + linear proj to conv"""
    if "head.fc.weight" in state_dict:
        return state_dict

    if "state_dict" in state_dict:
        _state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        _state_dict = state_dict["model"]
    else:
        _state_dict = state_dict

    # strip prefix of state_dict
    if next(iter(_state_dict.keys())).startswith("module."):
        _state_dict = {k[7:]: v for k, v in _state_dict.items()}

    if weights_are_swin_implementation(_state_dict):
        # keep only encoder weights
        state_dict = OrderedDict()
        for k, v in _state_dict.items():
            if k.startswith("encoder."):
                state_dict[k[8:]] = v
            elif not k.startswith("decoder"):
                state_dict[k] = v
        state_dict = convert_weights_swin2mmseg(state_dict)
    else:
        # keep only encoder weights
        state_dict = OrderedDict()

        for k, v in _state_dict.items():
            if k.startswith("backbone."):
                state_dict[k[9:]] = v
            else:
                state_dict[k] = v

    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    # Sometimes the checkpoints can contain an unexpected prefix that must be
    # removed. 
    prefix = identify_prefix(state_dict, model)

    for table_key in relative_position_bias_table_keys:

        # The checkpoints can sometimes contain unexpected prefixes.
        # TODO Guarantee that it will not happen in the future.
        if prefix:
            table_key_ = table_key.replace(prefix, "")
        else:
            table_key_  = table_key

        # In an unexpected behavior, the prefix can sometimes contain
        # "_" or ".". We are enforcing ".". 
        # TODO Standardize it. 
        table_key_ = adapt_prefix(table_key_)

        table_pretrained = state_dict[table_key]

        table_current = model.state_dict()[table_key_]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            warnings.warn(f"Error in loading {table_key}, pass", stacklevel=1)
        elif L1 != L2:
            S1 = int(L1**0.5)
            S2 = int(L2**0.5)
            table_pretrained_resized = torch.nn.functional.interpolate(
                table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                size=(S2, S2),
                mode="bicubic",
            )
            state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0).contiguous()

    if hasattr(model.head.fc, "weight"):
        state_dict["head.fc.weight"] = model.head.fc.weight.detach().clone()
        state_dict["head.fc.bias"] = model.head.fc.bias.detach().clone()

    state_dict = select_patch_embed_weights(state_dict, model, pretrained_bands, model_bands)
    return state_dict


def _create_swin_mmseg_transformer(
    variant: str,
    pretrained_bands: list[HLSBands] | None = None,
    model_bands: list[HLSBands | int] | None = None,
    pretrained: bool = False,  # noqa: FBT002, FBT001
    **kwargs,
):
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
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get("depths", (1, 1, 3, 1))))
    out_indices = kwargs.pop("out_indices", default_out_indices)

    # the current swin model is not multitemporal
    if "num_frames" in kwargs:
        kwargs = {k: v for k, v in kwargs.items() if k != "num_frames"}

    model_bands = generate_bands_intervals(model_bands)
    kwargs["in_chans"] = len(model_bands)

    def checkpoint_filter_wrapper_fn(state_dict, model):
        return checkpoint_filter_fn(state_dict, model, pretrained_bands, model_bands)

    # TODO Totally remove the usage of timm for Swin in the future.
    # When the pretrained configuration is not available in HF, we shift to 
    # pretrained=False
    try:
        model: MMSegSwinTransformer = build_model_with_cfg(
            MMSegSwinTransformer,
            variant,
            pretrained,
            pretrained_filter_fn=checkpoint_filter_wrapper_fn,
            pretrained_strict=False,
            feature_cfg={"flatten_sequential": True, "out_indices": out_indices},
            **kwargs,
        )
    except RuntimeError:
        print(f"No pretrained configuration was found for the model {variant}.")
        model: MMSegSwinTransformer = build_model_with_cfg(
            MMSegSwinTransformer,
            variant,
            False,
            pretrained_filter_fn=checkpoint_filter_wrapper_fn,
            pretrained_strict=False,
            feature_cfg={"flatten_sequential": True, "out_indices": out_indices},
            **kwargs,
        )

    model.pretrained_bands = pretrained_bands
    model.model_bands = model_bands

    def prepare_features_for_image_model(x):
        return [
            layer_output.permute(0, 3, 1, 2).contiguous()
            for layer_output in x
        ]

    # add permuting here
    model.prepare_features_for_image_model = prepare_features_for_image_model
    return model


@register_model
def prithvi_swin_B(
    pretrained: bool = False,  # noqa: FBT002, FBT001
    pretrained_bands: list[HLSBands] | None = None,
    bands: list[HLSBands | int] | None = None,
    **kwargs,
) -> SwinTransformer:
    """Prithvi Swin B"""
    if pretrained_bands is None:
        pretrained_bands = PRETRAINED_BANDS
    if bands is None:
        bands = pretrained_bands
        logger.info(
            f"Model bands not passed. Assuming bands are ordered in the same way as {PRETRAINED_BANDS}.\
            Pretrained patch_embed layer may be misaligned with current bands"
        )

    model_args = {
        "patch_size": 4,
        "window_size": 7,
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "in_chans": 6,
        "num_heads": (4, 8, 16, 32),
    }
    transformer = _create_swin_mmseg_transformer(
        "prithvi_swin_B", pretrained_bands, bands, pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return transformer


@register_model
def prithvi_swin_L(
    pretrained: bool = False,  # noqa: FBT002, FBT001
    pretrained_bands: list[HLSBands] | None = None,
    bands: list[HLSBands | int] | None = None,
    **kwargs,
) -> SwinTransformer:
    """Prithvi Swin L"""
    if pretrained_bands is None:
        pretrained_bands = PRETRAINED_BANDS
    if bands is None:
        bands = pretrained_bands
        logger.info(
            f"Model bands not passed. Assuming bands are ordered in the same way as {PRETRAINED_BANDS}.\
            Pretrained patch_embed layer may be misaligned with current bands"
        )

    model_args = {
        "patch_size": 4,
        "window_size": 7,
        "embed_dim": 192,
        "depths": (2, 2, 18, 2),
        "in_chans": 6,
        "num_heads": (6, 12, 24, 48),
    }
    transformer = _create_swin_mmseg_transformer(
        "prithvi_swin_L", pretrained_bands, bands, pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return transformer

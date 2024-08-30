# Copyright contributors to the Terratorch project

import warnings

import torch
from torch import nn

from terratorch.datasets import HLSBands


def patch_embed_weights_are_compatible(model_patch_embed: torch.Tensor, checkpoint_patch_embed: torch.Tensor) -> bool:
    # check all dimensions are the same except for channel dimension
    if len(model_patch_embed.shape) != len(checkpoint_patch_embed.shape):
        return False
    model_shape = [model_patch_embed.shape[i] for i in range(len(model_patch_embed.shape)) if i != 1]
    checkpoint_shape = [checkpoint_patch_embed.shape[i] for i in range(len(checkpoint_patch_embed.shape)) if i != 1]

    return model_shape == checkpoint_shape

def prithvi_select_patch_embed_weights(
    state_dict: dict, model: nn.Module, pretrained_bands: list[HLSBands | int], model_bands: list[HLSBands | int]
) -> dict:
    """Filter out the patch embedding weights according to the bands being used.
    If a band exists in the pretrained_bands, but not in model_bands, drop it.
    If a band exists in model_bands, but not pretrained_bands, randomly initialize those weights.


    Args:
        state_dict (dict): State Dict
        model (nn.Module): Model to load the weights onto.
        pretrained_bands (list[HLSBands]): List of bands the model was pretrained on, in the correct order.
        model_bands (list[HLSBands]): List of bands the model is going to be finetuned on, in the correct order

    Returns:
        dict: New state dict
    """
    _possible_keys_for_proj_weight = {
        "patch_embed.proj.weight",
        "module.patch_embed.proj.weight",
        "patch_embed.projection.weight",
        "module.patch_embed.projection.weight",
    }
    patch_embed_proj_weight_key = state_dict.keys() & _possible_keys_for_proj_weight
    if len(patch_embed_proj_weight_key) == 0:
        msg = "Could not find key for patch embed weight"
        raise Exception(msg)
    if len(patch_embed_proj_weight_key) > 1:
        msg = "Too many matches for key for patch embed weight"
        raise Exception(msg)

    # extract the single element from the set
    (patch_embed_proj_weight_key,) = patch_embed_proj_weight_key
    patch_embed_weight = state_dict[patch_embed_proj_weight_key]

    temp_weight = model.state_dict()[patch_embed_proj_weight_key].clone()

    # only do this if the patch size and tubelet size match. If not, start with random weights
    if temp_weight.shape == patch_embed_weight.shape:
        torch.nn.init.xavier_uniform_(temp_weight.view([temp_weight.shape[0], -1]))
        for index, band in enumerate(model_bands):
            if band in pretrained_bands:
                temp_weight[:, index] = patch_embed_weight[:, pretrained_bands.index(band)]
    else:
        warnings.warn(
            f"Incompatible shapes between patch embedding of model {temp_weight.shape} and of checkpoint {patch_embed_weight.shape}",
            stacklevel=1,
        )

    state_dict[patch_embed_proj_weight_key] = temp_weight
    return state_dict
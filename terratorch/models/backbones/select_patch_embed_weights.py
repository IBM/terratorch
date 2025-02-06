# Copyright contributors to the Terratorch project

import logging
import warnings

import torch
from torch import nn

from terratorch.datasets import HLSBands, OpticalBands, SARBands
import collections

def patch_embed_weights_are_compatible(model_patch_embed: torch.Tensor, checkpoint_patch_embed: torch.Tensor) -> bool:
    # check all dimensions are the same except for channel dimension
    if len(model_patch_embed.shape) != len(checkpoint_patch_embed.shape):
        return False

    model_shape = [model_patch_embed.shape[i] for i in range(len(model_patch_embed.shape)) if i != 1]
    checkpoint_shape = [checkpoint_patch_embed.shape[i] for i in range(len(checkpoint_patch_embed.shape)) if i != 1]
    return model_shape == checkpoint_shape

def get_state_dict(state_dict):

    if "state_dict" in state_dict.keys():
        return state_dict["state_dict"]
    else:
        return state_dict

def get_proj_key(state_dict, return_prefix=False):

    prefix_key = None 

    for key in state_dict.keys():
        if key.endswith('patch_embed.proj.weight') or key.endswith('patch_embed.projection.weight'):
            proj_key = key
            break

    if return_prefix and prefix_key:

        for sufix in ['patch_embed.proj.weight', 'patch_embed.projection.weight']:
            if proj_key.endswith(sufix):
                prefix = proj_key.replace(sufix, "")
                break
    else:
        prefix = None

    return proj_key, prefix

def remove_prefixes(state_dict, prefix):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace(prefix, "")] = v
    return new_state_dict

def select_patch_embed_weights(
    state_dict: dict, model: nn.Module, pretrained_bands: list[HLSBands | int | OpticalBands| SARBands], model_bands: list[HLSBands | int | OpticalBands| SARBands], proj_key: str | None = None
) -> dict:
    """Filter out the patch embedding weights according to the bands being used.
    If a band exists in the pretrained_bands, but not in model_bands, drop it.
    If a band exists in model_bands, but not pretrained_bands, randomly initialize those weights.


    Args:
        state_dict (dict): State Dict
        model (nn.Module): Model to load the weights onto.
        pretrained_bands (list[HLSBands | int]): List of bands the model was pretrained on, in the correct order.
        model_bands (list[HLSBands | int]): List of bands the model is going to be finetuned on, in the correct order
        proj_key (str, optional): Key to patch embedding projection weight in state_dict.

    Returns:
        dict: New state dict
    """
    if (type(pretrained_bands) == type(model_bands)) | (type(pretrained_bands) == int) | (type(model_bands) == int):

        state_dict = get_state_dict(state_dict) 

        if proj_key is None:
            # Search for patch embedding weight in state dict
            proj_key, prefix = get_proj_key(state_dict, return_prefix=True)
        if proj_key is None or proj_key not in state_dict:
            raise Exception("Could not find key for patch embed weight in state_dict.")

        patch_embed_weight = state_dict[proj_key]

        # It seems `proj_key` can have different names for 
        # the checkpoint and the model instance
        proj_key_, _  = get_proj_key(model.state_dict())
        temp_weight = model.state_dict()[proj_key_].clone()

        # only do this if the patch size and tubelet size match. If not, start with random weights
        if patch_embed_weights_are_compatible(temp_weight, patch_embed_weight):
            torch.nn.init.xavier_uniform_(temp_weight.view([temp_weight.shape[0], -1]))
            for index, band in enumerate(model_bands):
                if band in pretrained_bands:
                    logging.info(f"Loaded weights for {band} in position {index} of patch embed")
                    temp_weight[:, index] = patch_embed_weight[:, pretrained_bands.index(band)]
        else:
            warnings.warn(
                f"Incompatible shapes between patch embedding of model {temp_weight.shape} and\
                of checkpoint {patch_embed_weight.shape}",
                category=UserWarning,
                stacklevel=1,
            )

        state_dict[proj_key] = temp_weight

    if prefix:
        state_dict = remove_prefixes(state_dict, prefix)

    return state_dict

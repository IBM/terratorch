# reference torchgeo https://torchgeo.readthedocs.io/en/latest/_modules/torchgeo/models/dofa.html#DOFA
import torch
import torchgeo.models.dofa as dofa
import logging
from collections.abc import Callable
from functools import partial
import huggingface_hub
import torch.nn as nn
from typing import List
import huggingface_hub
from torchvision.models._api import Weights, WeightsEnum


dofa_model_registry: dict[str, Callable] = {}

def register_dofa_model(constructor: Callable):
    dofa_model_registry[constructor.__name__] = constructor
    return constructor

class DOFAEncoderWrapper(nn.Module):

    """
    A wrapper for DOFA models from torchgeo to return only the forward pass of the encoder 
    Attributes:
        dofa_model (DOFA): The instantiated dofa model
    Methods:
        forward(x: List[torch.Tensor], wavelengths: list[float]) -> torch.Tensor:
            Forward pass for embeddings with specified indices.
    """

    def __init__(self, dofa_model, weights=None) -> None:
        """
        Args:
            dofa_model (DOFA): The decoder module to be wrapped.
            weights ()
        """
        super().__init__()
        self.dofa_model = dofa_model
        self.weights = weights
        if self.dofa_model.global_pool:
            self.out_channels = self.embed_dim
        else:
            self.out_channels = [self.embed_dim for x in range(self.depth)]

    def forward(self, x: List[torch.Tensor], wavelengths: list[float]) -> torch.Tensor:
        return self.dofa_model.forward_features(x, wavelengths)


@register_dofa_model
def dofa_small_patch16_224(**kwargs):
    model = dofa.dofa_small_patch16_224(**kwargs)
    return DOFAEncoderWrapper(model)

@register_dofa_model
def dofa_base_patch16_224(**kwargs):
    model = dofa.dofa_small_patch16_224(**kwargs)
    return DOFAEncoderWrapper(model, DOFABase16_Weights)

@register_dofa_model
def dofa_large_patch16_224(**kwargs):
    model = dofa.dofa_large_patch16_224(**kwargs)
    return DOFAEncoderWrapper(model, DOFALarge16_Weights)

@register_dofa_model
def dofa_huge_patch16_224(**kwargs):
    model = dofa.dofa_huge_patch16_224(**kwargs)
    return DOFAEncoderWrapper(model)


def load_dofa_weights(model: nn.Module, ckpt_data: str) -> nn.Module:
    if ckpt_data.find("https://hf.co/") > -1:
        ckpt_data = huggingface_hub.hf_hub_download(repo_id="torchgeo/dofa", filename=ckpt_data.split('/')[-1])
    checkpoint_model = torch.load(ckpt_data, map_location="cpu")
    state_dict = model.state_dict()

    for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                logging.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

    # if input_size != 224:
    #     if (
    #         "pos_embed" in checkpoint_model
    #         and checkpoint_model["pos_embed"].shape != state_dict["pos_embed"].shape
    #     ):
    #         logging.info("Removing key pos_embed from pretrained checkpoint")
    #         del checkpoint_model["pos_embed"]

    # checkpoint_model = select_patch_embed_weights(checkpoint_model, model, PRETRAINED_BANDS, model_bands)
    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)

    logging.info(msg)
    return model



def create_model(model_name: str, ckpt_path: str | None = None, **kwargs):
    try:
        constructor: Callable = dofa_model_registry[model_name]
    except KeyError as e:
        msg = f"Model {model_name} not in registry. Possible models are {list(dofa_model_registry.keys())}"
        raise Exception(msg) from e

    # if bands is None:
    #     bands: list[HLSBands | int | str] = PRETRAINED_BANDS
    #     logging.info(
    #         f"Model bands not passed. Assuming bands are ordered in the same way as {PRETRAINED_BANDS}.\
    #         Pretrained patch_embed layer may be misaligned with current bands"
    #     )
    # else:
    #     bands = [HLSBands.try_convert_to_hls_bands_enum(b) for b in bands]
    # kwargs["in_chans"] = len(bands)
    model = constructor(**kwargs)

    if ckpt_path:
        load_dofa_weights(model, ckpt_path)

    # # ScaleMAE does not use the pos_embed within ViT
    # model.pos_embed.requires_grad = False

    return model

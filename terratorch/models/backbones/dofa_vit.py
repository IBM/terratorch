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
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
import pdb

waves_list= {
  "COASTAL_AEROSOL": 0.44,
  "BLUE": 0.49,
  "GREEN": 0.56,
  "RED": 0.665,
  "RED_EDGE_1": 0.705,
  "RED_EDGE_2": 0.74, 
  "RED_EDGE_3": 0.783,
  "NIR_BROAD": 0.832,
  "NIR_NARROW": 0.864,
  "WATER_VAPOR": 0.945,
  "CIRRUS": 1.373,
  "SWIR_1": 1.61,
  "SWIR_2": 2.20,
  "THEMRAL_INFRARED_1": 10.90,
  "THEMRAL_INFRARED_12": 12.00, 
  "VV": 3.75,
  "VH": 3.75,
  "ASC_VV": 3.75,
  "ASC_VH": 3.75,
  "DSC_VV": 3.75,
  "DSC_VH": 3.75,
  "VV-VH": 3.75
}


class DOFAEncoderWrapper(nn.Module):

    """
    A wrapper for DOFA models from torchgeo to return only the forward pass of the encoder 
    Attributes:
        dofa_model (DOFA): The instantiated dofa model
    Methods:
        forward(x: List[torch.Tensor], wavelengths: list[float]) -> torch.Tensor:
            Forward pass for embeddings with specified indices.
    """

    def __init__(self, dofa_model, wavelengths, weights=None, out_indices=None) -> None:
        """
        Args:
            dofa_model (DOFA): The decoder module to be wrapped.
            weights ()
        """
        super().__init__()
        self.dofa_model = dofa_model
        self.weights = weights
        self.wavelengths = wavelengths

        self.out_indices = out_indices if out_indices else [-1]
        self.out_channels = [self.dofa_model.patch_embed.embed_dim] * len(self.out_indices)
        
    def forward(self, x: List[torch.Tensor], **kwargs) -> torch.Tensor:
        
        wavelist = torch.tensor(self.wavelengths, device=x.device).float()

        x, _ = self.dofa_model.patch_embed(x, wavelist)

        x = x + self.dofa_model.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.dofa_model.cls_token + self.dofa_model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        outs = []
        # apply Transformer blocks
        for i, block in enumerate(self.dofa_model.blocks):
            x = block(x)
            if i in self.out_indices:
                outs.append(x)
            elif (i == (len(self.dofa_model.blocks)-1)) & (-1 in self.out_indices):
                outs.append(x)
                
        return tuple(outs)

def get_wavelenghts(model_bands):

    wavelengths = [waves_list[x.split('.')[-1]] for x in model_bands]

    return wavelengths
    

@TERRATORCH_BACKBONE_REGISTRY.register
def dofa_small_patch16_224(model_bands, input_size = 224, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = None, out_indices: list | None = None, **kwargs):
    model = dofa.dofa_small_patch16_224(**kwargs)
    input_size = kwargs["img_size"] if "img_size" in kwargs else 224
    if pretrained:
        model = load_dofa_weights(model, ckpt_data, weights, input_size)
    wavelengths = get_wavelenghts(model_bands)
    
    return DOFAEncoderWrapper(model, wavelengths, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def dofa_base_patch16_224(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = dofa.DOFABase16_Weights.DOFA_MAE, out_indices: list | None = None, **kwargs):
    model = dofa.dofa_base_patch16_224(**kwargs)
    input_size = kwargs["img_size"] if "img_size" in kwargs else 224
    if pretrained:
        model = load_dofa_weights(model, ckpt_data, weights, input_size)
    wavelengths = get_wavelenghts(model_bands)
    
    return DOFAEncoderWrapper(model, wavelengths, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def dofa_large_patch16_224(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = dofa.DOFALarge16_Weights.DOFA_MAE, out_indices: list | None = None, **kwargs):
    model = dofa.dofa_large_patch16_224(**kwargs)
    input_size = kwargs["img_size"] if "img_size" in kwargs else 224
    if pretrained:
        model = load_dofa_weights(model, ckpt_data, weights, input_size)
    wavelengths = get_wavelenghts(model_bands)
    
    return DOFAEncoderWrapper(model, wavelengths, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def dofa_huge_patch16_224(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = None, out_indices: list | None = None, **kwargs):
    model = dofa.dofa_huge_patch16_224(**kwargs)
    input_size = kwargs["img_size"] if "img_size" in kwargs else 224
    if pretrained:
        model = load_dofa_weights(model, ckpt_data, weights, input_size)
    wavelengths = get_wavelenghts(model_bands)
    
    return DOFAEncoderWrapper(model, wavelengths, weights, out_indices)

def load_dofa_weights(model: nn.Module, ckpt_data: str | None = None,  weights: Weights | None = None, input_size = 224) -> nn.Module:
    state_dict = model.state_dict()
    print("Loading weights")
    if ckpt_data is not None:
        if ckpt_data.find("https://hf.co/") > -1:
            repo_id = ckpt_data.split("/resolve/")[0].replace("https://hf.co/", '')
            filename = ckpt_data.split("/")[-1]
            ckpt_data = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint_model = torch.load(ckpt_data, map_location="cpu", weights_only=True)

        for k in ["head.weight", "head.bias"]:
                if (
                    k in checkpoint_model
                    and checkpoint_model[k].shape != state_dict[k].shape
                ):
                    logging.info(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
        if input_size != 224:
            if (
                "pos_embed" in checkpoint_model
                and checkpoint_model["pos_embed"].shape != state_dict["pos_embed"].shape
            ):
                logging.info("Removing key pos_embed from pretrained checkpoint")
                del checkpoint_model["pos_embed"]

    
        msg = model.load_state_dict(checkpoint_model, strict=False)
    
        logging.info(msg)
    else:
        if weights is not None:
            
            checkpoint_model = weights.get_state_dict(progress=True)
            allowed_missing_keys =  {'fc_norm.weight', 'fc_norm.bias', 'head.weight', 'head.bias'}
            if input_size != 224:
                if (
                    "pos_embed" in checkpoint_model
                    and checkpoint_model["pos_embed"].shape != state_dict["pos_embed"].shape
                ):
                    logging.info("Removing key pos_embed from pretrained checkpoint")
                    del checkpoint_model["pos_embed"]
                    allowed_missing_keys.add('pos_embed')
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
            logging.info(f"Weights loaded.")
            # Both fc_norm and head are generated dynamically
            assert set(missing_keys) <= allowed_missing_keys
            assert not unexpected_keys
        else:
            print("No weights to load.")
            
    return model


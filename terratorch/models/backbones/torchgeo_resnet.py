# reference torchgeo https://torchgeo.readthedocs.io/en/stable/_modules/torchgeo/models/vit.html

import torchgeo.models.resnet as resnet
from torchgeo.models.resnet import ResNet, ResNet18_Weights, ResNet50_Weights, ResNet152_Weights, resnet18, resnet50, resnet152
import logging
from collections.abc import Callable
from functools import partial
import huggingface_hub
import torch.nn as nn
from typing import List
import huggingface_hub
from torchvision.models._api import Weights, WeightsEnum
import torch

torchgeo_resnet_model_registry: dict[str, Callable] = {}

def register_resnet_model(constructor: Callable):
    torchgeo_resnet_model_registry[constructor.__name__] = constructor
    return constructor

class ResNetEncoderWrapper(nn.Module):

    """
    A wrapper for ViT models from torchgeo to return only the forward pass of the encoder 
    Attributes:
        satlas_model (VisionTransformer): The instantiated dofa model
        weights
    Methods:
        forward(x: List[torch.Tensor], wavelengths: list[float]) -> torch.Tensor:
            Forward pass for embeddings with specified indices.
    """

    def __init__(self, resnet_model, weights=None) -> None:
        """
        Args:
            dofa_model (DOFA): The decoder module to be wrapped.
            weights ()
        """
        super().__init__()
        self.resnet_model = vit_model
        self.weights = weights

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.resnet_model(x)


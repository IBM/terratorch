# Copyright contributors to the Terratorch project
import logging
import timm
import torch
from torch import nn
import os
import typing 

import terratorch.models.decoders as decoder_registry
from terratorch.datasets import HLSBands
from terratorch.models.model import (
    ModelFactory,
    Model,
    ModelOutput,
    register_factory,
)
from granitewxc.utils.config import get_config
from granitewxc.utils.downscaling_model import get_finetune_model

logger = logging.getLogger(__name__)

class WxCModuleWrapper(Model, nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        
    def freeze_encoder(self):
        logger.info("freeze encoder")
        for param in self.module.backbone.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        logger.info("freeze decoder")
        for param in self.module.head.parameters():
            param.requires_grad = False

    def forward(self, x) -> ModelOutput:
        mo = self.module.forward(x)
        return ModelOutput(mo)
    
    def load_state_dict(self, state_dict: os.Mapping[str, typing.Any], strict: bool = True, assign: bool = False):

        return self.module.load_state_dict(state_dict, strict, assign)


@register_factory
class WxCModelFactory(ModelFactory):
    def build_model(
        self,
        backbone: str | nn.Module,
        aux_decoders,
        **kwargs,
    ) -> Model:
        module = get_finetune_model(kwargs['model_config'])
        logger.info("Using wrapper.")
        return WxCModuleWrapper(module)

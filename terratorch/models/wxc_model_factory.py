# Copyright contributors to the Terratorch project
import timm
import torch
from torch import nn
import os

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

class WxCModuleWrapper(Model, nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        
    def freeze_encoder(self):
        pass

    def freeze_decoder(self):
        pass

    def forward(self, x) -> ModelOutput:
        mo = self.module.forward(x)
        return ModelOutput(mo)

@register_factory
class WxCModelFactory(ModelFactory):
    def build_model(
        self,
        backbone: str | nn.Module,
        aux_decoders,
        **kwargs,
    ) -> Model:
        #wxc_config_path = os.environ.get('wxc_config_path')
        #if wxc_config_path is None:
        #    raise EnvironmentError("WXC model path not set")
        #config = get_config(wxc_config_path)
        module = get_finetune_model(kwargs['model_config'])
        return WxCModuleWrapper(module)
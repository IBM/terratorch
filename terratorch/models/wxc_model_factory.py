# Copyright contributors to the Terratorch project
import timm
import torch
from torch import nn

import os
import typing 
import logging

import terratorch.models.decoders as decoder_registry
from terratorch.datasets import HLSBands
from terratorch.models.model import (
    Model,
    ModelFactory,
    ModelOutput,
)
from terratorch.registry import MODEL_FACTORY_REGISTRY

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
        
        self.module.load_state_dict(state_dict, strict, assign)


@MODEL_FACTORY_REGISTRY.register
class WxCModelFactory(ModelFactory):
    def build_model(
        self,
        backbone: str | nn.Module,
        aux_decoders,
        checkpoint_path:str=None,
        **kwargs,
    ) -> Model:
        if backbone == 'gravitywave':
            try:
                __import__('prithviwxc.gravitywave.inference')
                from prithviwxc.gravitywave.inference import get_model
                from prithviwxc.gravitywave.config import get_cfg
                cfg = get_cfg()
                model_wrapper = WxCModuleWrapper(get_model(cfg,'uvtp122', cfg.singular_sharded_checkpoint))
                if checkpoint_path:
                    model_wrapper.load_state_dict(torch.load(checkpoint_path, weights_only=True))
                return model_wrapper
            except ImportError as e:
                missing_module = e.name if hasattr(e, 'name') else "unknown module"
                print('prithvi wxc gravitywave not installed, missing module: {missing_module}')
                return None
        else:
            try:
                __import__('granitewxc.utils.config')
                from granitewxc.utils.config import get_config
                from granitewxc.utils.downscaling_model import get_finetune_model
                module = get_finetune_model(kwargs['model_config'])
                model_wrapper = WxCModuleWrapper(module)
                if checkpoint_path:
                    model_wrapper.load_state_dict(torch.load(checkpoint_path, weights_only=True))
                return model_wrapper
            except ImportError:
                print('granite wxc downscaling not installed')

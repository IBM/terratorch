# Copyright contributors to the Terratorch project
import timm
import torch
from torch import nn

import os
import typing 
import logging
import importlib

import terratorch.models.decoders as decoder_registry
from terratorch.datasets import HLSBands
from terratorch.models.model import (
    Model,
    ModelFactory,
    ModelOutput,
)
from terratorch.registry import MODEL_FACTORY_REGISTRY

from terratorch.models.pincers.unet_pincer import UNetPincer

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


@MODEL_FACTORY_REGISTRY.register
class WxCModelFactory(ModelFactory):
    def build_model(
        self,
        backbone: str | nn.Module,
        aux_decoders,
        backbone_weights: str = None,
        **kwargs,
    ) -> Model:
        if backbone == 'prithviwxc':
            try:
                prithviwxc = importlib.import_module('PrithviWxC.model')
            except ModuleNotFoundError as e:
                print(f"Module not found: {e.name}. Please install PrithviWxC using pip install PrithviWxC")
                raise

            backbone = prithviwxc.PrithviWxC(**kwargs)

            # Freeze PrithviWxC model parameters
            for param in backbone.parameters():
                param.requires_grad = False

            # Load pre-trained weights if checkpoint is provided
            if backbone_weights is not None:

                print(f"Starting to load model from {backbone_weights}")
                state_dict = torch.load(
                    f=backbone_weights, weights_only=True
                )

                # Compare the keys in model and saved state_dict
                model_keys = set(backbone.state_dict().keys())
                saved_state_dict_keys = set(state_dict.keys())

                # Find keys that are in the model but not in the saved state_dict
                missing_in_saved = model_keys - saved_state_dict_keys
                # Find keys that are in the saved state_dict but not in the model
                missing_in_model = saved_state_dict_keys - model_keys
                # Find keys that are common between the model and the saved state_dict
                common_keys = model_keys & saved_state_dict_keys

                # Print the common keys
                if common_keys:
                    print(f"Keys loaded : {common_keys}")

                # Print the discrepancies
                if missing_in_saved:
                    print(f"Keys present in model but missing in saved state_dict: {missing_in_saved}")
                if missing_in_model:
                    print(f"Keys present in saved state_dict but missing in model: {missing_in_model}")

                # Load the state_dict with strict=False to allow partial loading
                backbone.load_state_dict(state_dict=state_dict, strict=False)
                print('=>'*10, f"Model loaded from {backbone_weights}...")
                print("Loaded backbone weights")
            else:
                print('Not loading backbone model weigts')

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            backbone.to(device)

            defaults = {
                "encoder_hidden_channels_multiplier" : [1, 2, 4, 8],
                "encoder_num_encoder_blocks" : 4,
                "decoder_hidden_channels_multiplier" : [(16, 8), (12, 4), (6, 2), (3, 1)],
                "decoder_num_decoder_blocks" : 4,
            }
            valid_overrides = {k: v for k, v in kwargs.items() if k in defaults}
            kwargs_updated = defaults.copy()
            kwargs_updated.update(valid_overrides)

            model_to_return = UNetPincer(backbone, **kwargs_updated).to(device)
            return model_to_return
            #return WxCModuleWrapper(backbone)


        # starting from there only for backwards compatibility, deprecated
        if backbone == 'gravitywave':
            try:
                __import__('prithviwxc.gravitywave.inference')
                from prithviwxc.gravitywave.inference import get_model
                from prithviwxc.gravitywave.config import get_cfg
                cfg = get_cfg()
                return WxCModuleWrapper(get_model(cfg,'uvtp122', cfg.singular_sharded_checkpoint))
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
                return WxCModuleWrapper(module)
            except ImportError:
                print('granite wxc downscaling not installed')

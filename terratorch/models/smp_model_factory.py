# Copyright contributors to the Terratorch project

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import encoders as ENCODERS
import importlib
import math

import torch
from torch import nn

from terratorch.datasets import HLSBands
from terratorch.models.backbones.prithvi_vit import checkpoint_filter_fn
from terratorch.models.model import Model, ModelFactory, ModelOutput, register_factory
from terratorch.models.backbones.vit_encoder_decoder import TemporalViTEncoder
from terratorch.models.backbones.swin_encoder_decoder import MMSegSwinTransformer

import torch.nn.functional as F  # noqa: N812

class PrithviModelWrapper(nn.Module):
    def __init__(self, prithvi_class, **kwargs) -> None:
        super().__init__()

        self.config = kwargs
        self.prithvi_class = str(prithvi_class)
        if "MMSegSwinTransformer" in self.prithvi_class:
            self.model = prithvi_class(**kwargs)
            # Default swin preapre_features_for_image_model, can be changed later.
            def prepare_features_for_image_model(x):
                x = list(x)
                outs = [i for i in x if not isinstance(i, tuple)]
                return [
                    layer_output.reshape(
                        -1,
                        int(math.sqrt(layer_output.shape[1])),
                        int(math.sqrt(layer_output.shape[1])),
                        layer_output.shape[2],
                    ).permute(0,3,1,2).contiguous()
                    for layer_output in outs
                ]

            self.model.prepare_features_for_image_model = prepare_features_for_image_model
        elif "TemporalViTEncoder" in self.prithvi_class:
            self.model = prithvi_class(**kwargs, encoder_only=True)
        else:
            self.model = prithvi_class(**kwargs) 
    
    def forward(self, x):
        return self.model.forward_features(x)

    def prepare_features_for_image_model(self, x):
        return self.model.prepare_features_for_image_model(x)
    
    def channels(self):
        return self.config["num_heads"]*[self.config["embed_dim"]]
        

class SMPModelWrapper(Model, nn.Module):
    def __init__(
            self, 
            smp_model, 
            rescale = True, 
            relu=False, 
            squeeze_single_class=False
        ) -> None:
        
        super().__init__()
        self.rescale = rescale
        self.smp_model = smp_model
        self.final_act = nn.ReLU() if relu else nn.Identity()
        self.squeeze_single_class = squeeze_single_class

    def forward(self, x):
        input_size = x.shape[-2:]
        smp_output = self.smp_model(x)
        smp_output = self.final_act(smp_output)

        #TODO: support auxiliary head labels
        if isinstance(smp_output, tuple):
            smp_output, labels = smp_output
        
        if smp_output.shape[1] == 1 and self.squeeze_single_class:
            smp_output = smp_output.squeeze(1)

        if self.rescale and smp_output.shape[-2:] != input_size:
            smp_output = F.interpolate(smp_output, size=input_size, mode="bilinear")
        return ModelOutput(smp_output)

    def freeze_encoder(self):
        freeze_module(self.smp_model.encoder)

    def freeze_decoder(self):
        freeze_module(self.smp_model.decoder)


@register_factory
class SMPModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str,
        decoder: str,
        bands: list[HLSBands | int],
        in_channels: int | None = None,
        num_classes: int = 1,
        pretrained: str | bool | None = True,
        regression_relu: bool = False,
        **kwargs,
    ) -> Model:
        """
        Factory class for creating SMP (Segmentation Models Pytorch) based models with optional customization.

        This factory handles the instantiation of segmentation and regression models using specified
        encoders and decoders from the SMP library, along with custom modifications and extensions such
        as auxiliary decoders or modified encoders.

        Attributes:
            task (str): Specifies the task for which the model is being built. Supported tasks include
                        "segmentation" and "regression".
            backbone (str, nn.Module): Specifies the backbone model to be used. If a string, it should be
                        recognized by the model factory and be able to be parsed appropriately.
            decoder (str): Specifies the decoder to be used for constructing the
                        segmentation model.
            bands (list[terratorch.datasets.HLSBands | int]): A list specifying the bands that the model
                        will operate on. These are expected to be from terratorch.datasets.HLSBands.
            in_channels (int, optional): Specifies the number of input channels. Defaults to None.
            num_classes (int, optional): The number of output classes for the model.
            pretrained (bool | Path, optional): Indicates whether to load pretrained weights for the
                        backbone. Can also specify a path to weights. Defaults to True.
            num_frames (int, optional): Specifies the number of timesteps the model should handle. Useful
                        for temporal models.
            regression_relu (bool): Whether to apply ReLU activation in the case of regression tasks.
            **kwargs: Additional arguments that might be passed to further customize the backbone, decoder,
                        or any auxiliary heads. These should be prefixed appropriately

        Raises:
            ValueError: If the specified decoder is not supported by SMP.
            Exception: If the specified task is not "segmentation" or "regression".

        Returns:
            nn.Module: A model instance wrapped in SMPModelWrapper configured according to the specified
                    parameters and tasks.
        """
        self.CPU_ONLY = not torch.cuda.is_available()
        bands = [HLSBands.try_convert_to_hls_bands_enum(b) for b in bands]
        if in_channels is None:
            in_channels = len(bands)

        if task.lower() not in ["segmentation", "regression"]:
            msg = f"SMP models can only perform pixel wise tasks, but got task {task}"
            raise Exception(msg)
        
        backbone_kwargs = _extract_prefix_keys(kwargs, "backbone_") # Encoder params should be prefixed backbone_
        smp_kwargs = _extract_prefix_keys(backbone_kwargs, "smp_") # Smp model params should be prefixed smp_
        aux_params = _extract_prefix_keys(backbone_kwargs, "aux_") # Auxiliary head params should be prefixed aux_

        # Gets decoder module.
        decoder_module = getattr(smp, decoder, None)
        if decoder_module is None:
            raise ValueError(f"Decoder {decoder} is not supported in SMP.")
        
        if isinstance(pretrained, bool):
            if pretrained:
                pretrained = "imagenet"
            else:
                pretrained = None  
        
        model_dict = None
        # If encoder not currently supported by SMP (either Prithvi or custom encoder).
        if backbone not in ENCODERS:
            # These params must be included in the config file with appropriate prefix.
            required_params = {
                'encoder_depth': smp_kwargs,
                'out_channels': backbone_kwargs,
                'output_stride': backbone_kwargs
            }
            
            for param, config_dict in required_params.items():
                if param not in config_dict:
                    raise ValueError(f"Config must include the '{param}' parameter")
            # Can load model from local checkpoint. 
            if "checkpoint_path" in backbone_kwargs:
                
                checkpoint_path = backbone_kwargs.pop("checkpoint_path")
                print(f"Trying to load from the path defined in the config file, {checkpoint_path}.")

                if self.CPU_ONLY:
                    model_dict = torch.load(checkpoint_path, map_location="cpu")
                else:
                    model_dict = torch.load(checkpoint_path)        

            if backbone.startswith("prithvi"): # Using Prithvi encoder (ViT or Swin).
                backbone_class = self._make_smp_encoder(PrithviModelWrapper)
                if  backbone.startswith("prithvi_swin"):
                    backbone_kwargs['prithvi_class'] = MMSegSwinTransformer
                elif backbone.startswith("prithvi_vit"):
                    backbone_kwargs['prithvi_class'] = TemporalViTEncoder
                else:
                    msg = f"Prithvi Backbone not found."
                    raise NotImplementedError(msg)
            # Using new encoder (not Prithvi or SMP).
            else: 
                backbone_class = self._make_smp_encoder(backbone)

            # Registering custom encoder into SMP.
            self._register_custom_encoder(backbone_class, backbone_kwargs, pretrained)

            model_args = {
                "encoder_name": "SMPEncoderWrapperWithPFFIM",
                "encoder_weights": pretrained,  
                "in_channels": in_channels,   
                "classes": num_classes,
                **smp_kwargs
            }
        # Using SMP encoder.
        else: 
            model_args = {
                "encoder_name": backbone,
                "encoder_weights": pretrained,  
                "in_channels": in_channels,   
                "classes": num_classes,
                **smp_kwargs
            }
            
        if aux_params:  
            model = decoder_module(**model_args, aux_params=aux_params)
        else:
            model = decoder_module(**model_args)

        # Loads state dict from checkpoint.
        if model_dict:
            if hasattr(model, "prithvi_class") and "TemporalViTEncoder" in model.prithvi_class:
                model_dict = checkpoint_filter_fn(model_dict, model=model.encoder, pretrained_bands=bands, model_bands=bands)
            model.encoder.load_state_dict(model_dict, strict=False)


        return SMPModelWrapper(
            model, 
            relu=task == "regression" and regression_relu, 
            squeeze_single_class=task == "regression"
        )

    # Registers a custom encoder into SMP.
    def _register_custom_encoder(self, Encoder, params, pretrained):

        ENCODERS["SMPEncoderWrapperWithPFFIM"] = {
            'encoder': Encoder,
            'params': params,
            'pretrained_settings': pretrained
        }

    # Gets class either from string or from Module reference.
    def _make_smp_encoder(self, Encoder):
        if isinstance(Encoder, str):
            BaseClass = _get_class_from_string(Encoder)
        else:
            BaseClass = Encoder
        
        # Creates Wrapper for SMP Encoder with PFFIM.
        # Wrapper needed to include SMP params and PFFIM
        class SMPEncoderWrapperWithPFFIM(BaseClass, nn.Module):
            def __init__(
                    self, 
                    depth,
                    output_stride,
                    out_channels,
                    *args,
                    **kwargs
                ) -> None:
                super().__init__(*args, **kwargs)
                self._depth = depth
                self._output_stride = output_stride
                self._out_channels = out_channels
                
                # If PFFIM is passed from config file
                if "prepare_features_for_image_model" in kwargs:
                    path = kwargs['prepare_features_for_image_model'] 
                    self.prepare_features_for_image_model = _get_callable_from_path(path)  
                # If PFFIM is in super                  
                elif hasattr(super(), 'prepare_features_for_image_model'):
                    self.prepare_features_for_image_model = super().prepare_features_for_image_model
                # No PFFIM
                else:
                    self.prepare_features_for_image_model = lambda x: x

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                features = super().forward(x)
                features = self.prepare_features_for_image_model(features)

                return features

            @property
            def out_channels(self):
                if hasattr(super(), 'out_channels'):
                    return super().out_channels()
                
                return self._out_channels

            @property
            def output_stride(self):
                if hasattr(super(), 'output_stride'):
                    return super().output_stride()
                
                return min(self._output_stride, 2**self._depth)

            def set_in_channels(self, in_channels, pretrained):
                if hasattr(super(), 'set_in_channels'):
                    return super().set_in_channels(in_channels, pretrained)
                else:
                    pass
                
            def make_dilated(self, output_stride):
                if hasattr(super(), 'make_dilated'):
                    return super().make_dilated(output_stride)
                else:
                    pass
    
        return SMPEncoderWrapperWithPFFIM


def _extract_prefix_keys(d: dict, prefix: str) -> dict:
    extracted_dict = {}
    keys_to_del = []
    for k, v in d.items():
        if k.startswith(prefix):
            extracted_dict[k.split(prefix)[1]] = v
            keys_to_del.append(k)

    for k in keys_to_del:
        del d[k]

    return extracted_dict


def _get_class_from_string(class_path):
    module_path, name = class_path.rsplit('.', 1)  
    module = importlib.import_module(module_path)
    return getattr(module, name)

def _get_callable_from_path(func_path):
    parts = func_path.split('.')
    method_name = parts[-1]
    module_path = parts[:-1]

    if not module_path:  
        raise ValueError(f"No callable named '{method_name}' found scope.")

    try:
        module_part = '.'.join(module_path)
        module = importlib.import_module(module_part)
        result = module
        if hasattr(result, method_name) and callable(getattr(result, method_name)):
            return getattr(result, method_name)
        else:
            for part in module_path[1:]:
                result = getattr(result, part)
                if hasattr(result, method_name) and callable(getattr(result, method_name)):
                    return getattr(result, method_name)
            raise AttributeError(f"Failed to resolve '{func_path}' to a valid callable.")
    except ImportError as ie:
        raise ImportError(f"Failed to import '{module_part}': {ie}")
    except AttributeError as ae:
        raise AttributeError(f"Attribute error in resolving the callable: {ae}")

def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad_(False)


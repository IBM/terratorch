# Copyright contributors to the Terratorch project

"""
This is just an example of a possible structure to include SMP models
Right now it always returns a UNET, but could easily be extended to many of the models provided by SMP.
"""

from torch import nn
import torch
from terratorch.models.model import Model, ModelFactory, ModelOutput, register_factory
from terratorch.tasks.segmentation_tasks import to_segmentation_prediction

import importlib

def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad_(False)

@register_factory
class GenericUnetModelFactory(ModelFactory):
    def build_model(
        self,
        task: str = "segmentation",
        backbone: str = None,
        decoder: str = None,
        dilations: tuple[int] = (1, 6, 12, 18),
        in_channels: int = 6,
        pretrained: str | bool | None = True,
        num_classes: int = 1,
        regression_relu: bool = False,
        **kwargs,
    ) -> Model:
        """Factory to create model based on SMP.

        Args:
            task (str): Must be "segmentation".
            model (str): Decoder architecture. Currently only supports "unet".
            in_channels (int): Number of input channels.
            pretrained(str | bool): Which weights to use for the backbone. If true, will use "imagenet". If false or None, random weights. Defaults to True.
            num_classes (int): Number of classes.
            regression_relu (bool). Whether to apply a ReLU if task is regression. Defaults to False.

        Returns:
            Model: SMP model wrapped in SMPModelWrapper.
        """
        if task not in ["segmentation", "regression"]:
            msg = f"SMP models can only perform pixel wise tasks, but got task {task}"
            raise Exception(msg)

        builtin_engine_decoders = importlib.import_module("terratorch.models.decoders")
        builtin_engine_encoders = importlib.import_module("terratorch.models.backbones")

        try:
            engine_decoders = importlib.import_module("mmseg.models.decode_heads")
            engine_encoders = importlib.import_module("mmseg.models.backbones")
            _has_mmseg = True
        except:
            engine_decoders = None
            engine_encoders = None
            _has_mmseg = False
            print("mmseg is not installed.")

        if backbone:
            backbone_kwargs = _extract_prefix_keys(kwargs, "backbone_")
            model = backbone
            model_kwargs = backbone_kwargs
            engine = engine_encoders
            builtin_engine = builtin_engine_encoders
        elif decoder: 
            decoder_kwargs = _extract_prefix_keys(kwargs, "decoder_")
            model = decoder
            model_kwargs = decoder_kwargs
            engine = engine_decoders
            builtin_engine = builtin_engine_decoders
        else:
            print("It is necessary to define a backbone and/or a decoder.")

        try:
            print(f"Using module {model} from terratorch.")
            model_class = getattr(builtin_engine, model)
        except:
            if _has_mmseg:
                print("Module not available on terratorch.")
                print(f"Using module {model} from mmseg.")
                model_class = getattr(engine, model)
            else:
                raise Exception("mmseg is not installed.")

        model = model_class(
           **model_kwargs,
        )
       
        return GenericUnetModelWrapper(
            model, relu=task == "regression" and regression_relu, squeeze_single_class=task == "regression"
        )

class GenericUnetModelWrapper(Model, nn.Module):
    def __init__(self, unet_model, relu=False, squeeze_single_class=False) -> None:
        super().__init__()
        self.unet_model = unet_model
        self.final_act = nn.ReLU() if relu else nn.Identity()
        self.squeeze_single_class = squeeze_single_class

    def forward(self, *args, **kwargs):

        # It supposes the input has dimension (B, C, H, W)
        input_data = [args[0]] # It adapts the input to became a list of time 'snapshots'
        args = (input_data,)

        unet_output = self.unet_model(*args, **kwargs)
        unet_output = self.final_act(unet_output)

        if unet_output.shape[1] == 1 and self.squeeze_single_class:
            unet_output = unet_output.squeeze(1)

        model_output = ModelOutput(unet_output)

        return model_output

    def freeze_encoder(self):
        raise NotImplementedError()

    def freeze_decoder(self):
        raise freeze_module(self.unet_model)


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

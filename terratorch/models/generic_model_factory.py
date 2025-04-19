# Copyright contributors to the Terratorch project

"""
This is just an example of a possible structure to include SMP models
Right now it always returns a UNET, but could easily be extended to many of the models provided by SMP.
"""

import importlib

import torch
from torch import nn

from terratorch.models.model import Model, ModelFactory, ModelOutput
from terratorch.models.utils import extract_prefix_keys
from terratorch.registry import MODEL_FACTORY_REGISTRY
from terratorch.tasks.segmentation_tasks import to_segmentation_prediction
from terratorch.registry import BACKBONE_REGISTRY

def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad_(False)

@MODEL_FACTORY_REGISTRY.register
class GenerictModelFactory(ModelFactory):

    def build_model(
        self,
        task: str = "segmentation",
        backbone: str | None = None,
        decoder: str | None = None,
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

        model_kwargs = _extract_prefix_keys(kwargs, "backbone_")

        try:
            model_class = BACKBONE_REGISTRY.get("model")
            model = model_class(
               **model_kwargs,
            )
        except ValueError:
            raise Exception("Model {model} not found in the registry.")

        return GenericModelWrapper(
            backbone, decoder=decoder, relu=task == "regression" and regression_relu, squeeze_single_class=task == "regression"
        )

class GenericModelWrapper(Model, nn.Module):
    def __init__(self, model, decoder=None, relu=False, squeeze_single_class=False) -> None:
        super().__init__()
        self.model = model
        self.squeeze_single_class = squeeze_single_class

    def forward(self, *args, **kwargs):

        # It supposes the input has dimension (B, C, H, W)
        input_data = [args[0]] # It adapts the input to became a list of time 'snapshots'
        args = (input_data,)

        model_output = self.model(*args, **kwargs)

        model_output = ModelOutput(model_output)

        return model_output

    def freeze_model(self):
        raise freeze_module(self.model)

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


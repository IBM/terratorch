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
class GenericModelFactory(ModelFactory):

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
        """Factory to create models from any custom module.

        Args:
            task (str): The task we are using.
            model (str): The name for the model class.
            in_channels (int): Number of input channels.
            pretrained(str | bool): Which weights to use for the backbone. If true, will use "imagenet". If false or None, random weights. Defaults to True.
            num_classes (int): Number of classes.
            regression_relu (bool). Whether to apply a ReLU if task is regression. Defaults to False.

        Returns:
            Model: A wrapped generic model.
        """

        model_kwargs = _extract_prefix_keys(kwargs, "backbone_")

        try:
            model = BACKBONE_REGISTRY.build(backbone, **model_kwargs)

        except KeyError:
            raise KeyError(f"Model {backbone} not found in the registry.")

        return GenericModelWrapper(model)

class GenericModelWrapper(Model, nn.Module):
    """
    A wrapper to adapt a generic model to be used with TerraTorch

    Args: 
        model (torch.nn.Module): The model we want to wrap. 

    Returns:
        The wrapped model.
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def freeze_encoder(self):
        freeze_module(self.model)

    def freeze_decoder(self):
        freeze_module(self.model)

    def forward(self, *args, **kwargs):
        """
        The forward method is prepared to receive any argument or keyword the wrapped model
        need to run. 
        """
        # It supposes the input has dimension (B, C, H, W)
        input_data = [args[0]] # It adapts the input to became a list of time 'snapshots'

        model_output = self.model(*args, **kwargs)

        model_output = ModelOutput(model_output)

        return model_output

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


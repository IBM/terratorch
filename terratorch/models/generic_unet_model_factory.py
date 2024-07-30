# Copyright contributors to the Terratorch project

"""
This is just an example of a possible structure to include SMP models
Right now it always returns a UNET, but could easily be extended to many of the models provided by SMP.
"""

from torch import nn

from terratorch.models.model import Model, ModelFactory, ModelOutput, register_factory

from mmseg.models.decode_heads import ASPPHead 

import importlib

@register_factory
class GenericUnetModelFactory(ModelFactory):
    def build_model(
        self,
        task: str = "segmentation",
        model: str = "ASPPHead",
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

        model_class = getattr(mmseg.models.decode_heads, model)

        model = model_class(
           dilations=dilations 
        )
       
        return SMPModelWrapper(
            model, relu=task == "regression" and regression_relu, squeeze_single_class=task == "regression"
        )


class GenericUnetModelWrapper(Model, nn.Module):
    def __init__(self, smp_model, relu=False, squeeze_single_class=False) -> None:
        super().__init__()
        self.smp_model = smp_model
        self.final_act = nn.ReLU() if relu else nn.Identity()
        self.squeeze_single_class = squeeze_single_class

    def forward(self, *args, **kwargs):
        smp_output = self.smp_model(*args, **kwargs)
        smp_output = self.final_act(smp_output)
        if smp_output.shape[1] == 1 and self.squeeze_single_class:
            smp_output = smp_output.squeeze(1)
        return ModelOutput(smp_output)

    def freeze_encoder(self):
        raise NotImplementedError()

    def freeze_decoder(self):
        raise NotImplementedError()


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
# Copyright contributors to the Terratorch project

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from terratorch.models.model import AuxiliaryHeadWithDecoderWithoutInstantiatedHead, Model, ModelOutput

def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad_(False)

class PreTrainModel(Model):
    """Model that encapsulates encoder and decoder and heads
    Expects decoder to have a "forward_features" method, an embed_dims property
    and optionally a "prepare_features_for_image_model" method.
    """

    def __init__(
        self,
        model: nn.Module,
        rescale: bool = True,  # noqa: FBT002, FBT001
    ) -> None:
        """Constructor

        Args:
            task (str): Task to be performed. One of segmentation or regression.
            model (nn.Module): Complete Encoder-Decoder module.
            rescale (bool, optional): Rescale the output of the model if it has a different size than the ground truth.
                Uses bilinear interpolation. Defaults to True.
        """
        super().__init__()

        self.encoder_decoder = model

        self.rescale = rescale

    def freeze(self):
        freeze_module(self.encoder_decoder)

    # TODO: do this properly
    def check_input_shape(self, x: torch.Tensor) -> bool:  # noqa: ARG002
        return True

    @staticmethod
    def _check_for_single_channel_and_squeeze(x):
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> ModelOutput:
        """Sequentially pass `x` through model`s encoder, decoder and heads"""
        self.check_input_shape(x)
        if isinstance(x, torch.Tensor):
            input_size = x.shape[-2:]
        elif hasattr(kwargs, 'image_size'):
            input_size = kwargs['image_size']
        elif isinstance(x, dict):
            # Multimodal input in passed as dict
            input_size = list(x.values())[0].shape[-2:]
        else:
            ValueError('Could not infer input shape.')
        reconstruction = self.encoder_decoder(x, **kwargs)

        if self.rescale and mask.shape[-2:] != input_size:
            target = F.interpolate(target, size=input_size, mode="bilinear")
        target = self._check_for_single_channel_and_squeeze(mask)

        return ModelOutput(output=reconstruction)


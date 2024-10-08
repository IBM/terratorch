# Copyright contributors to the Terratorch project

import torch
import torch.nn.functional as F  # noqa: N812
from segmentation_models_pytorch.base import SegmentationModel
from torch import nn

from terratorch.models.heads import RegressionHead, SegmentationHead
from terratorch.models.model import AuxiliaryHeadWithDecoderWithoutInstantiatedHead, Model, ModelOutput


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad_(False)


class PixelWiseModel(Model, SegmentationModel):
    """Model that encapsulates encoder and decoder and heads
    Expects decoder to have a "forward_features" method, an embed_dims property
    and optionally a "prepare_features_for_image_model" method.
    """

    def __init__(
        self,
        task: str,
        encoder: nn.Module,
        decoder: nn.Module,
        head_kwargs: dict,
        auxiliary_heads: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] | None = None,
        neck: nn.Module | None = None,
        rescale: bool = True,  # noqa: FBT002, FBT001
    ) -> None:
        """Constructor

        Args:
            task (str): Task to be performed. One of segmentation or regression.
            encoder (nn.Module): Encoder to be used
            decoder (nn.Module): Decoder to be used
            head_kwargs (dict): Arguments to be passed at instantiation of the head.
            auxiliary_heads (list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] | None, optional): List of
                AuxiliaryHeads with heads to be instantiated. Defaults to None.
            neck (nn.Module | None): Module applied between backbone and decoder.
                Defaults to None, which applies the identity.
            rescale (bool, optional): Rescale the output of the model if it has a different size than the ground truth.
                Uses bilinear interpolation. Defaults to True.
        """
        super().__init__()

        self.task = task
        self.encoder = encoder
        self.decoder = decoder
        final_head_included = getattr(self.decoder, "includes_head", False)  # some models already include a head
        self.head = (
            self._get_head(task, decoder.output_embed_dim, head_kwargs) if not final_head_included else nn.Identity()
        )

        if auxiliary_heads is not None:
            aux_heads = {}
            for aux_head_to_be_instantiated in auxiliary_heads:
                aux_head: nn.Module = self._get_head(
                    task, aux_head_to_be_instantiated.decoder.output_embed_dim, head_kwargs
                )
                aux_head = nn.Sequential(aux_head_to_be_instantiated.decoder, aux_head)
                aux_heads[aux_head_to_be_instantiated.name] = aux_head
        else:
            aux_heads = {}
        self.aux_heads = nn.ModuleDict(aux_heads)

        self.neck = neck
        self.rescale = rescale

    def freeze_encoder(self):
        freeze_module(self.encoder)

    def freeze_decoder(self):
        freeze_module(self.encoder)
        freeze_module(self.head)

    # TODO: do this properly
    def check_input_shape(self, x: torch.Tensor) -> bool:  # noqa: ARG002
        return True

    @staticmethod
    def _check_for_single_channel_and_squeeze(x):
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Sequentially pass `x` through model`s encoder, decoder and heads"""
        self.check_input_shape(x)
        input_size = x.shape[-2:]
        features = self.encoder(x)

        ## only for backwards compatibility with pre-neck times.
        if self.neck:
            prepare = self.neck
        else:
            # for backwards compatibility, if this is defined in the encoder, use it
            prepare = getattr(self.encoder, "prepare_features_for_image_model", lambda x: x)

        features = prepare(features)
        decoder_output = self.decoder([f.clone() for f in features])
        mask = self.head(decoder_output)
        if self.rescale and mask.shape[-2:] != input_size:
            mask = F.interpolate(mask, size=input_size, mode="bilinear")
        mask = self._check_for_single_channel_and_squeeze(mask)
        aux_outputs = {}
        for name, decoder in self.aux_heads.items():
            aux_output = decoder([f.clone() for f in features])
            if self.rescale and aux_output.shape[-2:] != input_size:
                aux_output = F.interpolate(aux_output, size=input_size, mode="bilinear")
            aux_output = self._check_for_single_channel_and_squeeze(aux_output)
            aux_outputs[name] = aux_output
        return ModelOutput(output=mask, auxiliary_heads=aux_outputs)

    def _get_head(self, task: str, input_embed_dim: int, head_kwargs):
        if task == "segmentation":
            if "num_classes" not in head_kwargs:
                msg = "num_classes must be defined for segmentation task"
                raise Exception(msg)
            return SegmentationHead(input_embed_dim, **head_kwargs)
        if task == "regression":
            return RegressionHead(input_embed_dim, **head_kwargs)
        msg = "Task must be one of segmentation or regression."
        raise Exception(msg)

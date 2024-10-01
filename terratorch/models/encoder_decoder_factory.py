# Copyright contributors to the Terratorch project

from collections.abc import Callable

import segmentation_models_pytorch as smp
import timm
import torch
from torch import nn

from terratorch.datasets import HLSBands
from terratorch.models.model import (
    AuxiliaryHead,
    AuxiliaryHeadWithDecoderWithoutInstantiatedHead,
    Model,
    ModelFactory,
    register_factory,
)
from terratorch.models.pixel_wise_model import PixelWiseModel
from terratorch.models.registry import BACKBONE_REGISTRY, DECODER_REGISTRY
from terratorch.models.scalar_output_model import ScalarOutputModel
from terratorch.models.utils import extract_prefix_keys

PIXEL_WISE_TASKS = ["segmentation", "regression"]
SCALAR_TASKS = ["classification"]
SUPPORTED_TASKS = PIXEL_WISE_TASKS + SCALAR_TASKS

def _get_backbone(backbone: str | nn.Module, **backbone_kwargs) -> nn.Module:
    if not isinstance(backbone, nn.Module):
        return backbone
    return BACKBONE_REGISTRY.build(backbone, **backbone_kwargs)

def _get_decoder(decoder: str | nn.Module, channels: int, **decoder_kwargs) -> nn.Module:
    if isinstance(decoder, nn.Module):
        return decoder
    return DECODER_REGISTRY.build(decoder, channels, **decoder_kwargs)

@register_factory
class EncoderDecoderFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module,
        num_classes: int | None = None,
        post_backbone_ops: list[Callable] | None = None,
        aux_decoders: list[AuxiliaryHead] | None = None,
        rescale: bool = True,  # noqa: FBT002, FBT001
        **kwargs,
    ) -> Model:
        """Generic model factory that combines an encoder and decoder, together with a head, for a specific task.

        Further arguments to be passed to the backbone, decoder or head. They should be prefixed with
        `backbone_`, `decoder_` and `head_` respectively.

        Args:
            task (str): Task to be performed. Currently supports "segmentation" and "regression".
            backbone (str, nn.Module): Backbone to be used. If a string, will look for such models in the different
                registries supported (internal terratorch registry, timm, ...). If a torch nn.Module, will use it
                directly. In this case, it should have a timm FeautureInfo under an attribute named `feature_info`.
            decoder (Union[str, nn.Module], optional): Decoder to be used for the segmentation model.
                    If a string, will look for such decoders in the different
                    registries supported (internal terratorch registry, smp, ...).
                    If an nn.Module, we expect it to expose a property `decoder.output_embed_dim`.
                    Pixel wise tasks will be concatenated with a Conv2d for the final convolution.
                    Defaults to "FCNDecoder".
            num_classes (int, optional): Number of classes. None for regression tasks.
            pretrained (Union[bool, Path], optional): Whether to load pretrained weights for the backbone, if available.
                Defaults to True.
            num_frames (int, optional): Number of timesteps for the model to handle. Defaults to 1.
            post_backbone_ops (list[Callable]): Functions to be called in succession on encoder features
                before passing them to the decoder. Defaults to None, which applies the identity function.
            aux_decoders (list[AuxiliaryHead] | None): List of AuxiliaryHead decoders to be added to the model.
                These decoders take the input from the encoder as well.
            rescale (bool): Whether to apply bilinear interpolation to rescale the model output if its size
                is different from the ground truth. Only applicable to pixel wise models
                (e.g. segmentation, pixel wise regression). Defaults to True.


        Returns:
            nn.Module: Full model with encoder, decoder and head.
        """
        task = task.lower()
        if task not in SUPPORTED_TASKS:
            msg = f"Task {task} not supported. Please choose one of {SUPPORTED_TASKS}"
            raise NotImplementedError(msg)

        backbone_kwargs, kwargs = extract_prefix_keys(kwargs, "backbone_")
        backbone = _get_backbone(backbone, **backbone_kwargs)

        decoder_kwargs, kwargs = extract_prefix_keys(kwargs, "decoder_")
        decoder = _get_decoder(decoder, backbone.feature_info.channels(), **decoder_kwargs)

        head_kwargs, kwargs = extract_prefix_keys(kwargs, "head_")
        if num_classes:
            head_kwargs["num_classes"] = num_classes
        if aux_decoders is None:
            return _build_appropriate_model(
                task, backbone, decoder, head_kwargs, post_backbone_ops, rescale=rescale
            )

        to_be_aux_decoders: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] = []
        for aux_decoder in aux_decoders:
            args = aux_decoder.decoder_args if aux_decoder.decoder_args else {}
            aux_decoder_cls: nn.Module = _get_decoder(aux_decoder.decoder)
            aux_decoder_kwargs, kwargs = extract_prefix_keys(args, "decoder_")
            aux_decoder_instance = aux_decoder_cls(backbone.feature_info.channels(), **aux_decoder_kwargs)

            aux_head_kwargs, kwargs = extract_prefix_keys(args, "head_")
            if num_classes:
                aux_head_kwargs["num_classes"] = num_classes
            to_be_aux_decoders.append(
                AuxiliaryHeadWithDecoderWithoutInstantiatedHead(aux_decoder.name, aux_decoder_instance, aux_head_kwargs)
            )

        return _build_appropriate_model(
            task,
            backbone,
            decoder,
            head_kwargs,
            post_backbone_ops,
            rescale=rescale,
            auxiliary_heads=to_be_aux_decoders,
        )


def _build_appropriate_model(
    task: str,
    backbone: nn.Module,
    decoder: nn.Module,
    head_kwargs: dict,
    post_backbone_ops: Callable,
    rescale: bool = True,  # noqa: FBT001, FBT002
    auxiliary_heads: dict | None = None,
):
    if task in PIXEL_WISE_TASKS:
        return PixelWiseModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            post_backbone_ops=post_backbone_ops,
            rescale=rescale,
            auxiliary_heads=auxiliary_heads,
        )
    elif task in SCALAR_TASKS:
        return ScalarOutputModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            post_backbone_ops=post_backbone_ops,
            auxiliary_heads=auxiliary_heads,
        )

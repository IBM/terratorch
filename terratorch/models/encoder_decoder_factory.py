# Copyright contributors to the Terratorch project


import torch
from torch import nn

from terratorch.models.model import (
    AuxiliaryHead,
    AuxiliaryHeadWithDecoderWithoutInstantiatedHead,
    Model,
    ModelFactory,
)
from terratorch.models.necks import Neck, build_neck_list
from terratorch.models.pixel_wise_model import PixelWiseModel
from terratorch.models.scalar_output_model import ScalarOutputModel
from terratorch.models.utils import extract_prefix_keys
from terratorch.registry import BACKBONE_REGISTRY, DECODER_REGISTRY, MODEL_FACTORY_REGISTRY

PIXEL_WISE_TASKS = ["segmentation", "regression"]
SCALAR_TASKS = ["classification"]
SUPPORTED_TASKS = PIXEL_WISE_TASKS + SCALAR_TASKS



class TemporalWrapper(nn.Module):
    def __init__(self, encoder, pooling="mean"):
        super().__init__()
        self.encoder = encoder
        if pooling == "mean":
            self.pooling = torch.mean
        elif pooling == "max":
            self.pooling = torch.max
        else:
            msg = "Pooling must be 'mean' or 'max'"
            raise ValueError(msg)

    def forward(self, x):
        # x is a list of tensors, each corresponding to a different timestamp
        features = [self.encoder(t) for t in x]
        # Stack features along a new dimension and apply pooling
        features = torch.stack(features, dim=0)
        if self.pooling == torch.max:
            pooled_features, _ = self.pooling(features, dim=0)
        else:
            pooled_features = self.pooling(features, dim=0)
        return pooled_features


def _get_backbone(backbone: str | nn.Module, **backbone_kwargs) -> nn.Module:
    if isinstance(backbone, nn.Module):
        return backbone
    return BACKBONE_REGISTRY.build(backbone, **backbone_kwargs)


def _get_decoder_and_head_kwargs(
    decoder: str | nn.Module,
    channel_list: list[int],
    decoder_kwargs: dict,
    head_kwargs: dict,
    num_classes: int | None = None,
) -> tuple[nn.Module, dict, bool]:
    # if its already an nn Module, check if it includes a head. if it doesnt, pass num classes to head kwargs
    if isinstance(decoder, nn.Module):
        if not getattr(decoder, "includes_head", False) and num_classes is not None:
            head_kwargs["num_classes"] = num_classes
        elif head_kwargs:
            msg = "Decoder already includes a head, but `head_` arguments were specified. These should be removed."
            raise ValueError(msg)
        return decoder, head_kwargs, False

    # if its not an nn module, check if the class includes a head
    # depending on that, pass num classes to either head kwrags or decoder
    decoder_includes_head = DECODER_REGISTRY.find_registry(decoder).includes_head
    if num_classes is not None:
        if decoder_includes_head:
            decoder_kwargs["num_classes"] = num_classes
            if head_kwargs:
                msg = "Decoder already includes a head, but `head_` arguments were specified. These should be removed."
                raise ValueError(msg)
        else:
            head_kwargs["num_classes"] = num_classes

    return DECODER_REGISTRY.build(decoder, channel_list, **decoder_kwargs), head_kwargs, decoder_includes_head


def _check_all_args_used(kwargs):
    if kwargs:
        msg = f"arguments {kwargs} were passed but not used."
        raise ValueError(msg)


@MODEL_FACTORY_REGISTRY.register
class EncoderDecoderFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module,
        num_classes: int | None = None,
        necks: list[dict] | None = None,
        aux_decoders: list[AuxiliaryHead] | None = None,
        rescale: bool = True,  # noqa: FBT002, FBT001,
        use_temporal: bool = False,
        temporal_pooling: str = "mean",
        **kwargs,
    ) -> Model:
        """Generic model factory that combines an encoder and decoder, together with a head, for a specific task.

        Further arguments to be passed to the backbone, decoder or head. They should be prefixed with
        `backbone_`, `decoder_` and `head_` respectively.

        Args:
            task (str): Task to be performed. Currently supports "segmentation" and "regression".
            backbone (str, nn.Module): Backbone to be used. If a string, will look for such models in the different
                registries supported (internal terratorch registry, timm, ...). If a torch nn.Module, will use it
                directly. The backbone should have and `out_channels` attribute and its `forward` should return a list[Tensor].
            decoder (Union[str, nn.Module], optional): Decoder to be used for the segmentation model.
                    If a string, will look for such decoders in the different
                    registries supported (internal terratorch registry, smp, ...).
                    If an nn.Module, we expect it to expose a property `decoder.out_channels`.
                    Pixel wise tasks will be concatenated with a Conv2d for the final convolution.
                    Defaults to "FCNDecoder".
            num_classes (int, optional): Number of classes. None for regression tasks.
            necks (list[dict]): nn.Modules to be called in succession on encoder features
                before passing them to the decoder. Should be registered in the NECKS_REGISTRY registry.
                Expects each one to have a key "name" and subsequent keys for arguments, if any.
                Defaults to None, which applies the identity function.
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

        try:
            out_channels = backbone.out_channels
        except AttributeError as e:
            msg = "backbone must have out_channels attribute"
            raise AttributeError(msg) from e

        if necks is None:
            necks = []
        neck_list, channel_list = build_neck_list(necks, out_channels)

        # some decoders already include a head
        # for these, we pass the num_classes to them
        # others dont include a head
        # for those, we dont pass num_classes
        decoder_kwargs, kwargs = extract_prefix_keys(kwargs, "decoder_")
        head_kwargs, kwargs = extract_prefix_keys(kwargs, "head_")

        decoder, head_kwargs, decoder_includes_head = _get_decoder_and_head_kwargs(
            decoder, channel_list, decoder_kwargs, head_kwargs, num_classes=num_classes
        )

        # Add temporal wrapper if enabled
        if use_temporal:
            backbone = TemporalWrapper(backbone, pooling=temporal_pooling)

        if aux_decoders is None:
            _check_all_args_used(kwargs)
            return _build_appropriate_model(task, backbone, decoder, head_kwargs, necks=neck_list, decoder_includes_head=decoder_includes_head, rescale=rescale)

        to_be_aux_decoders: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] = []
        for aux_decoder in aux_decoders:
            args = aux_decoder.decoder_args if aux_decoder.decoder_args else {}
            aux_decoder_kwargs, args = extract_prefix_keys(args, "decoder_")
            aux_head_kwargs, args = extract_prefix_keys(args, "head_")
            aux_decoder_instance, aux_head_kwargs, aux_decoder_includes_head = _get_decoder_and_head_kwargs(
                aux_decoder.decoder, channel_list, aux_decoder_kwargs, aux_head_kwargs, num_classes=num_classes
            )
            to_be_aux_decoders.append(
                AuxiliaryHeadWithDecoderWithoutInstantiatedHead(aux_decoder.name, aux_decoder_instance, aux_head_kwargs)
            )
            _check_all_args_used(args)

        _check_all_args_used(kwargs)

        return _build_appropriate_model(
            task,
            backbone,
            decoder,
            head_kwargs,
            necks=neck_list,
            decoder_includes_head=decoder_includes_head,
            rescale=rescale,
            auxiliary_heads=to_be_aux_decoders,
        )


def _build_appropriate_model(
    task: str,
    backbone: nn.Module,
    decoder: nn.Module,
    head_kwargs: dict,
    decoder_includes_head: bool = False,
    necks: list[Neck] | None = None,
    rescale: bool = True,  # noqa: FBT001, FBT002
    auxiliary_heads: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] | None = None,
):
    if necks:
        neck_module: nn.Module = nn.Sequential(*necks)
    else:
        neck_module = None
    if task in PIXEL_WISE_TASKS:
        return PixelWiseModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            decoder_includes_head=decoder_includes_head,
            neck=neck_module,
            rescale=rescale,
            auxiliary_heads=auxiliary_heads,
        )
    elif task in SCALAR_TASKS:
        return ScalarOutputModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            decoder_includes_head=decoder_includes_head,
            neck=neck_module,
            auxiliary_heads=auxiliary_heads,
        )

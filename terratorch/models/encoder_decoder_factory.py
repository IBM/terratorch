# Copyright contributors to the Terratorch project


from torch import nn

from terratorch.models.model import (
    AuxiliaryHead,
    AuxiliaryHeadWithDecoderWithoutInstantiatedHead,
    Model,
    ModelFactory,
)
from terratorch.models.necks import build_neck_list
from terratorch.models.pixel_wise_model import PixelWiseModel
from terratorch.models.scalar_output_model import ScalarOutputModel
from terratorch.models.utils import extract_prefix_keys
from terratorch.registry import BACKBONE_REGISTRY, DECODER_REGISTRY, MODEL_FACTORY_REGISTRY

PIXEL_WISE_TASKS = ["segmentation", "regression"]
SCALAR_TASKS = ["classification"]
SUPPORTED_TASKS = PIXEL_WISE_TASKS + SCALAR_TASKS

def _get_backbone(backbone: str | nn.Module, **backbone_kwargs) -> nn.Module:
    if isinstance(backbone, nn.Module):
        return backbone
    return BACKBONE_REGISTRY.build(backbone, **backbone_kwargs)


def _get_decoder(decoder: str | nn.Module, channel_list: list[int], **decoder_kwargs) -> nn.Module:
    if isinstance(decoder, nn.Module):
        return decoder
    return DECODER_REGISTRY.build(decoder, channel_list, **decoder_kwargs)

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
                directly. The backbone should have and `out_channels` attribute.
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

        decoder_kwargs, kwargs = extract_prefix_keys(kwargs, "decoder_")
        decoder = _get_decoder(decoder, channel_list, **decoder_kwargs)

        decoder_kwargs, kwargs = extract_prefix_keys(kwargs, "decoder_")
        head_kwargs, kwargs = extract_prefix_keys(kwargs, "head_")

        # if num_classes:
        #     if decoder_sources[0] != "mmseg":
        #         head_kwargs["num_classes"] = num_classes
        #     else:
        #         decoder_kwargs["num_classes"] = num_classes
        decoder = _get_decoder(decoder, channel_list, **decoder_kwargs)

        if aux_decoders is None:
            return _build_appropriate_model(task, backbone, decoder, head_kwargs, neck_list, rescale=rescale)

        to_be_aux_decoders: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] = []
        for aux_decoder in aux_decoders:
            args = aux_decoder.decoder_args if aux_decoder.decoder_args else {}
            aux_decoder_cls: nn.Module = DECODER_REGISTRY[aux_decoder.decoder]
            aux_decoder_kwargs, kwargs = extract_prefix_keys(args, "decoder_")
            aux_decoder_instance = aux_decoder_cls(channel_list, **aux_decoder_kwargs)

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
            neck_list,
            rescale=rescale,
            auxiliary_heads=to_be_aux_decoders,
        )


def _build_appropriate_model(
    task: str,
    backbone: nn.Module,
    decoder: nn.Module,
    head_kwargs: dict,
    necks: list[nn.Module] | None = None,
    rescale: bool = True,  # noqa: FBT001, FBT002
    auxiliary_heads: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] | None = None,
):
    if necks is not None:
        necks = nn.Sequential(*necks)
    if task in PIXEL_WISE_TASKS:
        return PixelWiseModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            neck=necks,
            rescale=rescale,
            auxiliary_heads=auxiliary_heads,
        )
    elif task in SCALAR_TASKS:
        return ScalarOutputModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            neck=necks,
            auxiliary_heads=auxiliary_heads,
        )

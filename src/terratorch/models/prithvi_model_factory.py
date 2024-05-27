from collections.abc import Callable

import timm
import torch
from torch import nn

import terratorch.models.decoders as decoder_registry
from terratorch.datasets import HLSBands
from terratorch.models.model import (
    AuxiliaryHead,
    AuxiliaryHeadWithDecoderWithoutInstantiatedHead,
    Model,
    ModelFactory,
    register_factory,
)
from terratorch.models.pixel_wise_model import PixelWiseModel
from terratorch.models.scalar_output_model import ScalarOutputModel

PIXEL_WISE_TASKS = ["segmentation", "regression"]
SCALAR_TASKS = ["classification"]
SUPPORTED_TASKS = PIXEL_WISE_TASKS + SCALAR_TASKS


class DecoderNotFoundError(Exception):
    pass

@register_factory
class PrithviModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module,
        bands: list[HLSBands | int],
        in_channels: int | None = None,  # this should be removed, can be derived from bands. But it is a breaking change
        num_classes: int | None = None,
        pretrained: bool = True,  # noqa: FBT001, FBT002
        num_frames: int = 1,
        prepare_features_for_image_model: Callable | None = None,
        aux_decoders: list[AuxiliaryHead] | None = None,
        rescale: bool = True,  # noqa: FBT002, FBT001
        **kwargs,
    ) -> Model:
        """Model factory for prithvi models.

        Further arguments to be passed to the backbone, decoder or head. They should be prefixed with
        `backbone_`, `decoder_` and `head_` respectively.

        Args:
            task (str): Task to be performed. Currently supports "segmentation" and "regression".
            backbone (str, nn.Module): Backbone to be used. If string, should be able to be parsed
                by the specified factory. Defaults to "prithvi_100".
            decoder (Union[str, nn.Module], optional): Decoder to be used for the segmentation model.
                    If a string, it will be created from a class exposed in decoder.__init__.py with the same name.
                    If an nn.Module, we expect it to expose a property `decoder.output_embed_dim`.
                    Will be concatenated with a Conv2d for the final convolution. Defaults to "FCNDecoder".
            in_channels (int, optional): Number of input channels. Defaults to 3.
            bands (list[terratorch.datasets.HLSBands], optional): Bands the model will be trained on.
                    Should be a list of terratorch.datasets.HLSBands.
                    Defaults to [HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE].
            num_classes (int, optional): Number of classes. None for regression tasks.
            pretrained (Union[bool, Path], optional): Whether to load pretrained weights for the backbone, if available.
                Defaults to True.
            num_frames (int, optional): Number of timesteps for the model to handle. Defaults to 1.
            prepare_features_for_image_model (Callable | None): Function to be called on encoder features
                before passing them to the decoder. Defaults to None, which applies the identity function.
            aux_decoders (list[AuxiliaryHead] | None): List of AuxiliaryHead deciders to be added to the model.
                These decoders take the input from the encoder as well.
            rescale (bool): Whether to apply bilinear interpolation to rescale the model output if its size
                is different from the ground truth. Only applicable to pixel wise models
                (e.g. segmentation, pixel wise regression). Defaults to True.


        Returns:
            nn.Module: Full model with encoder, decoder and head.
        """
        if not torch.cuda.is_available():
            self.CPU_ONLY = True
        else:
            self.CPU_ONLY = False

        bands = [HLSBands.try_convert_to_hls_bands_enum(b) for b in bands]
        if in_channels is None:
            in_channels = len(bands)
        # TODO: support auxiliary heads
        if not isinstance(backbone, nn.Module):
            if not backbone.startswith("prithvi_"):
                msg = "This class only handles models for `prithvi` encoders"
                raise NotImplementedError(msg)

            task = task.lower()
            if task not in SUPPORTED_TASKS:
                msg = f"Task {task} not supported. Please choose one of {SUPPORTED_TASKS}"
                raise NotImplementedError(msg)

            backbone_kwargs, kwargs = _extract_prefix_keys(kwargs, "backbone_")

            backbone: nn.Module = timm.create_model(
                backbone,
                pretrained=pretrained,
                in_chans=in_channels,  # this can be removed, can be derived from bands. But is a breaking change.
                num_frames=num_frames,
                bands=bands,
                features_only=True,
                **backbone_kwargs,
            )
        # allow decoder to be a module passed directly
        decoder_cls = _get_decoder(decoder)

        decoder_kwargs, kwargs = _extract_prefix_keys(kwargs, "decoder_")

        # TODO: remove this
        decoder: nn.Module = decoder_cls(backbone.feature_info.channels(), **decoder_kwargs)
        # decoder: nn.Module = decoder_cls([128, 256, 512, 1024], **decoder_kwargs)

        head_kwargs, kwargs = _extract_prefix_keys(kwargs, "head_")
        if num_classes:
            head_kwargs["num_classes"] = num_classes
        if aux_decoders is None:
            return _build_appropriate_model(
                task, backbone, decoder, head_kwargs, prepare_features_for_image_model, rescale=rescale
            )

        to_be_aux_decoders: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] = []
        for aux_decoder in aux_decoders:
            args = aux_decoder.decoder_args if aux_decoder.decoder_args else {}
            aux_decoder_cls: nn.Module = _get_decoder(aux_decoder.decoder)
            aux_decoder_kwargs, kwargs = _extract_prefix_keys(args, "decoder_")
            aux_decoder_instance = aux_decoder_cls(backbone.feature_info.channels(), **aux_decoder_kwargs)

            aux_head_kwargs, kwargs = _extract_prefix_keys(args, "head_")
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
            prepare_features_for_image_model,
            rescale=rescale,
            auxiliary_heads=to_be_aux_decoders,
        )


def _build_appropriate_model(
    task: str,
    backbone: nn.Module,
    decoder: nn.Module,
    head_kwargs: dict,
    prepare_features_for_image_model: Callable,
    rescale: bool = True,  # noqa: FBT001, FBT002
    auxiliary_heads: dict | None = None,
):
    if task in PIXEL_WISE_TASKS:
        return PixelWiseModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            prepare_features_for_image_model=prepare_features_for_image_model,
            rescale=rescale,
            auxiliary_heads=auxiliary_heads,
        )
    elif task in SCALAR_TASKS:
        return ScalarOutputModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            prepare_features_for_image_model=prepare_features_for_image_model,
            auxiliary_heads=auxiliary_heads,
        )


def _get_decoder(decoder: str | nn.Module) -> nn.Module:
    if isinstance(decoder, nn.Module):
        return decoder
    if isinstance(decoder, str):
        try:
            decoder = getattr(decoder_registry, decoder)
            return decoder
        except AttributeError as decoder_not_found_exception:
            msg = f"Decoder {decoder} was not found in the registry."
            raise DecoderNotFoundError(msg) from decoder_not_found_exception
    msg = "Decoder must be str or nn.Module"
    raise Exception(msg)


def _extract_prefix_keys(d: dict, prefix: str) -> dict:
    extracted_dict = {}
    remaining_dict = {}
    for k, v in d.items():
        if k.startswith(prefix):
            extracted_dict[k.split(prefix)[1]] = v
        else:
            remaining_dict[k] = v

    return extracted_dict, remaining_dict

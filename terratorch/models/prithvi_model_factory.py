# Copyright contributors to the Terratorch project

import warnings
from collections.abc import Callable
from typing import Optional

from torch import nn
import timm
import terratorch.models.decoders as decoder_registry
from terratorch.datasets import HLSBands
from terratorch.models import EncoderDecoderFactory
from terratorch.models.model import (
    AuxiliaryHead,
    Model,
    ModelFactory,
    AuxiliaryHeadWithDecoderWithoutInstantiatedHead
)
from terratorch.registry import MODEL_FACTORY_REGISTRY
from terratorch.models.pixel_wise_model import PixelWiseModel

PIXEL_WISE_TASKS = ["segmentation", "regression", "pretraining"]
SCALAR_TASKS = ["classification"]
SUPPORTED_TASKS = PIXEL_WISE_TASKS + SCALAR_TASKS


@MODEL_FACTORY_REGISTRY.register
class PrithviModelFactory(ModelFactory):
    def __init__(self) -> None:
        self._factory: EncoderDecoderFactory = EncoderDecoderFactory()
    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: Optional[str | nn.Module],
        bands: list[HLSBands | int],
        in_channels: int
        | None = None,  # this should be removed, can be derived from bands. But it is a breaking change
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
                    If an nn.Module, we expect it to expose a property `decoder.out_channels`.
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
        # This factory will replaced by the more general EncoderDecoder factory in the future.
        warnings.warn("PrithviModelFactory is deprecated. Please switch to EncoderDecoderFactory.", stacklevel=1)

        if task in ["segmentation", "regression", "classification"]:
            if not decoder:
                raise ValueError(f"Decoder is required for 'segmentation' and 'regression' tasks, but received {decoder}.")

        bands = [HLSBands.try_convert_to_hls_bands_enum(b) for b in bands]

        if in_channels is None:
            in_channels = len(bands)

        # TODO: support auxiliary heads

        kwargs["backbone_bands"] = bands
        kwargs["backbone_in_chans"] = in_channels
        kwargs["backbone_pretrained"] = pretrained
        kwargs["backbone_num_frames"] = num_frames

        if prepare_features_for_image_model:
            msg = (
                "This functionality is no longer supported. Please migrate to EncoderDecoderFactory\
                         and use necks."
            )
            raise RuntimeError(msg)

        if not isinstance(backbone, nn.Module):
            if not backbone.startswith("prithvi_"):
                msg = "This class only handles models for `prithvi` encoders"
                raise NotImplementedError(msg)

            task = task.lower()
            if task not in SUPPORTED_TASKS:
                msg = f"Task {task} not supported. Please choose one of {SUPPORTED_TASKS}"
                raise NotImplementedError(msg)

            backbone_kwargs, kwargs = _extract_prefix_keys(kwargs, "backbone_")
            # These params are used in case we need a SMP decoder
            # but should not be used for timm encoder
            output_stride = backbone_kwargs.pop("output_stride", None)
            out_channels = backbone_kwargs.pop("out_channels", None)

            # When the task is "pre-training", the original
            # decoder is maintained as is and the training is performed from
            # scratch 
            if task == "pretraining":
                features_only = False
            else:
                features_only = True

            # Instantiating backbone
            backbone: nn.Module = timm.create_model(
                backbone,
                features_only=features_only,
                **backbone_kwargs,
            )

        # These steps are necessary just when a fine-tuning task is 
        # performed (segmentation and regression).
        if task in ["segmentation", "regression", "classification"]:

            decoder_kwargs, kwargs = _extract_prefix_keys(kwargs, "decoder_")
            # TODO: remove this
            if decoder.startswith("smp_"):
                decoder: nn.Module = _get_smp_decoder(
                    decoder,
                    backbone_kwargs,
                    decoder_kwargs,
                    out_channels,
                    in_channels,
                    num_classes,
                    output_stride,
                )
            else:
                # allow decoder to be a module passed directly
                decoder_cls = _get_decoder(decoder)
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
        else: 
            # By-passing entities not required for pre-training tasks.
            decoder = None
            aux_head_kwargs = None 
            head_kwargs = None 
            to_be_aux_decoders = None


        return self._factory.build_model(task,
                                         backbone,
                                         decoder,
                                         num_classes=num_classes,
                                         necks=None,
                                         aux_decoders=aux_decoders,
                                         rescale=rescale,
                                         **kwargs)


class SMPDecoderForPrithviWrapper(nn.Module):
    """
    A wrapper for SMP decoders designed to handle single or multiple embeddings with specified indices.

    Attributes:
        decoder (nn.Module): The SMP decoder module being wrapped.
        channels (int): The number of output channels of the decoder.
        in_index (Union[int, List[int]]): Index or indices of the embeddings to pass to the decoder.

    Methods:
        forward(x: List[torch.Tensor]) -> torch.Tensor:
            Forward pass for embeddings with specified indices.
    """

    def __init__(self, decoder, num_channels, in_index=-1) -> None:
        """
        Args:
            decoder (nn.Module): The SMP decoder module to be wrapped.
            num_channels (int): The number of output channels of the decoder.
            in_index (Union[int, List[int]], optional): Index or indices of the input embeddings to pass to the decoder.
            Defaults to -1.
        """
        super().__init__()
        self.decoder = decoder
        self.channels = num_channels
        self.in_index = in_index

    @property
    def output_embed_dim(self):
        return self.channels

    def forward(self, x):
        if isinstance(self.in_index, int):
            selected_inputs = [x[self.in_index]]
        else:
            selected_inputs = [x[i] for i in self.in_index]

        return self.decoder(*selected_inputs)


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
            #prepare_features_for_image_model=prepare_features_for_image_model,
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


def _get_smp_decoder(
    decoder: str,
    backbone_kwargs: dict,
    decoder_kwargs: dict,
    out_channels: list[int] | int,
    in_channels: int,
    num_classes: int,
    output_stride: int,
):
    """
    Creates and configures a decoder from the Segmentation Models Pytorch (SMP) library.

    This function constructs a decoder module based on the specified parameters and wraps it in a
    custom wrapper that allows handling single or multiple embeddings. It also ensures that the
    appropriate encoder parameters are passed and registered correctly.

    Args:
        decoder (str): The name of the SMP decoder to use.
        backbone_kwargs (dict): Dictionary of parameters for configuring the backbone.
        decoder_kwargs (dict): Dictionary of parameters specific to the decoder.
        out_channels (Union[list[int], int]): The number of output channels for each layer of the decoder.
        in_channels (int): The number of input channels.
        num_classes (int): The number of output classes for the model.
        output_stride (int): The output stride of the decoder.

    Returns:
        SMPDecoderForPrithviWrapper: A wrapped decoder module configured based on the provided parameters.

    Raises:
        ValueError: If the specified decoder is not supported by SMP.
    """
    decoder = decoder.removeprefix("smp_")
    decoder_module = getattr(smp, decoder, None)
    if decoder_module is None:
        msg = f"Decoder {decoder} is not supported in SMP."
        raise ValueError(msg)

    # Little hack to make SMP model accept our encoder.
    # passes a dummy encoder to be changed later.
    # this is needed to pass encoder params.
    aux_kwargs, decoder_kwargs = _extract_prefix_keys(decoder_kwargs, "aux_")
    smp_kwargs, decoder_kwargs = _extract_prefix_keys(decoder_kwargs, "smp_")
    backbone_kwargs["out_channels"] = out_channels
    backbone_kwargs["output_stride"] = output_stride
    aux_kwargs = None if aux_kwargs == {} else aux_kwargs

    dummy_encoder = make_smp_encoder()

    register_custom_encoder(dummy_encoder, backbone_kwargs, None)

    dummy_encoder = dummy_encoder(
        depth=smp_kwargs["encoder_depth"],
        output_stride=backbone_kwargs["output_stride"],
        out_channels=backbone_kwargs["out_channels"],
    )

    model_args = {
        "encoder_name": "SMPEncoderWrapperWithPFFIM",
        "encoder_weights": None,
        "in_channels": in_channels,
        "classes": num_classes,
        **smp_kwargs,
    }

    # Creates model with dummy encoder and decoder.
    model = decoder_module(**model_args, aux_params=aux_kwargs)

    smp_decoder = SMPDecoderForPrithviWrapper(
        decoder=model.decoder,
        num_channels=out_channels[-1],
        in_index=decoder_kwargs["in_index"],
    )

    return smp_decoder


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
            extracted_dict[k[len(prefix) :]] = v
        else:
            remaining_dict[k] = v

    return extracted_dict, remaining_dict

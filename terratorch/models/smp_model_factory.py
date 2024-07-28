# Copyright contributors to the Terratorch project

import importlib
from collections.abc import Callable

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F  # noqa: N812
from segmentation_models_pytorch.encoders import encoders as smp_encoders
from torch import nn

from terratorch.datasets import HLSBands
from terratorch.models.model import Model, ModelFactory, ModelOutput, register_factory


class SMPDecoderForPrithviWrapper(nn.Module):
    """
    A wrapper for SMP decoders designed to handle single or multiple embeddings.

    Attributes:
        decoder (nn.Module): The SMP decoder module being wrapped.
        channels (int): The number of output channels of the decoder.

    Methods:
        forward_multiple_embeds(*x) -> torch.Tensor:
            Forward pass for multiple embeddings.
        forward_single_embed(x) -> torch.Tensor:
            Forward pass for a single embedding.
    """

    def __init__(self, decoder, num_channels) -> None:
        """
        Args:
            decoder (nn.Module): The SMP decoder module to be wrapped.
            num_channels (int): The number of output channels of the decoder.
        """
        super().__init__()
        self.decoder = decoder
        self.channels = num_channels

    @property
    def output_embed_dim(self):
        return self.channels

    def forward_multiple_embeds(self, *x):
        return self.decoder(*x)

    def forward_single_embed(self, x):
        return self.decoder(x[-1])


class SMPModelWrapper(Model, nn.Module):
    """
    Wrapper class for SMP models.

    This class provides additional functionalities on top of SMP models.

    Attributes:
        rescale (bool): Whether to rescale the output to match the input dimensions.
        smp_model (nn.Module): The base SMP model being wrapped.
        final_act (nn.Module): The final activation function to be applied on the output.
        squeeze_single_class (bool): Whether to squeeze the output if there is a single output class.

    Methods:
        forward(x: torch.Tensor) -> ModelOutput:
            Forward pass through the model, optionally rescaling the output.
        freeze_encoder() -> None:
            Freezes the parameters of the encoder part of the model.
        freeze_decoder() -> None:
            Freezes the parameters of the decoder part of the model.
    """

    def __init__(self, smp_model, rescale=True, relu=False, squeeze_single_class=False) -> None: # noqa: FBT002
        super().__init__()
        """
        Args:
            smp_model (nn.Module): The base SMP model to be wrapped.
            rescale (bool, optional): Whether to rescale the output to match the input dimensions. Defaults to True.
            relu (bool, optional): Whether to apply ReLU activation on the output.
            If False, Identity activation is used. Defaults to False.
            squeeze_single_class (bool, optional): Whether to squeeze the output if there is a single output class.
            Defaults to False.
        """
        self.rescale = rescale
        self.smp_model = smp_model
        self.final_act = nn.ReLU() if relu else nn.Identity()
        self.squeeze_single_class = squeeze_single_class

    def forward(self, x):
        input_size = x.shape[-2:]
        smp_output = self.smp_model(x)
        smp_output = self.final_act(smp_output)

        # TODO: support auxiliary head labels
        if isinstance(smp_output, tuple):
            smp_output, labels = smp_output

        if smp_output.shape[1] == 1 and self.squeeze_single_class:
            smp_output = smp_output.squeeze(1)

        if self.rescale and smp_output.shape[-2:] != input_size:
            smp_output = F.interpolate(smp_output, size=input_size, mode="bilinear")
        return ModelOutput(smp_output)

    def freeze_encoder(self):
        freeze_module(self.smp_model.encoder)

    def freeze_decoder(self):
        freeze_module(self.smp_model.decoder)


@register_factory
class SMPModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str,
        model: str,
        bands: list[HLSBands | int],
        in_channels: int | None = None,
        num_classes: int = 1,
        pretrained: str | bool | None = True, # noqa: FBT002
        prepare_features_for_image_model: Callable | None = None,
        regression_relu: bool = False, # noqa: FBT001, FBT002
        **kwargs,
    ) -> Model:
        """
        Factory class for creating SMP (Segmentation Models Pytorch) based models with optional customization.

        This factory handles the instantiation of segmentation and regression models using specified
        encoders and decoders from the SMP library, along with custom modifications and extensions such
        as auxiliary decoders or modified encoders.

        Attributes:
            task (str): Specifies the task for which the model is being built. Supported tasks include
                        "segmentation" and "regression".
            backbone (str, nn.Module): Specifies the backbone model to be used. If a string, it should be
                        recognized by the model factory and be able to be parsed appropriately.
            decoder (str): Specifies the decoder to be used for constructing the
                        segmentation model.
            bands (list[terratorch.datasets.HLSBands | int]): A list specifying the bands that the model
                        will operate on. These are expected to be from terratorch.datasets.HLSBands.
            in_channels (int, optional): Specifies the number of input channels. Defaults to None.
            num_classes (int, optional): The number of output classes for the model.
            pretrained (bool | Path, optional): Indicates whether to load pretrained weights for the
                        backbone. Can also specify a path to weights. Defaults to True.
            num_frames (int, optional): Specifies the number of timesteps the model should handle. Useful
                        for temporal models.
            regression_relu (bool): Whether to apply ReLU activation in the case of regression tasks.
            **kwargs: Additional arguments that might be passed to further customize the backbone, decoder,
                        or any auxiliary heads. These should be prefixed appropriately

        Raises:
            ValueError: If the specified decoder is not supported by SMP.
            Exception: If the specified task is not "segmentation" or "regression".

        Returns:
            nn.Module: A model instance wrapped in SMPModelWrapper configured according to the specified
                    parameters and tasks.
        """

        bands = [HLSBands.try_convert_to_hls_bands_enum(b) for b in bands]
        if in_channels is None:
            in_channels = len(bands)

        # Gets decoder module.
        model_module = getattr(smp, model, None)
        if model_module is None:
            msg = f"Decoder {model} is not supported in SMP."
            raise ValueError(msg)

        backbone_kwargs = _extract_prefix_keys(kwargs, "backbone_")  # Encoder params should be prefixed backbone_
        smp_kwargs = _extract_prefix_keys(backbone_kwargs, "smp_")  # Smp model params should be prefixed smp_
        aux_params = _extract_prefix_keys(backbone_kwargs, "aux_")  # Auxiliary head params should be prefixed aux_
        aux_params = None if aux_params == {} else aux_params

        if isinstance(pretrained, bool):
            if pretrained:
                pretrained = "imagenet"
            else:
                pretrained = None

        # If encoder not currently supported by SMP (custom encoder).
        if backbone not in smp_encoders:
            # These params must be included in the config file with appropriate prefix.
            required_params = {
                "encoder_depth": smp_kwargs,
                "out_channels": backbone_kwargs,
                "output_stride": backbone_kwargs,
            }

            for param, config_dict in required_params.items():
                if param not in config_dict:
                    msg = f"Config must include the '{param}' parameter"
                    raise ValueError(msg)

            # Using new encoder.
            backbone_class = self._make_smp_encoder(backbone)
            backbone_kwargs["prepare_features_for_image_model"] = prepare_features_for_image_model
            # Registering custom encoder into SMP.
            _register_custom_encoder(backbone_class, backbone_kwargs, pretrained)

            model_args = {
                "encoder_name": "SMPEncoderWrapperWithPFFIM",
                "encoder_weights": pretrained,
                "in_channels": in_channels,
                "classes": num_classes,
                **smp_kwargs,
            }
        # Using SMP encoder.
        else:
            model_args = {
                "encoder_name": backbone,
                "encoder_weights": pretrained,
                "in_channels": in_channels,
                "classes": num_classes,
                **smp_kwargs,
            }

        model = model_module(**model_args, aux_params=aux_params)

        return SMPModelWrapper(
            model, relu=task == "regression" and regression_relu, squeeze_single_class=task == "regression"
        )


def get_smp_decoder(
    decoder: str,
    backbone_kwargs: dict,
    smp_kwargs: dict,
    aux_kwargs: dict,
    args: dict,
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
        smp_kwargs (dict): Dictionary of parameters specific to the SMP model.
        aux_kwargs (dict): Dictionary of parameters for auxiliary configurations.
        args (dict): Additional arguments, typically containing head-specific parameters.
        out_channels (list[int] | int): The number of output channels for each layer of the decoder.
        in_channels (int): The number of input channels.
        num_classes (int): The number of output classes for the model.
        output_stride (int): The output stride of the decoder.

    Returns:
        SMPDecoderForPrithviWrapper: A wrapped decoder module configured based on the provided parameters.

    Raises:
        ValueError: If the specified decoder is not supported by SMP.
    """

    # Gets decoder class.
    decoder = decoder.removeprefix("smp_")
    decoder_module = getattr(smp, decoder, None)
    if decoder_module is None:
        msg = f"Decoder {decoder} is not supported in SMP."
        raise ValueError(msg)

    # head args for handling embeds.
    head_kwargs = _extract_prefix_keys(args, "head_")

    # Little hack to make SMP model accept our encoder.
    # passes a dummy encoder to be changed later.
    # this is needed to pass encoder params.
    backbone_kwargs["out_channels"] = out_channels
    backbone_kwargs["output_stride"] = output_stride
    aux_kwargs = None if aux_kwargs == {} else aux_kwargs

    dummy_encoder = _make_smp_encoder()

    _register_custom_encoder(dummy_encoder, backbone_kwargs, None)

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

    # Wrapper for SMP Decoder.
    smp_decoder = SMPDecoderForPrithviWrapper(decoder=model.decoder, num_channels=out_channels[-1])
    if "multiple_embed" in head_kwargs:
        smp_decoder.forward = smp_decoder.forward_multiple_embeds
    else:
        smp_decoder.forward = smp_decoder.forward_single_embed

    return smp_decoder


# Registers a custom encoder into SMP.
def _register_custom_encoder(encoder, params, pretrained):
    smp_encoders["SMPEncoderWrapperWithPFFIM"] = {
        "encoder": encoder,
        "params": params,
        "pretrained_settings": pretrained
    }


# Gets class either from string or from Module reference.
def _make_smp_encoder(encoder=None):
    if isinstance(encoder, str):
        base_class = _get_class_from_string(encoder)
    else:
        base_class = nn.Module

    # Wrapper needed to include SMP params and PFFIM
    class SMPEncoderWrapperWithPFFIM(base_class):
        def __init__(
            self,
            depth: int,
            output_stride: int,
            out_channels: list[int],
            prepare_features_for_image_model: Callable | None = None,
            *args,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self._depth = depth
            self._output_stride = output_stride
            self._out_channels = out_channels
            self.model = None

            if prepare_features_for_image_model:
                self.prepare_features_for_image_model = prepare_features_for_image_model
            elif not hasattr(super(), "prepare_features_for_image_model"):
                self.prepare_features_for_image_model = lambda x: x

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.model:
                features = self.model(x)
                if hasattr(self.model, "prepare_features_for_image_model"):
                    return self.model.prepare_features_for_image_model(features)

            features = super().forward(x)
            return self.prepare_features_for_image_model(features)

        @property
        def out_channels(self):
            if hasattr(super(), "out_channels"):
                return super().out_channels()

            return self._out_channels

        @property
        def output_stride(self):
            if hasattr(super(), "output_stride"):
                return super().output_stride()

            return min(self._output_stride, 2**self._depth)

        def set_in_channels(self, in_channels, pretrained):
            if hasattr(super(), "set_in_channels"):
                return super().set_in_channels(in_channels, pretrained)
            else:
                pass

        def make_dilated(self, output_stride):
            if hasattr(super(), "make_dilated"):
                return super().make_dilated(output_stride)
            else:
                pass

    return SMPEncoderWrapperWithPFFIM


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


def _get_class_from_string(class_path):
    try:
        module_path, name = class_path.rsplit(".", 1)
    except ValueError as vr:
        msg = "Path must contain a '.' separating module from the class name"
        raise ValueError(msg) from vr

    try:
        module = importlib.import_module(module_path)
    except ImportError as ie:
        msg = f"Could not import module '{module_path}'."
        raise ImportError(msg) from ie

    try:
        return getattr(module, name)
    except AttributeError as ae:
        msg = f"The class '{name}' was not found in the module '{module_path}'."
        raise AttributeError(msg) from ae


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad_(False)

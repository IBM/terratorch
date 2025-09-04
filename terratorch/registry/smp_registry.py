import importlib
import inspect
import logging
from collections.abc import Callable, Mapping, Set

from torch import nn

from terratorch.models.smp_model_factory import make_smp_encoder, register_custom_encoder
from terratorch.models.utils import extract_prefix_keys
from terratorch.registry import DECODER_REGISTRY


class SMPDecoderWrapper(nn.Module):
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

    def __init__(self, decoder, out_channels: int) -> None:
        """
        Args:
            decoder (nn.Module): The SMP decoder module to be wrapped.
            out_channels (int): Output channels of the decoder, needed for terratorch head.
                Necessary since not all smp decoders have the `out_channels` attribute.
            Defaults to -1.
        """
        super().__init__()
        self._smp_decoder = decoder
        self.out_channels = out_channels

    def forward(self, x):
        return self._smp_decoder(*x)


class SMPRegistry(Set):
    """Registry wrapper for segmentation_models_pytorch.

    Not all decoders are guaranteed to work with all encoders without additional necks.
    Please check smp documentation to understand the embedding spatial dimensions expected by each decoder.

    In particular, smp seems to assume the first feature in the passed feature list has the same spatial resolution
    as the input, which may not always be true, and may break some decoders.

    In addition, for some decoders, the final 2 features have the same spatial resolution.
    Adding the AddBottleneckLayer neck will make this compatible.
    """
    includes_head: bool = False
    def __init__(self):
        if not importlib.util.find_spec("segmentation_models_pytorch"):
            msg = "segmentation_models_pytorch must be installed to instantiate an SMPRegistry"
            raise ImportError(msg)

        self.smp_decoders = {x for x, _ in inspect.getmembers(smp, inspect.isclass)}

    def register(self, constructor: Callable | type) -> Callable:
        raise NotImplementedError()

    def build(self, name: str, out_channels: list[int], **decoder_kwargs) -> nn.Module:
        try:
            decoder_module = getattr(smp, name)
        except AttributeError as e:
            msg = f"Decoder {name} not found"
            raise KeyError(msg) from e

        # Little hack to make SMP model accept our encoder.
        # passes a dummy encoder to be changed later.
        # this is needed to pass encoder params.
        aux_kwargs, decoder_kwargs = extract_prefix_keys(decoder_kwargs, "aux_")

        dummy_backbone_kwargs = {}
        dummy_backbone_kwargs["out_channels"] = out_channels
        dummy_backbone_kwargs["output_stride"] = 1  # hardcode to 1 for dummy encoder
        if len(aux_kwargs) == 0:
            aux_kwargs = None

        dummy_encoder = make_smp_encoder()

        register_custom_encoder(dummy_encoder, dummy_backbone_kwargs, None)
        encoder_depth = len(out_channels) - 1
        dummy_encoder = dummy_encoder(depth=encoder_depth, **dummy_backbone_kwargs)

        # encoder parameters are dummies so are hardcoded to 1 here
        model_args = {
            "encoder_name": "SMPEncoderWrapperWithPFFIM",
            "encoder_weights": None,
            "in_channels": 1,
            "classes": 1,
            "encoder_depth": encoder_depth,
            **decoder_kwargs,
        }

        # Creates model with dummy encoder and decoder.
        model = decoder_module(**model_args, aux_params=aux_kwargs)

        smp_decoder = SMPDecoderWrapper(
            model.decoder,
            model.segmentation_head[
                0
            ].in_channels,  # not all decoders have a out_channels property. get it from the segmentation head that the model creates
        )

        return smp_decoder

    def __iter__(self):
        return iter(self.smp_decoders)

    def __len__(self):
        return len(self.smp_decoders)

    def __contains__(self, key):
        return key in self.smp_decoders

    def __getitem__(self, name):
        try:
            return getattr(smp, name)
        except AttributeError as e:
            msg = f"Decoder {name} not found"
            raise KeyError(msg) from e

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return f"SMP registry with {len(self)} registered backbones"


if importlib.util.find_spec("segmentation_models_pytorch"):
    import segmentation_models_pytorch as smp

    SMP_DECODER_REGISTRY = SMPRegistry()
    DECODER_REGISTRY.register_source("smp", SMP_DECODER_REGISTRY)
else:
    logging.getLogger("terratorch").debug("segmentation_models_pytorch not installed, so SMPRegistry not created")

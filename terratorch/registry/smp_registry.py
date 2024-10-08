import importlib
import inspect
import logging
from collections.abc import Callable, Mapping

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
        self.output_embed_dim = out_channels

    def forward(self, x):
        return self._smp_decoder(*x)


class SMPRegistry(Mapping):
    """Registry wrapper for segmentation_models_pytorch.

    smp does not seem to have the set of supported models / decoders exposed at all in the API.
    The cleanest way I've found to handle this is to just enumerate this im this module.
    Unfortunately, this means there might be models that this registry can
    instantiate (they will work when the user calls .build with them), but
    will report `False` when tested with `in` and will not count towards the `len()` of the registry.

    This will have to be updated manually when SMP is updated...
    """

    def __init__(self):
        if not importlib.util.find_spec("segmentation_models_pytorch"):
            msg = "segmentation_models_pytorch must be installed to instantiate an SMPRegistry"
            raise ImportError(msg)

        self.smp_decoders = {x for x, _ in inspect.getmembers(smp, inspect.isclass)}

    def register(self, constructor: Callable | type) -> Callable:
        raise NotImplementedError()

    def build(self, name: str, out_channels: list[int], **decoder_kwargs) -> nn.Module:
        decoder_module = self[name]

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

        dummy_encoder = dummy_encoder(depth=len(out_channels), **dummy_backbone_kwargs)

        # encoder parameters are dummies so are hardcoded to 1 here
        model_args = {
            "encoder_name": "SMPEncoderWrapperWithPFFIM",
            "encoder_weights": None,
            "in_channels": 1,
            "classes": 1,
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
        return repr(self.smp_decoders)

    def __str__(self):
        return f"SMP registry with {len(self)} registered backbones"


if importlib.util.find_spec("segmentation_models_pytorch"):
    import segmentation_models_pytorch as smp

    SMP_DECODER_REGISTRY = SMPRegistry()
    DECODER_REGISTRY.register_source("smp", SMP_DECODER_REGISTRY)
else:
    logging.debug("segmentation_models_pytorch not installed, so SMPRegistry not created")

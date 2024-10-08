import importlib
import inspect
import logging
from collections.abc import Callable, Mapping

from torch import nn

from terratorch.models.smp_model_factory import make_smp_encoder, register_custom_encoder
from terratorch.models.utils import extract_prefix_keys
from terratorch.registry import DECODER_REGISTRY

import importlib
import inspect
import pdb


class MMsegDecoderWrapper(nn.Module):
    """
    A wrapper for decoders designed to handle single or multiple embeddings with specified indices.

    Attributes:
        decoder (nn.Module): The decoder module being wrapped.
        channels (int): The number of output channels of the decoder.
        in_index (Union[int, List[int]]): Index or indices of the embeddings to pass to the decoder.

    Methods:
        forward(x: List[torch.Tensor]) -> torch.Tensor:
            Forward pass for embeddings with specified indices.
    """

    def __init__(self, decoder) -> None:
        """
        Args:
            decoder (nn.Module): The decoder module to be wrapped.
            out_channels (int): Output channels of the decoder, needed for terratorch head.
                Necessary since not all smp decoders have the `out_channels` attribute.
            Defaults to -1.
        """
        super().__init__()
        self.decoder = decoder
        self.final_head_needed = False

    def forward(self, x):
        return self.decoder(x)


class MmsegDecoderRegistry(Mapping):
    """Registry wrapper for mmseg
    """

    def __init__(self):

        self.mmseg_reg=importlib.import_module("mmseg.models.decode_heads")
        self.mmseg_decoder_names = [x for x, y in inspect.getmembers(self.mmseg_reg, inspect.isclass)]
    
    def register(self, constructor: Callable | type) -> Callable:
        raise NotImplementedError()

    def build(self, name: str, in_channels, *constructor_args, **constructor_kwargs) -> nn.Module:
        """Build and return the component.
        Use prefixes ending with _ to forward to a specific source
        """
        decoder = self[name]
        if len(in_channels) == 1: in_channels = in_channels[0]
        if "num_classes" not in constructor_kwargs: raise Exception("Error: num_classes is a required argument for mmseg decoders. If you are using a mmseg decoder for a regression task please include num_classes=1.")
        decoder = decoder(in_channels=in_channels, *constructor_args, **constructor_kwargs)
        return MMsegDecoderWrapper(decoder)

    def __iter__(self):
        return iter(self.mmseg_decoder_names)

    def __len__(self):
        return len(self.mmseg_decoder_names)

    def __contains__(self, key):
        return key in self.mmseg_decoder_names

    def __getitem__(self, name):
        try:
            return getattr(self.mmseg_reg, name)
        except AttributeError as e:
            msg = f"Decoder {name} not found"
            raise KeyError(msg) from e

    def __repr__(self):
        return repr(self.mmseg_reg)

    def __str__(self):
        return f"Mmseg registry with {len(self)} registered backbones"


if importlib.util.find_spec("mmseg"):
    import mmseg

    MMSEG_DECODER_REGISTRY = MmsegDecoderRegistry()
    DECODER_REGISTRY.register_source("mmseg", MMSEG_DECODER_REGISTRY)
else:
    logging.debug("mmseg not installed, so MmsegDecoderRegistry not created")


from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
import warnings

from terratorch.registry import TERRATORCH_DECODER_REGISTRY
from .utils import ConvModule


# Adapted from MMSegmentation
@TERRATORCH_DECODER_REGISTRY.register
class GalileoDecoder(nn.Module):

    def __init__(
        self,
        encoder_embedding_size: int = 128,
        decoder_embedding_size: int = 128,
        depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
        max_patch_size: int = 8,
        learnable_channel_embeddings: bool = False,
        output_embedding_size: Optional[int] = None,
    ):

        super(GalileoDecoder, self).__init__()

        # Checking if the package galileo is installed.
        try:
          from galileo.galileo import Decoder
        except ModuleNotFoundError:
          raise Exception("It's necessary to install the package `galileo` to access these models: `pip install terratorch[galileo]`")

        self.decoder = Decoder(encoder_embedding_size=encoder_embedding_size,
                               decoder_embedding_size=decoder_embedding_size,
                               depth=depth,
                               mlp_ratio=mlp_ratio,
                               num_heads=num_heads,
                               max_sequence_length=max_sequence_length,
                               max_patch_size=max_patch_size,
                               learnable_channel_embeddings=learnable_channel_embeddings,
                               output_embedding_size=output_embedding_size
                               ) 


    def forward(self, *args, **kwargs):

        output = self.decoder(*args, **kwargs)[0]
        dims = tuple(output.shape)

        b = dims[0]
        h = dims[1]
        w = dims[2]
        extra_dims = dims[3:]

        output = output.reshape([-1, h, w, np.prod(extra_dims)])
        output = output.permute(0, -1, 1, 2)
        return output

import torch
from segmentation_models_pytorch.base.initialization import initialize_decoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from torch import nn

from terratorch.registry import TERRATORCH_DECODER_REGISTRY


@TERRATORCH_DECODER_REGISTRY.register
class UNetDecoder(nn.Module):
    """UNetDecoder. Wrapper around UNetDecoder from segmentation_models_pytorch to avoid ignoring the first layer."""

    def __init__(
        self, embed_dim: list[int], channels: list[int], use_batchnorm: bool = True, attention_type: str | None = None
    ):
        """Constructor

        Args:
            embed_dim (list[int]): Input embedding dimension for each input.
            channels (list[int]): Channels used in the decoder.
            use_batchnorm (bool, optional): Whether to use batchnorm. Defaults to True.
            attention_type (str | None, optional): Attention type to use. Defaults to None
        """
        if len(embed_dim) != len(channels):
            msg = "channels should have the same length as embed_dim"
            raise ValueError(msg)
        super().__init__()
        self.decoder = UnetDecoder(
            encoder_channels=[embed_dim[0], *embed_dim],
            decoder_channels=channels,
            n_blocks=len(channels),
            use_batchnorm=use_batchnorm,
            center=False,
            attention_type=attention_type,
        )
        initialize_decoder(self.decoder)
        self.out_channels = channels[-1]

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        # The first layer is ignored in the original UnetDecoder, so we need to duplicate the first layer
        x = [x[0].clone(), *x]
        return self.decoder(*x)

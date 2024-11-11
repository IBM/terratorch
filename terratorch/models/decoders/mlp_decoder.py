# Copyright contributors to the Terratorch project

"""Pass the features straight through
"""

from torch import Tensor, nn
import torch
from terratorch.registry import TERRATORCH_DECODER_REGISTRY


@TERRATORCH_DECODER_REGISTRY.register
class MLPDecoder(nn.Module):
    """Identity decoder. Useful to pass the feature straight to the head."""

    def __init__(self, embed_dim: int, channels: int = 100, out_dim:int = 100, activation: str = "ReLU", out_index=-1) -> None:
        """Constructor

        Args:
            embed_dim (int): Input embedding dimension
            out_index (int, optional): Index of the input list to take.. Defaults to -1.
        """
        
        super().__init__()
        self.embed_dim = embed_dim
        self.channels = channels
        self.dim = out_index
        self.n_inputs = len(self.embed_dim)
        self.out_channels = self.embed_dim[self.dim]
        self.hidden_layer = torch.nn.Linear(self.out_channels*self.n_inputs, self.out_channels)
        self.activation = getattr(nn, activation)()

    def forward(self, x: list[Tensor]):

        data_ = torch.cat(x, axis=1)
        data_ = data_.permute(0, 2, 3, 1)
        data_ = self.activation(self.hidden_layer(data_))
        data_ = data_.permute(0, 3, 1, 2)

        return data_ 

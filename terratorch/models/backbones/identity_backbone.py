import torch
import torch.nn as nn

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

@TERRATORCH_BACKBONE_REGISTRY.register
class IdentityBackbone(nn.Module):
    """Identity backbone that returns the input tensor as feature map - for reading in embeddings.

    Args:
        None.
    """

    def __init__(self, out_channels: list[int] | None = None):
        """
        Constructor for the IdentityBackbone.
        Args:
            out_channels (list[int] | None, optional): Output channels of the backbone.
        """
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        """
        Forward pass of the identity backbone.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            List[torch.Tensor]: A single-element list containing the input.
        """
        return x
# Copyright contributors to the Terratorch project

from torch import nn


class SegmentationHead(nn.Module):
    """Segmentation head"""

    def __init__(
        self, in_channels: int, num_classes: int, channel_list: list[int] | None = None, dropout: float = 0
    ) -> None:
        """Constructor

        Args:
            in_channels (int): Number of input channels
            num_classes (int): Number of output classes
            channel_list (list[int] | None, optional):  List with number of channels for each Conv
                layer to be created. Defaults to None.
            dropout (float, optional): Dropout value to apply. Defaults to 0.
        """
        super().__init__()
        self.num_classes = num_classes
        if channel_list is None:
            pre_head = nn.Identity()
        else:

            def block(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1), nn.ReLU()
                )

            channel_list = [in_channels, *channel_list]
            pre_head = nn.Sequential(
                *[block(channel_list[i], channel_list[i + 1]) for i in range(len(channel_list) - 1)]
            )
            in_channels = channel_list[-1]
        dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.head = nn.Sequential(
            pre_head,
            dropout,
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_classes,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return self.head(x)

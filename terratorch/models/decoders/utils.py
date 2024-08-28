import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

"""
Adapted from https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py
"""
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, inplace=False) -> None:  # noqa: FBT002
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))



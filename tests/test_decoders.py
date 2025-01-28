import importlib
import os
import numpy as np

import pytest
import timm
import torch

import terratorch  # noqa: F401

from terratorch.models.decoders.aspp_head import ASPPSegmentationHead
from terratorch.models.decoders.unet_decoder import UNetDecoder
import gc


def test_aspphead():
    dilations = (1, 6, 12, 18)
    in_channels = 6
    channels = 10
    decoder = ASPPSegmentationHead(dilations=dilations, in_channels=in_channels, channels=channels, num_classes=2)

    image = [torch.from_numpy(np.random.rand(2, 6, 224, 224).astype("float32"))]

    assert decoder(image).shape == (2, 2, 224, 224)

    gc.collect()


def test_unetdecoder():
    embed_dim = [64, 128, 256, 512]
    channels = [256, 128, 64, 32]
    decoder = UNetDecoder(embed_dim=embed_dim, channels=channels)

    image = [
        torch.from_numpy(np.random.rand(2, 64, 224, 224).astype("float32")),
        torch.from_numpy(np.random.rand(2, 128, 112, 112).astype("float32")),
        torch.from_numpy(np.random.rand(2, 256, 56, 56).astype("float32")),
        torch.from_numpy(np.random.rand(2, 512, 28, 28).astype("float32")),
    ]

    assert decoder(image).shape == (
        2,
        32,
        448,
        448,
    )  # it doubles the size of the first input as it assumes it is already downsampled from the original image

    gc.collect()

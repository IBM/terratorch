import importlib
import os
import numpy as np

import pytest
import timm
import torch

import terratorch  # noqa: F401

from terratorch.models.decoders.aspp_head import ASPPSegmentationHead

def test_aspphead():
    dilations = (1, 6, 12, 18)
    in_channels=6
    channels=10
    decoder = ASPPSegmentationHead(dilations=dilations, in_channels=in_channels, channels=channels)

    image = torch.from_numpy(np.random.rand(2, 6, 224, 224).astype("float32"))

    assert decoder(image).shape == (2, 1, 224, 224)
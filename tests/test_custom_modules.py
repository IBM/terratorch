# Copyright contributors to the Terratorch project
import os
import sys
import warnings
import pytest
import timm
import torch
import gc 
from terratorch.models.backbones import scalemae
from terratorch.registry import BACKBONE_REGISTRY
import terratorch.models.backbones.torchgeo_vit as torchgeo_vit
from terratorch.cli_tools import build_lightning_cli

NUM_CHANNELS = 6
NUM_FRAMES = 4

@pytest.fixture
def input_224():
    return torch.ones((1, NUM_CHANNELS, 224, 224))

def test_custom_module(input_224):

    sys.path.append("examples/custom_modules")

    from alexnet import alexnet_encoder

    model = BACKBONE_REGISTRY.build("alexnet_encoder", num_channels=6)
    output = model(input_224)

@pytest.mark.parametrize("case", ["fit", "test", "validate"])
def test_custom_module_yaml(case):
    command_list = [case, "-c", f"examples/alexnet_custom_model_config.yaml"]
    _ = build_lightning_cli(command_list)

    gc.collect()



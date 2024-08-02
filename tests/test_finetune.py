import importlib
import os
import subprocess

import pytest
import timm
import torch

import terratorch
from terratorch.cli_tools import build_lightning_cli


@pytest.mark.parametrize("model_name", ["prithvi_swin_B", "prithvi_swin_L", "prithvi_vit_100", "prithvi_vit_300"])
def test_finetune_multiple_backbones(model_name, tmp_path):

    model_instance = timm.create_model(model_name)
    pretrained_bands = [0, 1, 2, 3, 4, 5]
    model_bands = [0, 1, 2, 3, 4, 5]

    state_dict = model_instance.state_dict()

    torch.save(state_dict, os.path.join(tmp_path, model_name + ".pt"))

    # Running the terratorch CLI
    command_list = ["fit", "-c", f"tests/manufactured-finetune_{model_name}.yaml"]
    _ = build_lightning_cli(command_list)

@pytest.mark.parametrize("model_name", ["prithvi_swin_B"])
def test_finetune_bands_intervals(model_name, tmp_path):

    model_instance = timm.create_model(model_name)

    state_dict = model_instance.state_dict()

    torch.save(state_dict, os.path.join(tmp_path, model_name + ".pt"))

    # Running the terratorch CLI
    command_list = ["fit", "-c", f"tests/manufactured-finetune_{model_name}_band_interval.yaml"]
    _ = build_lightning_cli(command_list)

@pytest.mark.parametrize("model_name", ["prithvi_swin_B"])
def test_finetune_bands_str(model_name, tmp_path):

    model_instance = timm.create_model(model_name)

    state_dict = model_instance.state_dict()

    torch.save(state_dict, os.path.join(tmp_path, model_name + ".pt"))

    # Running the terratorch CLI
    command_list = ["fit", "-c", f"tests/manufactured-finetune_{model_name}_string.yaml"]
    _ = build_lightning_cli(command_list)
    
@pytest.mark.parametrize("model_name", ["prithvi_swin_B"])
def test_finetune_bands_str(model_name, tmp_path):

    model_instance = timm.create_model(model_name)

    state_dict = model_instance.state_dict()

    torch.save(state_dict, os.path.join(tmp_path, model_name + ".pt"))

    # Running the terratorch CLI
    command_list = ["fit", "-c", f"tests/manufactured-finetune_{model_name}_metrics_from_file.yaml"]
    _ = build_lightning_cli(command_list)


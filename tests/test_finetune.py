import os
import shutil

import pytest
import timm
import torch

from terratorch.cli_tools import build_lightning_cli


@pytest.fixture(autouse=True)
def setup_and_cleanup(model_name):
    model_instance = timm.create_model(model_name)

    state_dict = model_instance.state_dict()

    torch.save(state_dict, os.path.join("tests", model_name + ".pt"))

    yield # everything after this runs after each test

    os.remove(os.path.join("tests", model_name + ".pt"))
    shutil.rmtree(os.path.join("tests", "all_ecos_random"))

@pytest.mark.parametrize("model_name", ["prithvi_swin_B", "prithvi_swin_L", "prithvi_vit_100", "prithvi_vit_300"])
def test_finetune_multiple_backbones(model_name):
    command_list = ["fit", "-c", f"tests/manufactured-finetune_{model_name}.yaml"]
    _ = build_lightning_cli(command_list)


@pytest.mark.parametrize("model_name", ["prithvi_swin_B"])
def test_finetune_bands_intervals(model_name):
    command_list = ["fit", "-c", f"tests/manufactured-finetune_{model_name}_band_interval.yaml"]
    _ = build_lightning_cli(command_list)


@pytest.mark.parametrize("model_name", ["prithvi_swin_B"])
def test_finetune_bands_str(model_name):
    command_list = ["fit", "-c", f"tests/manufactured-finetune_{model_name}_string.yaml"]
    _ = build_lightning_cli(command_list)

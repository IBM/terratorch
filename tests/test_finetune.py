import pytest
import timm
import torch
import importlib
import terratorch 
import subprocess
import os 

from terratorch.models.backbones.prithvi_vit import checkpoint_filter_fn as checkpoint_filter_fn_vit
from terratorch.models.backbones.prithvi_swin import checkpoint_filter_fn as checkpoint_filter_fn_swin

@pytest.mark.parametrize("model_name", ["prithvi_swin_B", "prithvi_swin_L", "prithvi_vit_100", "prithvi_vit_300"])
def test_finetune_multiple_backbones(model_name):

    model_instance = timm.create_model(model_name)
    pretrained_bands = [0, 1, 2, 3, 4, 5]
    model_bands = [0, 1, 2, 3, 4, 5]

    state_dict = model_instance.state_dict()

    torch.save(state_dict, os.path.join("tests/", model_name + ".pt"))

    # Running the terratorch CLI
    command_str = f"terratorch fit -c tests/manufactured-finetune_{model_name}.yaml" 

    command_out = subprocess.run(command_str, shell=True) 

    assert not command_out.returncode


    

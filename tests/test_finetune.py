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

    # Instantiating and creating a manufactured 
    # checkpoint just to test the finetuning pipeline
    if "vit" in model_name :
        module_str = "terratorch.models.backbones.prithvi_vit"
        ckpt_filter = checkpoint_filter_fn_vit
    elif "swin" in model_name:
        module_str = "terratorch.models.backbones.prithvi_swin"
        ckpt_filter = checkpoint_filter_fn_swin

    module_instance = importlib.import_module(module_str)

    model_template = getattr(module_instance, model_name)

    model_instance = model_template()

    pretrained_bands = [0, 1, 2, 3, 4, 5]
    model_bands = [0, 1, 2, 3, 4, 5]

    filtered_state_dict = ckpt_filter(model_instance.state_dict(), model_instance, pretrained_bands, model_bands)

    torch.save(filtered_state_dict, os.path.join("tests/", model_name + ".pt"))

    try:
        # Running the terratorch CLI
        command_str = f"terratorch fit -c tests/manufactured-finetune_{model_name}.yaml" 

        subprocess.run(command_str, shell=True) 
    except:
        Exception("Fine-tuning cannot be executed.")

    

import os
import shutil
import gc
import pytest
import timm
import torch

from terratorch.cli_tools import build_lightning_cli
from terratorch.registry import BACKBONE_REGISTRY

@pytest.fixture(autouse=True)
def setup_and_cleanup(model_name):
    model_instance = BACKBONE_REGISTRY.build(model_name)

    state_dict = model_instance.state_dict()

    torch.save(state_dict, os.path.join("tests", model_name + ".pt"))

    yield # everything after this runs after each test

    os.remove(os.path.join("tests", model_name + ".pt"))

    if os.path.isdir(os.path.join("tests", "all_ecos_random")):
        shutil.rmtree(os.path.join("tests", "all_ecos_random"))

# The backbones `prithvi_eo_v2_300` and `prithvi_swin_B` are already used in
# others tests and don't need be tested here again
@pytest.mark.parametrize("model_name",
                         ["terramind_v1_base", "prithvi_eo_v1_100", "prithvi_swin_L", "prithvi_eo_v2_600"])
@pytest.mark.parametrize("case", ["fit", "test", "validate",
                                  "compute_statistics"])
def test_finetune_multiple_backbones(model_name, case):

    command_list = [case, "-c", f"tests/resources/configs/manufactured-finetune_{model_name}.yaml"]
    _ = build_lightning_cli(command_list)

    gc.collect()

@pytest.mark.parametrize("model_name",
                         ["terramind_v1_base", "prithvi_eo_v1_100", "prithvi_swin_L", "prithvi_eo_v2_600"])
def test_finetune_multiple_backbones_with_prediction(model_name):

    command_list = ["fit", "-c", f"tests/resources/configs/manufactured-finetune_{model_name}.yaml"]
    _ = build_lightning_cli(command_list)

    gc.collect()

    command_list_prediction = ["predict", "-c",
                               f"tests/resources/configs/manufactured-finetune_{model_name}.yaml",
                               "--ckpt_path",
                               f"tests/all_ecos_random/version_0/checkpoints/epoch=0.ckpt",
                               "--predict_output_dir", "/tmp",
                               "--data.init_args.predict_data_root", "tests/resources/inputs"]

    _ = build_lightning_cli(command_list_prediction)

    gc.collect()

@pytest.mark.parametrize("model_name", ["prithvi_swin_B"])
@pytest.mark.parametrize("case", ["fit", "test", "validate"])
def test_finetune_bands_intervals(model_name, case):
    command_list = [case, "-c", f"tests/resources/configs/manufactured-finetune_{model_name}_band_interval.yaml"]
    _ = build_lightning_cli(command_list)

    gc.collect()

@pytest.mark.parametrize("model_name", ["prithvi_swin_B"])
@pytest.mark.parametrize("case", ["fit", "test", "validate"])
def test_finetune_bands_str(model_name, case):
    command_list = [case, "-c", f"tests/resources/configs/manufactured-finetune_{model_name}_string.yaml"]
    _ = build_lightning_cli(command_list)

    gc.collect()

@pytest.mark.parametrize("model_name", ["prithvi_eo_v2_300"])
@pytest.mark.parametrize("case", ["fit", "test", "validate"])
def test_finetune_pad(case):
    command_list = [case, "-c", f"tests/resources/configs/manufactured-finetune_prithvi_pixelwise_pad.yaml"]
    _ = build_lightning_cli(command_list)

    gc.collect()

@pytest.mark.parametrize("model_name", ["prithvi_swin_B"])
def test_finetune_metrics_from_file(model_name):

    model_instance = timm.create_model(model_name)

    state_dict = model_instance.state_dict()

    torch.save(state_dict, os.path.join("tests/resources/configs/", model_name + ".pt"))

    # Running the terratorch CLI
    command_list = ["fit", "-c", f"tests/resources/configs/manufactured-finetune_{model_name}_metrics_from_file.yaml"]
    _ = build_lightning_cli(command_list)

    gc.collect()

@pytest.mark.parametrize("model_name", ["prithvi_swin_B"])
@pytest.mark.parametrize("case", ["fit", "test", "validate"])
def test_finetune_segmentation_tiled(case, model_name):
    command_list = [case, "-c", f"tests/resources/configs/manufactured-finetune_{model_name}_segmentation.yaml"]
    _ = build_lightning_cli(command_list)

    gc.collect()




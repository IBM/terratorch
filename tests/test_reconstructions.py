
import pytest
import gc
import torch
from terratorch.cli_tools import build_lightning_cli
from terratorch.tasks import ReconstructionTask
from terratorch.registry import FULL_MODEL_REGISTRY


@pytest.mark.parametrize("model_name", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("case", ["fit", "test", "validate"])
def test_reconstruction_cli(model_name, case):
    command_list = [case, "-c", f"tests/resources/configs/manufactured-reconstruction_{model_name}.yaml"]
    _ = build_lightning_cli(command_list)

    gc.collect()


@pytest.mark.parametrize("model_name", ['prithvi_eo_v1_100_mae'])
def test_prithvi_mae_reconstruction(model_name):
    model = FULL_MODEL_REGISTRY.build(
        model_name,
        pretrained=False,
    )

    input = torch.ones((1, 6, 224, 224))
    loss, reconstruction, mask = model(input)

    assert 'loss' in loss
    assert reconstruction.shape == input.shape
    assert list(mask.shape) == [1, *reconstruction.shape[-2:]]

    gc.collect()


@pytest.mark.parametrize("model_name", ['terramind_v01_base_generate'])
def test_terramind_v01_generation(model_name):
    try:
        import diffusers
    except ImportError:
        pytest.skip("diffusers not installed")

    model = FULL_MODEL_REGISTRY.build(
        model_name,
        pretrained=False,
        modalities=['S2L2A', 'LULC'],
        output_modalities=['S1GRD', 'coords', 'captions'],
        timesteps=1,
        standardize=True,
        offset={'S2L2A': 1}
    )

    # Test kwargs inputs
    output = model(S2L2A=torch.ones((1, 12, 224, 224)), LULC=torch.ones((1, 1, 224, 224)))

    model = FULL_MODEL_REGISTRY.build(
        model_name,
        pretrained=False,
        modalities=['S2L2A', 'coords', 'captions'],
        output_modalities=['S1GRD', 'LULC', 'captions'],
        timesteps=1,
    )

    input = {
        "S2L2A": torch.ones((1, 12, 224, 224)),
        "coords": torch.ones((1, 2)),
        "captions": ["This is a test"],
    }
    output = model(input)

    gc.collect()


@pytest.mark.parametrize("model_name", ['terramind_v1_base_generate'])
def test_terramind_generation(model_name):
    try:
        import diffusers
    except ImportError:
        pytest.skip("diffusers not installed")

    model = FULL_MODEL_REGISTRY.build(
        model_name,
        pretrained=False,
        modalities=['S2L2A', 'coords'],
        output_modalities=['S1GRD', 'LULC'],
        timesteps=1,
    )

    input = {
        "S2L2A": torch.ones((1, 12, 224, 224)),
        "coords": torch.ones((1, 2)),
    }
    output = model(input)

    gc.collect()

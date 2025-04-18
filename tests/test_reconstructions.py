
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


@pytest.mark.parametrize("model_name", ["multimae_small", "multimae_base"])
def test_multi_mae_reconstruction(model_name):
    model_args = {
        "model": model_name,
        "pretrained": False,
        "input_adapters": ['S2L2A', 'S1GRD'],
        "output_adapters": ['S2L2A', 'S1GRD'],
    }

    task = ReconstructionTask(
        model_factory="FullModelFactory",
        model_args=model_args,
    )

    input = {"S2L2A": torch.ones((1, 12, 224, 224)),
             "S1GRD": torch.ones((1, 2, 224, 224)),}
    loss, reconstruction, mask = task.model(input)

    assert 'loss' in loss
    for r, m, t in zip(reconstruction.values(), mask.values(), input.values()):
        assert r.shape == t.shape
        assert list(m.shape) == [1, *r.shape[-2:]]

    gc.collect()


@pytest.mark.parametrize("model_name", ['prithvi_eo_v1_100_mae', 'prithvi_eo_v2_300_tl_mae'])
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


@pytest.mark.parametrize("model_name", ['terramind_v1_base_generate', 'terramind_v1_large_generate'])
def test_terramind_generation(model_name):
    try:
        import diffusers
    except ImportError:
        pytest.skip("diffusers not installed")

    model = FULL_MODEL_REGISTRY.build(
        model_name,
        pretrained=False,
        modalities=['S2L2A'],
        output_modalities=['S1GRD', 'LULC'],
        timesteps=1,
    )

    output = model(torch.ones((1, 12, 224, 224)))

    gc.collect()

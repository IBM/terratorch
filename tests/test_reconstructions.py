
import pytest
import gc
from terratorch.cli_tools import build_lightning_cli


@pytest.mark.parametrize("model_name", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("case", ["fit", "test", "validate"])
def test_finetune_bands_intervals(model_name, case):
    command_list = [case, "-c", f"tests/resources/configs/manufactured-reconstruction_{model_name}.yaml"]
    _ = build_lightning_cli(command_list)

    gc.collect()
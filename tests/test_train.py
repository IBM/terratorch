import pytest

from terratorch.cli_tools import build_lightning_cli

#@pytest.mark.parametrize("suffix", ["_regression_unet", "_segmentation_unet", "_upsample_regression_unet"])
#def test_train_unet_decoder(suffix):
#    # Running the terratorch CLI
#    command_list = ["fit", "-c", f"tests/resources/configs/manufactured-train{suffix}.yaml"]
#    _ = build_lightning_cli(command_list)

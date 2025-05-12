import pytest
import subprocess


@pytest.mark.parametrize(
    "model_name",
    [
        "encoderdecoder_eo_v1_100_model_factory",
        "encoderdecoder_eo_v2_300_model_factory",
        "encoderdecoder_eo_v2_600_model_factory",
        "prithvi_swinB_model_factory_config",
        "prithvi_swinL_model_factory_config",
        "smp_resnet34_model_factory_config",
        "encoderdecoder_timm_resnet34_model_factory",
    ],
)
def test_burns_fit(model_name):
    result = subprocess.run(
        ['terratorch', 'fit', '-c', f"./configs/test_{model_name}.yaml"], capture_output=True, text=True
    )

    # Print the captured output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Check the return code
    assert (
        result.returncode == 0
    ), f"Test failed with return code {result.returncode}STDOUT: {result.stdout}STDERR: {result.stderr}"

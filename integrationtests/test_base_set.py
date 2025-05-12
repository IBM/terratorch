import glob
import torch
import pytest
import requests
import os
import shutil
import re
import subprocess
import gc

from terratorch.cli_tools import LightningInferenceModel


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

    gc.collect()


def download_and_open_tiff(url, dest_path):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        pytest.fail(f"Failed to download TIFF image from URL: {url} (Status code: {response.status_code})")

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def create_deploy_config(config_path):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No such file: {config_path}")

    dir_name = os.path.dirname(config_path)
    name, ext = os.path.splitext(os.path.basename(config_path))
    deploy_config_path = os.path.join(dir_name, f"{name}_deploy{ext}")

    shutil.copyfile(config_path, deploy_config_path)

    return deploy_config_path


def update_grep_config_in_file(config_path: str, new_img_pattern: str):
    """Function to update img_grep in config

    Parameters
    ----------
    config_path : str
        Config file path
    new_img_pattern : str
        New img_grep pattern
    """

    with open(config_path, 'r') as file:
        config = file.read()

    # Find the current img_grep pattern (this assumes there is one img_grep line)
    current_img_pattern_match = re.search(r"img_grep:\s*'(.*?)'", config)

    # If the img_grep line exists, update it with the new img_pattern
    if current_img_pattern_match:
        config = re.sub(r"img_grep:\s*'.*'", f"img_grep: '{new_img_pattern}'", config)

    # Write the updated config back to the file
    with open(config_path, 'w') as file:
        file.write(config)


@pytest.fixture(scope="session")
def buildings_image(tmp_path_factory):
    url = "https://s3.waw3-2.cloudferro.com/swift/v1/geobuildings/78957_1250257_N-33-141-A-b-1-1.tif"
    temp_dir = tmp_path_factory.mktemp("data")
    local_path = temp_dir / "burnscars_image.tif"

    download_and_open_tiff(url=url, dest_path=local_path)

    return str(local_path)


@pytest.fixture(scope="session")
def burnscars_image(tmp_path_factory):
    url = " https://s3.us-east.cloud-object-storage.appdomain.cloud/geospatial-studio-example-data/examples-for-inference/park_fire_scaled.tif"
    temp_dir = tmp_path_factory.mktemp("data")
    local_path = temp_dir / "burnscars_image.tif"

    download_and_open_tiff(url=url, dest_path=local_path)

    return str(local_path)


@pytest.fixture(scope="session")
def floods_image(tmp_path_factory):
    url = 'https://s3.us-east.cloud-object-storage.appdomain.cloud/geospatial-studio-example-data/examples-for-[…]porto-allegre-floods-20240506-S2L2A.wgs84.tif'
    temp_dir = tmp_path_factory.mktemp("data")
    local_path = temp_dir / "burnscars_image.tif"

    download_and_open_tiff(url=url, dest_path=local_path)

    return str(local_path)


def run_inference(config, checkpoint, image):
    model = LightningInferenceModel.from_config(config_path=config, checkpoint_path=checkpoint)
    predictions = model.inference(image)

    return predictions


@pytest.mark.parametrize(
    "model_name",
    ["eo_v1_100", "eo_v2_300", "eo_v2_600", "smp_resnet34_model_factory", "encoderdecoder_resnet34_model_factory"],
)
def test_buildings_predict(buildings_image, model_name):
    # Models trained with an earlier terratorch version
    config_path = (
        f"/dccstor/terratorch/shared/integrationtests/testing_models/buildings_{model_name}/config_{model_name}.yaml"
    )
    checkpoint_path = f"/dccstor/terratorch/shared/integrationtests/testing_models/buildings_{model_name}/checkpoint_{model_name}.ckpt"

    preds = run_inference(config=config_path, checkpoint=checkpoint_path, image=buildings_image)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"

    gc.collect()


@pytest.mark.parametrize("model_name", ["swinb", "swinl"])
def test_burnscars_predict(burnscars_image, model_name):
    # Models trained with an earlier terratorch version
    config_path = (
        f"/dccstor/terratorch/shared/integrationtests/testing_models/burnscars_{model_name}/config_{model_name}.yaml"
    )
    checkpoint_path = f"/dccstor/terratorch/shared/integrationtests/testing_models/burnscars_{model_name}/checkpoint_{model_name}.ckpt"

    preds = run_inference(config=config_path, checkpoint=checkpoint_path, image=burnscars_image)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"

    gc.collect()


## Only run these tests after running test_finetune.py.
## Uses the recently created checkpoints to test if the
## current terratorch version runs inference successfully


@pytest.mark.parametrize("config_name", ["smp_resnet34", "enc_dec_resnet34"])
def test_current_terratorch_version_buildings_predict(config_name, buildings_image):
    # Models trained with current terratorch version
    config_path = f"/dccstor/terratorch/tmp/{config_name}/lightning_logs/version_0/config_deploy.yaml"

    pattern = os.path.join(f"/dccstor/terratorch/tmp/{config_name}/", "best-state_dict-epoch=*.ckpt")
    checkpoint_path = glob.glob(pattern)[0]

    # deploy_config_path = create_deploy_config(config_path)
    # ToDo: Remove after updating terratorch version and running fine-tune tests again.
    # update_grep_config_in_file(config_path=config_path, new_img_pattern="*.tif*")

    preds = run_inference(config=config_path, checkpoint=checkpoint_path, image=buildings_image)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"

    gc.collect()


@pytest.mark.parametrize("config_name", ["eo_v1_100", "eo_v2_300", "eo_v2_600", "swinb", "swinl"])
# Models trained with current terratorch version
def test_current_terratorch_version_burnscars_predict(config_name, burnscars_image):
    # config_path = f"configs/test_{config_name}.yaml"
    config_path = f"/dccstor/terratorch/tmp/{config_name}/lightning_logs/version_0/config_deploy.yaml"

    # ToDo: Remove after updating terratorch version and running fine-tune tests again.
    # update_grep_config_in_file(config_path=config_path, new_img_pattern="*.tif*")

    pattern = os.path.join(f"/dccstor/terratorch/tmp/{config_name}/", "best-state_dict-epoch=*.ckpt")
    checkpoint_path = glob.glob(pattern)[0]

    preds = run_inference(config=config_path, checkpoint=checkpoint_path, image=burnscars_image)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"

    gc.collect()


@pytest.mark.parametrize(
    "model_name", ["eo_v1_100", "eo_v2_300", "eo_v2_600", "swinb", "swinl", "smp_resnet34", "enc_dec_resnet34"]
)
def test_cleanup(model_name):

    # Delete all folders creating during finetuning after running inference.
    full_path = os.path.join("/dccstor/terratorch/tmp/", model_name)
    print("Attempting to delete:", full_path)
    try:
        shutil.rmtree(full_path)
        print(f"Deleted: {full_path}")
    except FileNotFoundError:
        print(f"Already deleted or missing: {full_path}")
    except PermissionError:
        print(f"Permission denied: {full_path}")
    except Exception as e:
        print(f"Error deleting {full_path}: {e}")

    assert not os.path.exists(full_path)

    gc.collect()

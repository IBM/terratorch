import pytest
import subprocess
import torch
import requests
from terratorch.cli_tools import LightningInferenceModel


def download_and_open_tiff(url, dest_path):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        pytest.fail(f"Failed to download TIFF image from URL: {url} (Status code: {response.status_code})")

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def run_inference(config, checkpoint, image):
    model = LightningInferenceModel.from_config(config_path=config, checkpoint_path=checkpoint)
    predictions = model.inference(image)

    return predictions


@pytest.fixture(scope="session")
def base_models_path():
    return "/dccstor/terratorch/shared/integrationtests/testing_models/"


@pytest.fixture(scope="session")
def burnscars_image(tmp_path_factory):
    url = " https://s3.us-east.cloud-object-storage.appdomain.cloud/geospatial-studio-example-data/examples-for-inference/park_fire_scaled.tif"
    temp_dir = tmp_path_factory.mktemp("data")
    local_path = temp_dir / "burnscars_image.tif"

    download_and_open_tiff(url=url, dest_path=local_path)

    return str(local_path)


@pytest.fixture(scope="session")
def buildings_image(tmp_path_factory):
    url = "https://s3.waw3-2.cloudferro.com/swift/v1/geobuildings/78957_1250257_N-33-141-A-b-1-1.tif"
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


def test_eo_v1_100_burns_fit():
    # Call the CLI program
    result = subprocess.run(
        ['terratorch', 'fit', '-c', 'test_encoderdecoder_eo_v1_100_model_factory.yaml'], capture_output=True, text=True
    )

    # Print the captured output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Check the return code
    assert (
        result.returncode == 0
    ), f"Test failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"



def test_eo_v1_100_buildings_predict(buildings_image, base_models_path):
    config_path = f"{base_models_path}buildings_eo_v1_100/config_eo_100.yaml"
    checkpoint_path = f"{base_models_path}buildings_eo_v1_100/checkpoint_eo_100.ckpt"

    preds = run_inference(config=config_path, checkpoint=checkpoint_path, image=buildings_image)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"


def test_eo_v2_300_burns_fit():
    # Call the CLI program
    result = subprocess.run(
        ['terratorch', 'fit', '-c', 'test_encoderdecoder_eo_v2_300_model_factory.yaml'], capture_output=True, text=True
    )

    # Print the captured output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Check the return code
    assert (
        result.returncode == 0
    ), f"Test failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"


def test_eo_v2_300_buildings_predict(buildings_image, base_models_path):
    config_path = f"{base_models_path}buildings_eo_v2_300/config_eo_300.yaml"
    checkpoint_path = f"{base_models_path}buildings_eo_v2_300/checkpoint_eo_300.ckpt"

    preds = run_inference(config=config_path, checkpoint=checkpoint_path, image=buildings_image)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"


def test_e0_v2_600_burns_fit():
    # Call the CLI program
    result = subprocess.run(
        ['terratorch', 'fit', '-c', 'test_encoderdecoder_eo_v2_600_model_factory.yaml'], capture_output=True, text=True
    )

    # Print the captured output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Check the return code
    assert (
        result.returncode == 0
    ), f"Test failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"


def test_eo_v2_600_buildings_predict(buildings_image, base_models_path):
    config_path = f"{base_models_path}buildings_eo_v2_600/config_eo_600.yaml"
    checkpoint_path = f"{base_models_path}buildings_eo_v2_600/checkpoint_eo_600.ckpt"

    preds = run_inference(config=config_path, checkpoint=checkpoint_path, image=buildings_image)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"


def test_prithvi_swinl_burns_fit():
    # Call the CLI program
    result = subprocess.run(
        ['terratorch', 'fit', '-c', 'test_prithvi_swinL_model_factory_config.yaml'], capture_output=True, text=True
    )

    # Print the captured output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Check the return code
    assert (
        result.returncode == 0
    ), f"Test failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"


def test_prithvi_swinl_burns_predict(burnscars_image, base_models_path):
    config_path = f"{base_models_path}burnscars_swinl/config_swinl.yaml"
    checkpoint_path = f"{base_models_path}burnscars_swinl/checkpoint_swinl.ckpt"

    preds = run_inference(config=config_path, checkpoint=checkpoint_path, image=burnscars_image)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"


def test_prithvi_swinb_burns_fit():
    # Call the CLI program
    result = subprocess.run(
        ['terratorch', 'fit', '-c', 'test_prithvi_swinB_model_factory_config.yaml'], capture_output=True, text=True
    )

    # Print the captured output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Check the return code
    assert (
        result.returncode == 0
    ), f"Test failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"


def test_prithvi_swinb_burns_predict(burnscars_image, base_models_path):
    config_path = f"{base_models_path}burnscars_swinb/config_swinb.yaml"
    checkpoint_path = f"{base_models_path}burnscars_swinb/checkpoint_swinb.ckpt"

    preds = run_inference(config=config_path, checkpoint=checkpoint_path, image=burnscars_image)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"


def test_resnet_enc_dec_builds_fit():
    # Call the CLI program
    result = subprocess.run(
        ['terratorch', 'fit', '-c', 'test_encoderdecoder_timm_resnet34_model_factory.yaml'],
        capture_output=True,
        text=True,
    )

    # Print the captured output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Check the return code
    assert (
        result.returncode == 0
    ), f"Test failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"


def test_resnet_enc_dec_buildings_predict(buildings_image, base_models_path):
    config_path = f"{base_models_path}buildings_encoderdecoder_resnet_model_factory/config_encdec_mf.yaml"
    checkpoint_path = f"{base_models_path}buildings_encoderdecoder_resnet_model_factory/checkpoint_encdec_mf.ckpt"

    preds = run_inference(config=config_path, checkpoint=checkpoint_path, image=buildings_image)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"


def test_resnet_smp_builds_fit():
    # Call the CLI program
    result = subprocess.run(
        ['terratorch', 'fit', '-c', 'test_smp_resnet34_model_factory_config.yaml'], capture_output=True, text=True
    )

    # Print the captured output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Check the return code
    assert (
        result.returncode == 0
    ), f"Test failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"


def test_resnet_smp_buildings_predict(buildings_image, base_models_path):
    config_path = f"{base_models_path}buildings_smp_resnet34_model_factory/config_smp_mf.yaml"
    checkpoint_path = f"{base_models_path}buildings_smp_resnet34_model_factory/checkpoint_smp_mf.ckpt"

    preds = run_inference(config=config_path, checkpoint=checkpoint_path, image=buildings_image)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"

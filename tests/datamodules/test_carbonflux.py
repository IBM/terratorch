import os
import gc 
import pandas as pd
import pytest
from utils import create_dummy_image


def create_carbon_flux_dummy_image(base_dir: str, split: str, filename: str, shape: tuple, pixel_values: list[int]) -> None:
    folder = split
    path = os.path.join(base_dir, folder, filename)
    create_dummy_image(path, shape, pixel_values)

@pytest.fixture
def dummy_carbon_flux_data(tmp_path) -> str:
    base_dir = tmp_path / "carbon_flux"
    base_dir.mkdir()
    for split in ["train", "test"]:
        (base_dir / split).mkdir(parents=True, exist_ok=True)
    metadata_file = "data_train_hls_37sites_v0_1.csv"
    metadata_path = base_dir / metadata_file
    df = pd.DataFrame({
        "Chip": ["HLS_CaseTrain.tiff", "HLS_CaseTest.tiff"],
        "T2MIN": [280.0, 281.0],
        "T2MAX": [290.0, 291.0],
        "T2MEAN": [285.0, 286.0],
        "TSMDEWMEAN": [275.0, 276.0],
        "GWETROOT": [0.5, 0.6],
        "LHLAND": [50.0, 51.0],
        "SHLAND": [50.0, 51.0],
        "SWLAND": [200.0, 201.0],
        "PARDFLAND": [20.0, 21.0],
        "PRECTOTLAND": [0.001, 0.002],
        "GPP": [10.0, 12.0]
    })
    df.to_csv(metadata_path, index=False)
    image_shape = (256, 256, 6)
    train_image = os.path.join(str(base_dir), "train", "HLS_CaseTrain.tiff")
    test_image = os.path.join(str(base_dir), "test", "HLS_CaseTest.tiff")
    create_dummy_image(train_image, image_shape, list(range(256)))
    create_dummy_image(test_image, image_shape, list(range(256)))
    return str(base_dir)

def test_carbon_flux_datamodule(dummy_carbon_flux_data):
    from terratorch.datamodules import CarbonFluxNonGeoDataModule
    from terratorch.datasets import CarbonFluxNonGeo
    batch_size = 1
    num_workers = 0
    bands = CarbonFluxNonGeo.all_band_names
    datamodule = CarbonFluxNonGeoDataModule(
        data_root=dummy_carbon_flux_data,
        batch_size=batch_size,
        num_workers=num_workers,
        bands=bands,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        predict_transform=None,
        no_data_replace=0.0001,
        use_metadata=False,
        modalities=("image", "merra_vars"),
    )
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    train_batch = next(iter(train_loader))
    assert "image" in train_batch, "Missing key 'image' in train batch"
    assert "merra_vars" in train_batch["image"], "Missing key 'merra_vars' in train batch"
    assert "mask" in train_batch, "Missing key 'mask' in train batch"
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    val_batch = next(iter(val_loader))
    assert "image" in val_batch, "Missing key 'image' in val batch"
    assert "merra_vars" in val_batch["image"], "Missing key 'merra_vars' in val batch"
    assert "mask" in val_batch, "Missing key 'mask' in val batch"
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    test_batch = next(iter(test_loader))
    assert "image" in test_batch, "Missing key 'image' in test batch"
    assert "merra_vars" in test_batch["image"], "Missing key 'merra_vars' in test batch"
    assert "mask" in test_batch, "Missing key 'mask' in test batch"
    datamodule.setup("predict")
    predict_loader = datamodule.predict_dataloader()
    predict_batch = next(iter(predict_loader))
    assert "image" in predict_batch, "Missing key 'image' in predict batch"

    gc.collect()

import os
import gc 
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from utils import create_dummy_image

from torchgeo.datasets import unbind_samples


def create_burn_intensity_dummy_image(base_dir: str, folder: str, filename: str, shape: tuple, pixel_values: list[int]) -> None:
    path = os.path.join(base_dir, folder, filename)
    create_dummy_image(path, shape, pixel_values)

@pytest.fixture
def dummy_burn_intensity_data(tmp_path) -> str:
    base_dir = tmp_path / "burn_intensity"
    base_dir.mkdir()
    csv_filename = "BS_files_raw.csv"
    csv_path = base_dir / csv_filename
    pd.DataFrame({"Case_Name": ["Case1", "Case1_val"]}).to_csv(csv_path, index=False)
    for split in ["train", "val"]:
        split_file = base_dir / f"{split}.txt"
        with open(split_file, "w") as f:
            if split == "train":
                f.write("HLS_Case1.tif\n")
            else:
                f.write("HLS_Case1_val.tif\n")
    for folder in ["pre", "during", "post"]:
        (base_dir / folder).mkdir(parents=True, exist_ok=True)
    feature_shape = (256, 256, 6)
    mask_shape = (256, 256)
    for folder in ["pre", "during", "post"]:
        create_burn_intensity_dummy_image(str(base_dir), folder, "HLS_Case1.tif", feature_shape, list(range(256)))
    create_burn_intensity_dummy_image(str(base_dir), "pre", "BS_Case1.tif", mask_shape, [0, 128, 255])
    for folder in ["pre", "during", "post"]:
        create_burn_intensity_dummy_image(str(base_dir), folder, "HLS_Case1_val.tif", feature_shape, list(range(256)))
    create_burn_intensity_dummy_image(str(base_dir), "pre", "BS_Case1_val.tif", mask_shape, [0, 128, 255])
    return str(base_dir)

def test_burn_intensity_datamodule(dummy_burn_intensity_data):
    from terratorch.datamodules import BurnIntensityNonGeoDataModule
    from terratorch.datasets import BurnIntensityNonGeo
    batch_size = 1
    num_workers = 0
    bands = BurnIntensityNonGeo.all_band_names
    datamodule = BurnIntensityNonGeoDataModule(
        data_root=dummy_burn_intensity_data,
        batch_size=batch_size,
        num_workers=num_workers,
        bands=bands,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        predict_transform=None,
    )
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    assert "image" in batch, "Missing key 'image' in train batch"
    assert "mask" in batch, "Missing key 'mask' in train batch"
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    batch = next(iter(val_loader))
    assert "image" in batch, "Missing key 'image' in val batch"
    assert "mask" in batch, "Missing key 'mask' in val batch"
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    batch = next(iter(test_loader))
    assert "image" in batch, "Missing key 'image' in test batch"
    assert "mask" in batch, "Missing key 'mask' in test batch"
    datamodule.setup("predict")
    predict_loader = datamodule.predict_dataloader()
    batch = next(iter(predict_loader))
    assert "image" in batch, "Missing key 'image' in predict batch"
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    batch = next(iter(val_loader))
    sample = unbind_samples(batch)[0]
    datamodule.plot(sample)
    plt.close()

    gc.collect()

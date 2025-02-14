import os
import pytest
import numpy as np
import rasterio
from pathlib import Path
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from torchgeo.datasets.utils import unbind_samples
from utils import create_dummy_tiff

@pytest.fixture
def dummy_sen1flood_data(tmp_path) -> str:

    root = tmp_path / "sen1floods11"
    data_dir = root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S2Hand"
    label_dir= root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand"
    split_dir= root / "v1.1" / "splits" / "flood_handlabeled"

    data_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    with open(split_dir / "flood_train_data.txt", "w") as f:
        f.write("loc1\n")
    with open(split_dir / "flood_valid_data.txt", "w") as f:
        f.write("loc1\n")
    with open(split_dir / "flood_test_data.txt", "w") as f:
        f.write("loc1\n")
    img_path = data_dir / "loc1_S2Hand.tif"
    create_dummy_tiff(str(img_path), shape=(13,64,64), pixel_values=range(256))
    lbl_path = label_dir / "loc1_LabelHand.tif"
    create_dummy_tiff(str(lbl_path), shape=(1,64,64), pixel_values=[0,1])

    return str(root)

def test_sen1floods11_datamodule(dummy_sen1flood_data):
    from terratorch.datamodules import Sen1Floods11NonGeoDataModule
    from terratorch.datasets import Sen1Floods11NonGeo

    dm = Sen1Floods11NonGeoDataModule(
        data_root=dummy_sen1flood_data,
        bands=Sen1Floods11NonGeo.all_band_names,
        batch_size=1,
        num_workers=0,
    )

    dm.setup("fit")
    train_loader = dm.train_dataloader()
    train_batch = next(iter(train_loader))
    assert "image" in train_batch, "Missing 'image' in train batch"
    assert "mask" in train_batch,  "Missing 'mask'  in train batch"
    dm.setup("validate")
    val_loader = dm.val_dataloader()
    val_batch = next(iter(val_loader))
    assert "image" in val_batch, "Missing 'image' in validation batch"
    assert "mask"  in val_batch, "Missing 'mask' in validation batch"
    dm.setup("test")
    test_loader = dm.test_dataloader()
    test_batch = next(iter(test_loader))
    assert "image" in test_batch, "Missing 'image' in test batch"
    assert "mask"  in test_batch, "Missing 'mask' in test batch"
    dm.setup("predict")
    predict_loader = dm.predict_dataloader()
    predict_batch = next(iter(predict_loader))
    assert "image" in predict_batch, "Missing 'image' in predict batch"
    dm.setup("validate")
    val_loader = dm.val_dataloader()
    val_batch = next(iter(val_loader))
    sample = unbind_samples(val_batch)[0]
    fig = dm.plot(sample)
    plt.close(fig)

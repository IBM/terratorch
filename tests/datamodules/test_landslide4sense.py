import gc
import matplotlib.pyplot as plt
import numpy as np
import pytest
from utils import create_dummy_h5

from torchgeo.datasets import unbind_samples


@pytest.fixture
def dummy_landslide_data(tmp_path) -> str:
    base_dir = tmp_path / "landsense"
    base_dir.mkdir()
    splits = {"train": "train", "val": "validation", "test": "test"}
    for _, folder in splits.items():
        (base_dir / "images" / folder).mkdir(parents=True, exist_ok=True)
        (base_dir / "annotations" / folder).mkdir(parents=True, exist_ok=True)
    image_shape = (256, 256, 14)
    mask_shape = (256, 256)
    for _, folder in splits.items():
        image_path = base_dir / "images" / folder / "image_dummy.h5"
        mask_path = base_dir / "annotations" / folder / "mask_dummy.h5"
        create_dummy_h5(str(image_path), ["img"], image_shape, label=1)
        create_dummy_h5(str(mask_path), ["mask"], mask_shape, label=1)
    return str(base_dir)

def test_landslide4sense_datamodule(dummy_landslide_data):
    from terratorch.datamodules import Landslide4SenseNonGeoDataModule
    from terratorch.datasets import Landslide4SenseNonGeo
    datamodule = Landslide4SenseNonGeoDataModule(
        data_root=dummy_landslide_data,
        bands=Landslide4SenseNonGeo.all_band_names,
    )
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    train_batch = next(iter(train_loader))
    assert "image" in train_batch, "Missing key 'image' in train batch"
    assert "mask" in train_batch, "Missing key 'mask' in train batch"
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    val_batch = next(iter(val_loader))
    assert "image" in val_batch, "Missing key 'image' in val batch"
    assert "mask" in val_batch, "Missing key 'mask' in val batch"
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    test_batch = next(iter(test_loader))
    assert "image" in test_batch, "Missing key 'image' in test batch"
    assert "mask" in test_batch, "Missing key 'mask' in test batch"
    datamodule.setup("predict")
    predict_loader = datamodule.predict_dataloader()
    predict_batch = next(iter(predict_loader))
    assert "image" in predict_batch, "Missing key 'image' in predict batch"
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    val_batch = next(iter(val_loader))
    sample = unbind_samples(val_batch)[0]
    datamodule.plot(sample)
    plt.close()
    gc.collect()

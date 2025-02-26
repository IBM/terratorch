import os
import gc 
import matplotlib.pyplot as plt
import pytest
from utils import create_dummy_image

from torchgeo.datasets import unbind_samples


@pytest.fixture
def dummy_fire_scars_data(tmp_path) -> str:
    base_dir = tmp_path / "fire_scars"
    base_dir.mkdir()
    training_dir = base_dir / "training"
    validation_dir = base_dir / "validation"
    training_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    image_shape = (256, 256, 6)
    mask_shape = (256, 256)
    train_image_path = os.path.join(str(training_dir), "dummy_merged.tif")
    train_mask_path = os.path.join(str(training_dir), "dummy.mask.tif")
    create_dummy_image(train_image_path, image_shape, list(range(256)))
    create_dummy_image(train_mask_path, mask_shape, [0, 1])
    val_image_path = os.path.join(str(validation_dir), "dummy_merged.tif")
    val_mask_path = os.path.join(str(validation_dir), "dummy.mask.tif")
    create_dummy_image(val_image_path, image_shape, list(range(256)))
    create_dummy_image(val_mask_path, mask_shape, [0, 1])
    return str(base_dir)

def test_fire_scars_datamodule(dummy_fire_scars_data):
    from terratorch.datamodules import FireScarsNonGeoDataModule
    from terratorch.datasets import FireScarsNonGeo

    batch_size = 1
    num_workers = 0
    bands = FireScarsNonGeo.all_band_names
    datamodule = FireScarsNonGeoDataModule(
        data_root=dummy_fire_scars_data,
        batch_size=batch_size,
        num_workers=num_workers,
        bands=bands,
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

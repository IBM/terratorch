import gc
import matplotlib.pyplot as plt
import pytest
from utils import create_dummy_tiff

from torchgeo.datasets import unbind_samples


@pytest.fixture
def dummy_multitemp_crop_data(tmp_path) -> str:
    data_root = tmp_path / "multitemp_crop"
    data_root.mkdir()
    train_chips = data_root / "training_chips"
    valid_chips = data_root / "validation_chips"
    train_chips.mkdir(parents=True, exist_ok=True)
    valid_chips.mkdir(parents=True, exist_ok=True)
    image_shape = (256, 256, 6)
    mask_shape = (256, 256)
    train_image_path = train_chips / "chip1_merged.tif"
    train_mask_path = train_chips / "chip1.mask.tif"
    create_dummy_tiff(str(train_image_path), image_shape, list(range(256)))
    create_dummy_tiff(str(train_mask_path), mask_shape, [0, 1])
    valid_image_path = valid_chips / "chip2_merged.tif"
    valid_mask_path = valid_chips / "chip2.mask.tif"
    create_dummy_tiff(str(valid_image_path), image_shape, list(range(256)))
    create_dummy_tiff(str(valid_mask_path), mask_shape, [0, 1])
    with open(data_root / "training_data.txt", "w") as f:
        f.write("chip1")
    with open(data_root / "validation_data.txt", "w") as f:
        f.write("chip2")
    return str(data_root)

def test_multitemp_crop_datamodule(dummy_multitemp_crop_data):
    from terratorch.datamodules import MultiTemporalCropClassificationDataModule
    from terratorch.datasets import MultiTemporalCropClassification

    batch_size = 1
    num_workers = 0
    bands = MultiTemporalCropClassification.all_band_names
    datamodule = MultiTemporalCropClassificationDataModule(
        data_root=dummy_multitemp_crop_data,
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
    assert "image" in val_batch, "Missing key 'image' in validation batch"
    assert "mask" in val_batch, "Missing key 'mask' in validation batch"
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

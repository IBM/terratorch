import matplotlib.pyplot as plt
import pytest
from utils import create_dummy_tiff

from torchgeo.datasets import unbind_samples


@pytest.fixture
def dummy_openearth_data(tmp_path) -> str:
    data_root = tmp_path / "openearth"
    data_root.mkdir()

    # Create split text files.
    for split, sample in zip(["train", "val", "test"], ["region1_001", "region2_002", "region3_003"], strict=False):
        with open(data_root / f"{split}.txt", "w") as f:
            f.write(sample + "\n")

    samples = [
        ("region1_001", (300, 300, 3), (300, 300, 1)),
        ("region2_002", (300, 300, 3), (300, 300, 1)),
        ("region3_003", (300, 300, 3), (300, 300, 1))
    ]
    for sample_name, img_shape, mask_shape in samples:
        folder_name = sample_name.rsplit("_", 1)[0]
        region_dir = data_root / folder_name
        (region_dir / "images").mkdir(parents=True, exist_ok=True)
        (region_dir / "labels").mkdir(parents=True, exist_ok=True)
        image_path = region_dir / "images" / sample_name
        label_path = region_dir / "labels" / sample_name
        create_dummy_tiff(str(image_path), img_shape, pixel_values=100, min_size=256)
        create_dummy_tiff(str(label_path), mask_shape, pixel_values=1, min_size=256)
    return str(data_root)

def test_openearthmap_datamodule(dummy_openearth_data):
    from terratorch.datamodules import OpenEarthMapNonGeoDataModule

    datamodule = OpenEarthMapNonGeoDataModule(
        data_root=str(dummy_openearth_data),
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
    fig = datamodule.plot(sample)
    plt.close(fig)

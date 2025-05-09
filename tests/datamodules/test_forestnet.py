import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from torchgeo.datasets import unbind_samples


@pytest.fixture
def dummy_forestnet_data(tmp_path) -> str:
    base_dir = tmp_path / "forestnet"
    base_dir.mkdir()
    splits_info = {
        "train": ("training", "Plantation"),
        "val": ("validation", "Smallholder agriculture"),
        "test": ("testing", "Grassland shrubland")
    }
    for split, (folder, label) in splits_info.items():
        df = pd.DataFrame({
            "example_path": [f"{folder}/sample1"],
            "merged_label": [label]
        })
        df.to_csv(base_dir / f"{split}_filtered.csv", index=False)
        sample_dir = base_dir / folder / "sample1"
        vis_dir = sample_dir / "images" / "visible"
        inf_dir = sample_dir / "images" / "infrared"
        vis_dir.mkdir(parents=True, exist_ok=True)
        inf_dir.mkdir(parents=True, exist_ok=True)
        image_shape = (256, 256, 3)
        vis_filenames = ["2020_01_01_cloud_10.png",
                         "2020_01_02_cloud_05.png",
                         "2020_01_03_cloud_15.png"]
        for fname in vis_filenames:
            path = str(vis_dir / fname)
            img_array = np.random.randint(0, 256, size=image_shape, dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(path)
        inf_filenames = ["2020_01_01_cloud_20.npy",
                         "2020_01_02_cloud_10.npy",
                         "2020_01_03_cloud_30.npy"]
        for fname in inf_filenames:
            path = str(inf_dir / fname)
            arr = np.random.randint(0, 256, size=image_shape, dtype=np.uint8)
            np.save(path, arr)
    return str(base_dir)

def test_forestnet_datamodule(dummy_forestnet_data):
    from terratorch.datamodules import ForestNetNonGeoDataModule
    from terratorch.datasets import ForestNetNonGeo

    batch_size = 1
    num_workers = 0
    label_map = ForestNetNonGeo.default_label_map
    bands = ForestNetNonGeo.all_band_names
    datamodule = ForestNetNonGeoDataModule(
        data_root=dummy_forestnet_data,
        batch_size=batch_size,
        num_workers=num_workers,
        label_map=label_map,
        bands=bands,
    )
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    train_batch = next(iter(train_loader))
    assert "image" in train_batch, "Missing key 'image' in train batch"
    assert "label" in train_batch, "Missing key 'label' in train batch"
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    val_batch = next(iter(val_loader))
    assert "image" in val_batch, "Missing key 'image' in val batch"
    assert "label" in val_batch, "Missing key 'label' in val batch"
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    test_batch = next(iter(test_loader))
    assert "image" in test_batch, "Missing key 'image' in test batch"
    assert "label" in test_batch, "Missing key 'label' in test batch"
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

import json
import gc
import matplotlib.pyplot as plt
import pytest
from utils import create_dummy_h5

from torchgeo.datasets import unbind_samples


@pytest.fixture
def dummy_mbigearth_data(tmp_path) -> str:
    from terratorch.datasets import MBigEarthNonGeo
    data_root = tmp_path / "dummy_mbigearth"
    data_root.mkdir()
    data_dir = data_root / "m-bigearthnet"
    data_dir.mkdir()
    dummy_label_map = {
        "sample_train": [0, 1, 0, 1],
        "sample_valid": [1, 0, 1, 0],
        "sample_test": [0, 0, 1, 1]
    }
    label_map_path = data_dir / "label_stats.json"
    with open(label_map_path, "w") as f:
        json.dump(dummy_label_map, f)
    partition = "default"
    partition_file = data_dir / f"{partition}_partition.json"
    dummy_partitions = {
        "train": ["sample_train"],
        "valid": ["sample_valid"],
        "test": ["sample_test"]
    }
    with open(partition_file, "w") as f:
        json.dump(dummy_partitions, f)
    keys_with_prefix = []
    for i, band in enumerate(MBigEarthNonGeo.all_band_names):
        key = f"{i:02d}_{band}"
        keys_with_prefix.append(key)
    dummy_shape = (256, 256)
    for sample_id in dummy_label_map.keys():
        file_path = data_dir / f"{sample_id}.hdf5"
        create_dummy_h5(str(file_path), keys_with_prefix, dummy_shape, label=0)
    return str(data_root)

def test_mbigearth_datamodule(dummy_mbigearth_data):
    from terratorch.datamodules import MBigEarthNonGeoDataModule
    from terratorch.datasets import MBigEarthNonGeo

    batch_size = 1
    num_workers = 0
    bands = MBigEarthNonGeo.all_band_names
    datamodule = MBigEarthNonGeoDataModule(
        data_root=dummy_mbigearth_data,
        batch_size=batch_size,
        num_workers=num_workers,
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

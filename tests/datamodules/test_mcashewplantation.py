import json
import gc
import matplotlib.pyplot as plt
import pytest
from utils import create_dummy_h5

from terratorch.datasets import MBeninSmallHolderCashewsNonGeo
from torchgeo.datasets import unbind_samples


@pytest.fixture
def dummy_benin_data(tmp_path) -> str:
    base_dir = tmp_path / "benin"
    base_dir.mkdir()
    data_dir = base_dir / "m-cashew-plant"
    data_dir.mkdir()
    partition_file = data_dir / "default_partition.json"
    dummy_partitions = {
        "train": ["sample_train"],
        "valid": ["sample_valid"],
        "test": ["sample_test"]
    }
    with open(partition_file, "w") as f:
        json.dump(dummy_partitions, f)
    bands_order = MBeninSmallHolderCashewsNonGeo.all_band_names
    keys = [f"{i:02d}_{band}" for i, band in enumerate(bands_order)]
    dummy_shape = (256, 256)
    for sample_id in ["sample_train", "sample_valid", "sample_test"]:
        file_path = data_dir / f"{sample_id}.hdf5"
        create_dummy_h5(str(file_path), keys, dummy_shape, label=1)
    return str(base_dir)

def test_benin_datamodule(dummy_benin_data):
    from terratorch.datamodules import MBeninSmallHolderCashewsNonGeoDataModule

    batch_size = 1
    num_workers = 0
    bands = MBeninSmallHolderCashewsNonGeo.all_band_names
    datamodule = MBeninSmallHolderCashewsNonGeoDataModule(
        data_root=dummy_benin_data,
        batch_size=batch_size,
        num_workers=num_workers,
        bands=bands,
    )
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    train_batch = next(iter(train_loader))
    assert "image" in train_batch, "Missing key 'image' in train batch"
    assert "mask" in train_batch, "Missing key 'label' in train batch"
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    val_batch = next(iter(val_loader))
    assert "image" in val_batch, "Missing key 'image' in validation batch"
    assert "mask" in val_batch, "Missing key 'label' in validation batch"
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    test_batch = next(iter(test_loader))
    assert "image" in test_batch, "Missing key 'image' in test batch"
    assert "mask" in test_batch, "Missing key 'label' in test batch"
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

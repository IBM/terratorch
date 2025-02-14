import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_pastis_data(tmp_path) -> str:
    data_root = tmp_path / "pastis_dummy"
    data_root.mkdir()
    df = pd.DataFrame({
        "ID_PATCH": [1, 2, 3, 4, 5],
        "Fold": [1, 2, 3, 4, 5],
        "dates-S2": [json.dumps({"0": "20180901"})] * 5
    })
    gdf = gpd.GeoDataFrame(df, geometry=[None]*5)
    meta_path = data_root / "metadata.geojson"
    gdf.to_file(meta_path, driver="GeoJSON")
    norm_dict = {}
    for f in range(1, 6):
        norm_dict[f"Fold_{f}"] = {"mean": [0.5, 0.5, 0.5], "std": [0.1, 0.1, 0.1]}
    norm_path = data_root / "NORM_S2_patch.json"
    with open(norm_path, "w") as f:
        json.dump(norm_dict, f)
    data_s2_dir = data_root / "DATA_S2"
    data_s2_dir.mkdir()
    dummy_array = np.random.rand(5, 3, 256, 256).astype(np.float32)
    for patch_id in [1, 2, 3, 4, 5]:
        file_path = data_s2_dir / f"S2_{patch_id}.npy"
        np.save(file_path, dummy_array)
    annotations_dir = data_root / "ANNOTATIONS"
    annotations_dir.mkdir()
    dummy_target = np.random.randint(0, 2, size=(1, 256, 256), dtype=np.int32)
    for patch_id in [1, 2, 3, 4, 5]:
        file_path = annotations_dir / f"TARGET_{patch_id}.npy"
        np.save(file_path, dummy_target)

    return str(data_root)

def test_pastis_datamodule(dummy_pastis_data):
    from terratorch.datamodules import PASTISDataModule

    batch_size = 1
    num_workers = 0
    dm = PASTISDataModule(
        data_root=dummy_pastis_data,
        norm=True,
        target="semantic",
        satellites=["S2"],
        batch_size=batch_size,
        num_workers=num_workers,
    )
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    train_batch = next(iter(train_loader))
    assert "image" in train_batch, "Missing key 'image' in train batch"
    assert "mask" in train_batch, "Missing key 'mask' in train batch"
    assert "dates" in train_batch, "Missing key 'dates' in train batch"
    dm.setup("validate")
    val_loader = dm.val_dataloader()
    val_batch = next(iter(val_loader))
    assert "image" in val_batch, "Missing key 'image' in validation batch"
    assert "mask" in val_batch, "Missing key 'mask' in validation batch"
    assert "dates" in val_batch, "Missing key 'dates' in validation batch"
    dm.setup("test")
    test_loader = dm.test_dataloader()
    test_batch = next(iter(test_loader))
    assert "image" in test_batch, "Missing key 'image' in test batch"
    assert "mask" in test_batch, "Missing key 'mask' in test batch"
    assert "dates" in test_batch, "Missing key 'dates' in test batch"
    dm.setup("predict")
    predict_loader = dm.predict_dataloader()
    predict_batch = next(iter(predict_loader))
    assert "image" in predict_batch, "Missing key 'image' in predict batch"

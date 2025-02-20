import os
import gc
import pandas as pd
import pytest
from torch.utils.data import DataLoader
from utils import create_dummy_image


def create_biomasters_dummy_image(base_dir: str, split: str, image_type: str, filename: str, shape: tuple, pixel_values: list[int]) -> None:
    dir_name = f"{split}_{'features' if image_type=='features' else 'agbm'}"
    path = os.path.join(base_dir, dir_name, filename)
    create_dummy_image(path, shape, pixel_values)

@pytest.fixture
def dummy_biomasters_data(tmp_path) -> str:
    base_dir = tmp_path / "biomasters"
    base_dir.mkdir()
    for split in ["train", "test"]:
        (base_dir / f"{split}_features").mkdir(parents=True)
        (base_dir / f"{split}_agbm").mkdir(parents=True)
    metadata_filename = "The_BioMassters_-_features_metadata.csv.csv"
    csv_path = base_dir / metadata_filename
    data = {
        "chip_id": [1, 1, 2, 2],
        "month": [9, 9, 10, 10],
        "satellite": ["S1", "S2", "S1", "S2"],
        "split": ["train", "train", "test", "test"],
        "filename": ["chip1_0_9_S1.tif", "chip1_0_9_S2.tif", "chip2_0_10_S1.tif", "chip2_0_10_S2.tif"],
        "corresponding_agbm": ["chip1_0_9_S1.tif", "chip1_0_9_S2.tif", "chip2_0_10_S1.tif", "chip2_0_10_S2.tif"]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    feature_shape = (256, 256, 3)
    mask_shape = (256, 256)
    create_biomasters_dummy_image(base_dir=str(base_dir), split="train", image_type="features", filename="chip1_0_9_S1.tif", shape=feature_shape, pixel_values=list(range(256)))
    create_biomasters_dummy_image(base_dir=str(base_dir), split="train", image_type="features", filename="chip1_0_9_S2.tif", shape=feature_shape, pixel_values=list(range(256)))
    create_biomasters_dummy_image(base_dir=str(base_dir), split="train", image_type="agbm", filename="chip1_0_9_S1.tif", shape=mask_shape, pixel_values=[0, 128, 255])
    create_biomasters_dummy_image(base_dir=str(base_dir), split="train", image_type="agbm", filename="chip1_0_9_S2.tif", shape=mask_shape, pixel_values=[0, 128, 255])
    create_biomasters_dummy_image(base_dir=str(base_dir), split="test", image_type="features", filename="chip2_0_10_S1.tif", shape=feature_shape, pixel_values=list(range(256)))
    create_biomasters_dummy_image(base_dir=str(base_dir), split="test", image_type="features", filename="chip2_0_10_S2.tif", shape=feature_shape, pixel_values=list(range(256)))
    create_biomasters_dummy_image(base_dir=str(base_dir), split="test", image_type="agbm", filename="chip2_0_10_S1.tif", shape=mask_shape, pixel_values=[0, 128, 255])
    create_biomasters_dummy_image(base_dir=str(base_dir), split="test", image_type="agbm", filename="chip2_0_10_S2.tif", shape=mask_shape, pixel_values=[0, 128, 255])
    return str(base_dir)

def test_biomasters_datamodule(dummy_biomasters_data):
    from terratorch.datamodules import BioMasstersNonGeoDataModule
    batch_size = 1
    num_workers = 0
    bands = {
        "S1": ["VV_Asc", "VH_Asc", "VV_Desc"],
        "S2": ["RED", "GREEN", "BLUE"]
    }
    sensors = ["S1", "S2"]
    datamodule = BioMasstersNonGeoDataModule(
        data_root=dummy_biomasters_data,
        batch_size=batch_size,
        num_workers=num_workers,
        bands=bands,
        sensors=sensors,
    )
    datamodule.setup("fit")
    train_loader: DataLoader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    assert "S1" in batch, "Key S1 not found on train_dataloader"
    assert "S2" in batch, "Key S2 not found on predict_dataloader"
    datamodule.setup("validate")
    val_loader: DataLoader = datamodule.val_dataloader()
    batch = next(iter(val_loader))
    assert "S1" in batch, "Key S1 not found on val_dataloader"
    assert "S2" in batch, "Key S2 not found on predict_dataloader"
    datamodule.setup("test")
    test_loader: DataLoader = datamodule.test_dataloader()
    batch = next(iter(test_loader))
    assert "S1" in batch, "Key S1 not found on test_dataloader"
    assert "S2" in batch, "Key S2 not found on predict_dataloader"
    datamodule.setup("predict")
    predict_loader: DataLoader = datamodule.predict_dataloader()
    batch = next(iter(predict_loader))
    assert "S1" in batch, "Key S1 not found on predict_dataloader"
    assert "S2" in batch, "Key S2 not found on predict_dataloader"

    gc.collect()

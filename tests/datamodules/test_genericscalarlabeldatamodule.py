import os
import gc
import pytest
from utils import create_dummy_tiff


@pytest.fixture
def dummy_classification_data(tmp_path):
    root = tmp_path / "cls_data"
    root.mkdir()

    def create_image_file(p, shape=(16,16,3), pixel_values=[0, 255]):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        create_dummy_tiff(str(p), shape=shape, pixel_values=pixel_values)

    for split in ["train", "val", "test", "predict"]:
        for class_label in ["class0", "class1"]:
            (root / split / class_label).mkdir(parents=True, exist_ok=True)
            sample_tif = root / split / class_label / "sample1.tif"
            create_image_file(sample_tif)

    return root

@pytest.fixture
def dummy_classification_data_float(tmp_path):
    root = tmp_path / "cls_data"
    root.mkdir()

    def create_image_file(p, shape=(16,16,3), pixel_values=[0.1, 0.2, 0.3, 0.4]):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        create_dummy_tiff(str(p), shape=shape, pixel_values=pixel_values)

    for split in ["train", "val", "test", "predict"]:
        for class_label in ["class0", "class1"]:
            (root / split / class_label).mkdir(parents=True, exist_ok=True)
            sample_tif = root / split / class_label / "sample1.tif"
            create_image_file(sample_tif)

    return root


def test_generic_non_geo_classification_datamodule(dummy_classification_data):
    from terratorch.datamodules.generic_scalar_label_data_module import GenericNonGeoClassificationDataModule
    dm = GenericNonGeoClassificationDataModule(
        batch_size=2,
        num_workers=0,
        train_data_root=dummy_classification_data / "train",
        val_data_root=dummy_classification_data / "val",
        test_data_root=dummy_classification_data / "test",
        predict_data_root=dummy_classification_data / "predict",
        means=[0, 0, 0],
        stds=[1, 1, 1],
        num_classes=2,
        drop_last=False,
    )

    dm.setup("fit")
    train_loader = dm.train_dataloader()
    train_batch = next(iter(train_loader))
    assert "image" in train_batch
    assert "label" in train_batch
    dm.setup("validate")
    val_loader = dm.val_dataloader()
    val_batch = next(iter(val_loader))
    assert "image" in val_batch
    assert "label" in val_batch
    dm.setup("test")
    test_loader = dm.test_dataloader()
    test_batch = next(iter(test_loader))
    assert "image" in test_batch
    assert "label" in test_batch
    dm.setup("predict")
    pred_loader = dm.predict_dataloader()
    pred_batch = next(iter(pred_loader))
    assert "image" in pred_batch

    gc.collect()

def test_generic_non_geo_classification_datamodule_float(dummy_classification_data_float):
    from terratorch.datamodules.generic_scalar_label_data_module import GenericNonGeoClassificationDataModule
    dm = GenericNonGeoClassificationDataModule(
        batch_size=2,
        num_workers=0,
        train_data_root=dummy_classification_data_float / "train",
        val_data_root=dummy_classification_data_float / "val",
        test_data_root=dummy_classification_data_float / "test",
        predict_data_root=dummy_classification_data_float / "predict",
        means=[0, 0, 0],
        stds=[1, 1, 1],
        num_classes=2,
        drop_last=False,
    )

    dm.setup("fit")
    train_loader = dm.train_dataloader()
    train_batch = next(iter(train_loader))
    assert "image" in train_batch
    assert "label" in train_batch
    assert train_batch["image"].max() == 0.4

    dm.setup("validate")
    val_loader = dm.val_dataloader()
    val_batch = next(iter(val_loader))
    assert "image" in val_batch
    assert "label" in val_batch
    assert train_batch["image"].max() == 0.4

    dm.setup("test")
    test_loader = dm.test_dataloader()
    test_batch = next(iter(test_loader))
    assert "image" in test_batch
    assert "label" in test_batch
    assert train_batch["image"].max() == 0.4

    dm.setup("predict")
    pred_loader = dm.predict_dataloader()
    pred_batch = next(iter(pred_loader))
    assert "image" in pred_batch
    assert train_batch["image"].max() == 0.4

    gc.collect()

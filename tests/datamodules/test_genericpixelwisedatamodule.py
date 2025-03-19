import os
import gc
import pytest
from utils import create_dummy_tiff


@pytest.fixture
def dummy_segmentation_data(tmp_path):
    root = tmp_path / "segdata"
    root.mkdir()
    train = root / "train"
    val = root / "val"
    test = root / "test"
    predict = root / "predict"
    train.mkdir(), val.mkdir(), test.mkdir(), predict.mkdir()
    train_lbl = train / "labels"
    val_lbl = val / "labels"
    test_lbl = test / "labels"
    train_lbl.mkdir(), val_lbl.mkdir(), test_lbl.mkdir()

    def create_image_file(p, shape=(16,16,3), pixel_values=[0, 255]):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        create_dummy_tiff(str(p), shape=shape, pixel_values=pixel_values)

    create_image_file(train / "sample1.tif")
    create_image_file(val / "sample1.tif")
    create_image_file(test / "sample1.tif")
    create_image_file(predict / "sample1.tif")

    def create_label_file(p, shape=(16,16), pixel_values=[0,1]):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        create_dummy_tiff(str(p), shape=shape, pixel_values=pixel_values)

    create_label_file(train_lbl / "sample1.tif")
    create_label_file(val_lbl / "sample1.tif")
    create_label_file(test_lbl / "sample1.tif")

    return root

def test_generic_non_geo_segmentation_datamodule(dummy_segmentation_data):
    from terratorch.datamodules.generic_pixel_wise_data_module import GenericNonGeoSegmentationDataModule
    dm = GenericNonGeoSegmentationDataModule(
        batch_size=2,
        num_workers=0,
        train_data_root=dummy_segmentation_data / "train",
        val_data_root=dummy_segmentation_data / "val",
        test_data_root=dummy_segmentation_data / "test",
        predict_data_root=dummy_segmentation_data / "predict",
        img_grep="*.tif",
        label_grep="*.tif",
        means=[0, 0, 0],
        stds=[1, 1, 1],
        num_classes=2,
        train_label_data_root=dummy_segmentation_data / "train" / "labels",
        val_label_data_root=dummy_segmentation_data / "val" / "labels",
        test_label_data_root=dummy_segmentation_data / "test" / "labels",
        drop_last=False,
        pin_memory=False,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert "image" in batch
    assert "mask" in batch
    dm.setup("validate")
    loader = dm.val_dataloader()
    batch = next(iter(loader))
    assert "image" in batch
    assert "mask" in batch
    dm.setup("test")
    loader = dm.test_dataloader()
    batch = next(iter(loader))
    assert "image" in batch
    assert "mask" in batch
    dm.setup("predict")
    loader = dm.predict_dataloader()
    batch = next(iter(loader))
    assert "image" in batch

@pytest.fixture
def dummy_regression_data(tmp_path):
    root = tmp_path / "regdata"
    root.mkdir()
    train = root / "train"
    val = root / "val"
    test = root / "test"
    predict = root / "predict"
    train.mkdir(), val.mkdir(), test.mkdir(), predict.mkdir()
    train_lbl = train / "labels"
    val_lbl = val / "labels"
    test_lbl = test / "labels"
    train_lbl.mkdir(), val_lbl.mkdir(), test_lbl.mkdir()

    def create_image_file(p, shape=(16,16,3), pixel_values=[0, 255]):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        create_dummy_tiff(str(p), shape=shape, pixel_values=pixel_values)

    def create_label_file(p, shape=(16,16), pixel_values=[0,100]):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        create_dummy_tiff(str(p), shape=shape, pixel_values=pixel_values)

    create_image_file(train / "sample1.tif")
    create_image_file(val / "sample1.tif")
    create_image_file(test / "sample1.tif")
    create_image_file(predict / "sample1.tif")
    create_label_file(train_lbl / "sample1.tif")
    create_label_file(val_lbl / "sample1.tif")
    create_label_file(test_lbl / "sample1.tif")

    return root

def test_generic_non_geo_pixelwise_regression_datamodule(dummy_regression_data):
    from terratorch.datamodules.generic_pixel_wise_data_module import GenericNonGeoPixelwiseRegressionDataModule
    dm = GenericNonGeoPixelwiseRegressionDataModule(
        batch_size=2,
        num_workers=0,
        train_data_root=dummy_regression_data / "train",
        val_data_root=dummy_regression_data / "val",
        test_data_root=dummy_regression_data / "test",
        predict_data_root=dummy_regression_data / "predict",
        means=[0, 0, 0],
        stds=[1, 1, 1],
        img_grep="*.tif",
        label_grep="*.tif",
        train_label_data_root=dummy_regression_data / "train" / "labels",
        val_label_data_root=dummy_regression_data / "val" / "labels",
        test_label_data_root=dummy_regression_data / "test" / "labels",
        drop_last=False,
        pin_memory=False,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert "image" in batch
    assert "mask" in batch
    dm.setup("validate")
    loader = dm.val_dataloader()
    batch = next(iter(loader))
    assert "image" in batch
    assert "mask" in batch
    dm.setup("test")
    loader = dm.test_dataloader()
    batch = next(iter(loader))
    assert "image" in batch
    assert "mask" in batch
    dm.setup("predict")
    loader = dm.predict_dataloader()
    batch = next(iter(loader))
    assert "image" in batch

    gc.collect()

from pathlib import Path
import gc
import pytest
from utils import create_dummy_tiff


@pytest.fixture
def dummy_multimodal_data(tmp_path: Path) -> Path:
    root = tmp_path / "generic_multimodal"
    root.mkdir(parents=True, exist_ok=True)
    train_data_dir = root / "train" / "mod1"
    val_data_dir   = root / "val"   / "mod1"
    test_data_dir  = root / "test"  / "mod1"
    for p in [train_data_dir, val_data_dir, test_data_dir]:
        p.mkdir(parents=True, exist_ok=True)
    train_label_dir = root / "train" / "labels"
    val_label_dir   = root / "val"   / "labels"
    test_label_dir  = root / "test"  / "labels"
    for p in [train_label_dir, val_label_dir, test_label_dir]:
        p.mkdir(parents=True, exist_ok=True)
    shape_img = (3, 16, 16)
    shape_mask = (1, 16, 16)
    pixel_values_image = [0, 50, 100, 200]
    pixel_values_label = [0, 1]

    create_dummy_tiff(
        path=str(train_data_dir / "sample1.tif"),
        shape=shape_img,
        pixel_values=pixel_values_image
    )
    create_dummy_tiff(
        path=str(val_data_dir / "sample1.tif"),
        shape=shape_img,
        pixel_values=pixel_values_image
    )
    create_dummy_tiff(
        path=str(test_data_dir / "sample1.tif"),
        shape=shape_img,
        pixel_values=pixel_values_image
    )
    create_dummy_tiff(
        path=str(train_label_dir / "sample1.tif"),
        shape=shape_mask,
        pixel_values=pixel_values_label
    )
    create_dummy_tiff(
        path=str(val_label_dir / "sample1.tif"),
        shape=shape_mask,
        pixel_values=pixel_values_label
    )
    create_dummy_tiff(
        path=str(test_label_dir / "sample1.tif"),
        shape=shape_mask,
        pixel_values=pixel_values_label
    )

    return root


def test_generic_multimodal_datamodule(dummy_multimodal_data: Path):
    from terratorch.datamodules.generic_multimodal_data_module import GenericMultiModalDataModule

    train_data_root = {"mod1": str(dummy_multimodal_data / "train" / "mod1")}
    val_data_root   = {"mod1": str(dummy_multimodal_data / "val"   / "mod1")}
    test_data_root  = {"mod1": str(dummy_multimodal_data / "test"  / "mod1")}
    predict_data_root = {"mod1": str(dummy_multimodal_data / "predict" / "mod1")}
    train_label_root = str(dummy_multimodal_data / "train" / "labels")
    val_label_root   = str(dummy_multimodal_data / "val"   / "labels")
    test_label_root  = str(dummy_multimodal_data / "test"  / "labels")
    means = {"mod1": [0.0, 0.0, 0.0]}
    stds  = {"mod1": [1.0, 1.0, 1.0]}

    dm = GenericMultiModalDataModule(
        modalities=["mod1"],
        task="segmentation",
        num_classes=2,
        batch_size=2,
        train_data_root=train_data_root,
        val_data_root=val_data_root,
        test_data_root=test_data_root,
        predict_data_root=predict_data_root,
        train_label_data_root=train_label_root,
        val_label_data_root=val_label_root,
        test_label_data_root=test_label_root,
        allow_substring_file_names=True,
        means=means,
        stds=stds,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        concat_bands=True,
    )

    dm.setup("fit")
    train_loader = dm.train_dataloader()
    train_batch = next(iter(train_loader))
    assert "image" in train_batch, "Missing key 'image' in train batch"
    assert "mask" in train_batch, "Missing key 'mask' in train batch"
    dm.setup("validate")
    val_loader = dm.val_dataloader()
    val_batch = next(iter(val_loader))
    assert "image" in val_batch, "Missing key 'image' in validation batch"
    assert "mask" in val_batch, "Missing key 'mask' in validation batch"
    dm.setup("test")
    test_loader = dm.test_dataloader()
    test_batch = next(iter(test_loader))
    assert "image" in test_batch, "Missing key 'image' in test batch"
    assert "mask" in test_batch, "Missing key 'mask' in test batch"

    gc.collect()

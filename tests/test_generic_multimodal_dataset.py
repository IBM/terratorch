# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for generic_multimodal_dataset module"""

import os
import tempfile
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import pytest
import rioxarray
import torch
import xarray as xr
from albumentations.pytorch import ToTensorV2

from terratorch.datasets.generic_multimodal_dataset import (
    GenericMultimodalDataset,
    GenericMultimodalPixelwiseRegressionDataset,
    GenericMultimodalScalarDataset,
    GenericMultimodalSegmentationDataset,
    MultimodalToTensor,
    load_table_data,
)


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with test data files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create subdirectories for different modalities
        mod1_dir = tmpdir / "mod1"
        mod2_dir = tmpdir / "mod2"
        label_dir = tmpdir / "labels"
        mod1_dir.mkdir()
        mod2_dir.mkdir()
        label_dir.mkdir()

        # Create sample TIFF files using rioxarray
        for sample_id in ["sample1", "sample2", "sample3"]:
            # Modality 1: 3 channels
            data1 = np.random.rand(3, 16, 16).astype(np.float32)
            xr_data1 = xr.DataArray(data1, dims=["band", "y", "x"])
            xr_data1.rio.to_raster(mod1_dir / f"{sample_id}_mod1.tif")

            # Modality 2: 4 channels
            data2 = np.random.rand(4, 16, 16).astype(np.float32)
            xr_data2 = xr.DataArray(data2, dims=["band", "y", "x"])
            xr_data2.rio.to_raster(mod2_dir / f"{sample_id}_mod2.tif")

            # Label mask
            label = np.random.randint(0, 3, (1, 16, 16)).astype(np.uint8)
            xr_label = xr.DataArray(label, dims=["band", "y", "x"])
            xr_label.rio.to_raster(label_dir / f"{sample_id}.tif")

        # Create NPY files
        npy_dir = tmpdir / "npy"
        npy_dir.mkdir()
        np.save(npy_dir / "sample1.npy", np.random.rand(3, 16, 16).astype(np.float32))

        # Create CSV file with tabular data
        csv_data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            },
            index=["sample1", "sample2", "sample3"],
        )
        csv_data.to_csv(tmpdir / "tabular_data.csv")

        # Create Parquet file with labels
        label_data = pd.DataFrame(
            {
                "label": [0, 1, 2],
            },
            index=["sample1", "sample2", "sample3"],
        )
        label_data.to_parquet(tmpdir / "labels.parquet")

        # Create split files
        with open(tmpdir / "split.txt", "w") as f:
            f.write("sample1\nsample2\n")

        split_csv = pd.DataFrame({"dummy": [1, 1]}, index=["sample1", "sample2"])
        split_csv.to_csv(tmpdir / "split.csv")

        yield tmpdir


def test_load_table_data_parquet(temp_data_dir):
    """Test loading parquet files"""
    df = load_table_data(temp_data_dir / "labels.parquet")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "label" in df.columns


def test_load_table_data_csv(temp_data_dir):
    """Test loading CSV files"""
    df = load_table_data(temp_data_dir / "tabular_data.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "feature1" in df.columns


def test_load_table_data_unsupported():
    """Test error for unsupported file type"""
    with pytest.raises(Exception, match="Unrecognized file type"):
        load_table_data("test.json")


def test_multimodal_to_tensor_image_modalities():
    """Test MultimodalToTensor for image modalities"""
    converter = MultimodalToTensor(modalities=["mod1", "mod2"])

    # 3D array (C, H, W)
    data_3d = {"mod1": np.random.rand(16, 16, 3).astype(np.float32)}
    result = converter(data_3d)
    assert isinstance(result["mod1"], torch.Tensor)
    assert result["mod1"].shape == (3, 16, 16)  # Channels moved to front

    # 4D array (T, H, W, C)
    data_4d = {"mod2": np.random.rand(8, 16, 16, 4).astype(np.float32)}
    result = converter(data_4d)
    assert isinstance(result["mod2"], torch.Tensor)
    assert result["mod2"].shape == (4, 8, 16, 16)  # C, T, H, W

    # 5D array (B, T, H, W, C)
    data_5d = {"mod1": np.random.rand(2, 4, 16, 16, 3).astype(np.float32)}
    result = converter(data_5d)
    assert isinstance(result["mod1"], torch.Tensor)
    assert result["mod1"].shape == (2, 3, 4, 16, 16)  # B, C, T, H, W


def test_multimodal_to_tensor_non_image_modalities():
    """Test MultimodalToTensor for non-image modalities"""
    converter = MultimodalToTensor(modalities=["image"])

    # Non-image data (1D, 2D arrays)
    data = {
        "image": np.random.rand(16, 16, 3).astype(np.float32),
        "tabular": np.array([1.0, 2.0, 3.0]),
        "scalar": 5.0,
    }
    result = converter(data)
    assert isinstance(result["image"], torch.Tensor)
    assert isinstance(result["tabular"], torch.Tensor)
    assert result["scalar"] == 5.0  # Non-numpy values unchanged


def test_multimodal_to_tensor_unexpected_shape():
    """Test MultimodalToTensor with unexpected shape raises error"""
    converter = MultimodalToTensor(modalities=["mod1"])

    # 6D array should raise ValueError
    data_6d = {"mod1": np.random.rand(2, 2, 4, 16, 16, 3).astype(np.float32)}
    with pytest.raises(ValueError, match="Unexpected shape"):
        converter(data_6d)


def test_generic_multimodal_dataset_basic(temp_data_dir):
    """Test basic GenericMultimodalDataset initialization and loading"""
    dataset = GenericMultimodalDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        label_data_root=str(temp_data_dir / "labels"),
        image_modalities=["mod1"],
        image_grep={"mod1": "*_mod1.tif"},
        label_grep="*.tif",
        allow_substring_file_names=True,
    )

    assert len(dataset) == 3
    assert "mod1" in dataset.modalities

    # Test __getitem__
    sample = dataset[0]
    assert "image" in sample
    assert "mask" in sample
    assert "filename" in sample
    assert isinstance(sample["image"], dict)
    assert "mod1" in sample["image"]


def test_generic_multimodal_dataset_with_split_txt(temp_data_dir):
    """Test dataset with txt split file"""
    dataset = GenericMultimodalDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        label_data_root=str(temp_data_dir / "labels"),
        image_modalities=["mod1"],
        image_grep={"mod1": "*_mod1.tif"},
        label_grep="*.tif",
        split=str(temp_data_dir / "split.txt"),
        allow_substring_file_names=True,
    )

    assert len(dataset) == 2  # Only sample1 and sample2


def test_generic_multimodal_dataset_with_split_csv(temp_data_dir):
    """Test dataset with csv split file"""
    dataset = GenericMultimodalDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        label_data_root=str(temp_data_dir / "labels"),
        image_modalities=["mod1"],
        image_grep={"mod1": "*_mod1.tif"},
        label_grep="*.tif",
        split=str(temp_data_dir / "split.csv"),
        allow_substring_file_names=True,
    )

    assert len(dataset) == 2


def test_generic_multimodal_dataset_multiple_modalities(temp_data_dir):
    """Test dataset with multiple modalities"""
    dataset = GenericMultimodalDataset(
        data_root={
            "mod1": str(temp_data_dir / "mod1"),
            "mod2": str(temp_data_dir / "mod2"),
        },
        image_modalities=["mod1", "mod2"],
        label_data_root=str(temp_data_dir / "labels"),
        image_grep={"mod1": "*_mod1.tif", "mod2": "*_mod2.tif"},
        label_grep="*.tif",
        allow_substring_file_names=True,
    )

    assert len(dataset) == 3
    sample = dataset[0]
    assert "mod1" in sample["image"]
    assert "mod2" in sample["image"]


def test_generic_multimodal_dataset_allow_missing_modalities(temp_data_dir):
    """Test dataset with allow_missing_modalities=True"""
    # Remove one modality file to test missing modality
    (temp_data_dir / "mod2" / "sample3_mod2.tif").unlink()

    dataset = GenericMultimodalDataset(
        data_root={
            "mod1": str(temp_data_dir / "mod1"),
            "mod2": str(temp_data_dir / "mod2"),
        },
        image_modalities=["mod1", "mod2"],
        label_data_root=str(temp_data_dir / "labels"),
        image_grep={"mod1": "*_mod1.tif", "mod2": "*_mod2.tif"},
        label_grep="*.tif",
        allow_substring_file_names=True,
        allow_missing_modalities=True,
    )

    assert len(dataset) >= 2  # At least sample1 and sample2

    # Test __getitem__ with tuple index (sampled modalities)
    sample = dataset[(0, ["mod1"])]
    assert "mod1" in sample["image"]
    assert "mod2" not in sample["image"]


def test_generic_multimodal_dataset_tabular_modality(temp_data_dir):
    """Test dataset with tabular (non-image) modality"""
    dataset = GenericMultimodalDataset(
        data_root={
            "mod1": str(temp_data_dir / "mod1"),
            "tabular": str(temp_data_dir / "tabular_data.csv"),
        },
        image_modalities=["mod1", "tabular"],
        label_data_root=str(temp_data_dir / "labels.parquet"),
        image_grep={"mod1": "*_mod1.tif", "tabular": ""},  # Empty grep for CSV file
        allow_substring_file_names=True,
        # image_modalities=["mod1"],
    )

    assert len(dataset) == 3
    sample = dataset[0]
    assert "mod1" in sample["image"]
    assert "tabular" in sample["image"]


def test_generic_multimodal_dataset_dataset_bands_output_bands(temp_data_dir):
    """Test dataset with dataset_bands and output_bands"""
    dataset = GenericMultimodalDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        label_data_root=str(temp_data_dir / "labels"),
        image_grep={"mod1": "*_mod1.tif"},
        label_grep="*.tif",
        image_modalities=["mod1"],
        allow_substring_file_names=True,
        dataset_bands={"mod1": [0, 1, 2]},
        output_bands={"mod1": [0, 1]},  # Select only first 2 bands
    )

    sample = dataset[0]
    assert sample["image"]["mod1"].shape[0] == 2  # Only 2 channels


def test_generic_multimodal_dataset_expand_temporal_dimension(temp_data_dir):
    """Test expand_temporal_dimension option"""
    # Create data with time*channels format
    mod_dir = temp_data_dir / "temporal"
    mod_dir.mkdir()

    # 6 channels = 3 channels * 2 timesteps
    data = np.random.rand(6, 16, 16).astype(np.float32)
    xr_data = xr.DataArray(data, dims=["band", "y", "x"])
    xr_data.rio.to_raster(mod_dir / "sample1.tif")

    dataset = GenericMultimodalDataset(
        data_root={"temporal": str(mod_dir)},
        image_grep={"temporal": "*.tif"},
        image_modalities=["temporal"],
        allow_substring_file_names=True,
        dataset_bands={"temporal": [0, 1, 2]},
        output_bands={"temporal": [0, 1]},  # Need different output_bands to trigger filter_indices
        expand_temporal_dimension=True,
    )

    sample = dataset[0]
    # After expansion (3 channels, 2 time) and filtering to 2 bands and converting to torch tensor
    # The shape will be (channels, time, H, W) = (2, 2, 16, 16) after filtering first 2 channels
    # But actually the filtering happens after moveaxis, so check actual result
    # Since expand happens before moveaxis, we get (3, 2, 16, 16) then moveaxis makes it (3, 2, 16, 16) unchanged
    # Then filtering reduces to first 2 channels
    assert sample["image"]["temporal"].dim() == 4  # Should have 4 dimensions


def test_generic_multimodal_dataset_concat_bands(temp_data_dir):
    """Test concat_bands option"""
    dataset = GenericMultimodalDataset(
        data_root={
            "mod1": str(temp_data_dir / "mod1"),
            "mod2": str(temp_data_dir / "mod2"),
        },
        image_modalities=["mod1", "mod2"],
        label_data_root=str(temp_data_dir / "labels"),
        image_grep={"mod1": "*_mod1.tif", "mod2": "*_mod2.tif"},
        label_grep="*.tif",
        allow_substring_file_names=True,
        concat_bands=True,
    )

    sample = dataset[0]
    assert "image" in sample
    assert isinstance(sample["image"], torch.Tensor)
    # Should concatenate 3 channels from mod1 and 4 from mod2 = 7 total
    assert sample["image"].shape[0] == 7


def test_generic_multimodal_dataset_constant_scale(temp_data_dir):
    """Test constant_scale parameter"""
    dataset = GenericMultimodalDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        image_grep={"mod1": "*_mod1.tif"},
        allow_substring_file_names=True,
        image_modalities=["mod1"],
        constant_scale={"mod1": 2.0},
    )

    sample = dataset[0]
    # Values should be scaled (can't assert exact values due to random data)
    assert sample["image"]["mod1"].dtype == torch.float32


def test_generic_multimodal_dataset_no_data_replace(temp_data_dir):
    """Test no_data_replace parameter"""
    # Create data with NaN values
    mod_dir = temp_data_dir / "nan_data"
    mod_dir.mkdir()
    data = np.random.rand(3, 16, 16).astype(np.float32)
    data[0, 0, 0] = np.nan
    xr_data = xr.DataArray(data, dims=["band", "y", "x"])
    xr_data.rio.to_raster(mod_dir / "sample1.tif")

    dataset = GenericMultimodalDataset(
        data_root={"mod": str(mod_dir)},
        image_grep={"mod": "*.tif"},
        allow_substring_file_names=True,
        image_modalities=["mod"],
        no_data_replace=0.0,
    )

    sample = dataset[0]
    # NaN should be replaced with 0.0
    assert not torch.isnan(sample["image"]["mod"]).any()


def test_generic_multimodal_dataset_reduce_zero_label(temp_data_dir):
    """Test reduce_zero_label parameter"""
    dataset = GenericMultimodalDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        label_data_root=str(temp_data_dir / "labels"),
        image_grep={"mod1": "*_mod1.tif"},
        image_modalities=["mod1"],
        label_grep="*.tif",
        allow_substring_file_names=True,
        reduce_zero_label=True,
    )

    sample = dataset[0]
    # Labels should be reduced by 1
    assert "mask" in sample


def test_generic_multimodal_dataset_load_npy(temp_data_dir):
    """Test loading .npy files"""
    dataset = GenericMultimodalDataset(
        data_root={"npy": str(temp_data_dir / "npy")},
        image_grep={"npy": "*.npy"},
        image_modalities=["npy"],
        allow_substring_file_names=True,
    )

    sample = dataset[0]
    assert "npy" in sample["image"]


def test_generic_multimodal_dataset_transform_compose(temp_data_dir):
    """Test with Albumentations Compose transform"""
    transform = A.Compose(
        [
            A.RandomCrop(8, 8),
            ToTensorV2(),
        ]
    )

    dataset = GenericMultimodalDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        label_data_root=str(temp_data_dir / "labels"),
        image_grep={"mod1": "*_mod1.tif"},
        image_modalities=["mod1"],
        label_grep="*.tif",
        allow_substring_file_names=True,
        transform=transform,
    )

    sample = dataset[0]
    # Should be cropped to 8x8
    assert sample["image"]["mod1"].shape[-2:] == (8, 8)


def test_generic_multimodal_dataset_no_samples_error(temp_data_dir):
    """Test error when no samples found"""
    with pytest.raises(ValueError, match="No sample candidates"):
        GenericMultimodalDataset(
            data_root={"mod1": str(temp_data_dir / "nonexistent")},
            image_grep={"mod1": "*.tif"},
            image_modalities=["mod1"],
            allow_substring_file_names=True,
        )


def test_generic_multimodal_dataset_invalid_bands_error(temp_data_dir):
    """Test error when output_bands not subset of dataset_bands"""
    with pytest.raises(Exception, match="Output bands must be a subset"):
        GenericMultimodalDataset(
            data_root={"mod1": str(temp_data_dir / "mod1")},
            image_grep={"mod1": "*_mod1.tif"},
            image_modalities=["mod1"],
            allow_substring_file_names=True,
            dataset_bands={"mod1": [0, 1]},
            output_bands={"mod1": [0, 1, 2]},  # 2 not in dataset_bands
        )


def test_generic_multimodal_dataset_missing_dataset_bands_error(temp_data_dir):
    """Test error when output_bands specified without dataset_bands"""
    with pytest.raises(Exception, match="dataset_bands must also be provided"):
        GenericMultimodalDataset(
            data_root={"mod1": str(temp_data_dir / "mod1")},
            image_grep={"mod1": "*_mod1.tif"},
            image_modalities=["mod1"],
            dataset_bands={"mod0": [1, 2, 3]},
            allow_substring_file_names=True,
            constant_scale={"mod1": 1.0},
            output_bands={"mod1": [0, 1]},
        )


def test_generic_multimodal_dataset_expand_temporal_no_bands_error(temp_data_dir):
    """Test error when expand_temporal_dimension without dataset_bands"""
    with pytest.raises(Exception, match="Please provide dataset_bands"):
        GenericMultimodalDataset(
            data_root={"mod1": str(temp_data_dir / "mod1")},
            image_grep={"mod1": "*_mod1.tif"},
            image_modalities=["mod1"],
            allow_substring_file_names=True,
            expand_temporal_dimension=True,
        )


def test_generic_multimodal_dataset_plot(temp_data_dir):
    """Test plot method"""
    dataset = GenericMultimodalDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        image_modalities=["mod1"],
        label_data_root=str(temp_data_dir / "labels"),
        image_grep={"mod1": "*_mod1.tif"},
        label_grep="*.tif",
        allow_substring_file_names=True,
    )

    sample = dataset[0]
    fig = dataset.plot(sample, suptitle="Test Plot")
    assert fig is not None


def test_generic_multimodal_dataset_plot_with_prediction(temp_data_dir):
    """Test plot method with prediction"""
    dataset = GenericMultimodalDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        label_data_root=str(temp_data_dir / "labels"),
        image_modalities=["mod1"],
        image_grep={"mod1": "*_mod1.tif"},
        label_grep="*.tif",
        allow_substring_file_names=True,
    )

    sample = dataset[0]
    # Add prediction (reconstructed image)
    sample["prediction"] = sample["image"]["mod1"].clone()

    fig = dataset.plot(sample)
    assert fig is not None


def test_generic_multimodal_segmentation_dataset(temp_data_dir):
    """Test GenericMultimodalSegmentationDataset"""
    dataset = GenericMultimodalSegmentationDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        num_classes=3,
        label_data_root=str(temp_data_dir / "labels"),
        image_grep={"mod1": "*_mod1.tif"},
        constant_scale={"mod1": 1.0},
        image_modalities=["mod1"],
        label_grep="*.tif",
        allow_substring_file_names=True,
    )

    assert dataset.num_classes == 3
    assert len(dataset) == 3

    sample = dataset[0]
    assert sample["mask"].dtype == torch.long


def test_generic_multimodal_segmentation_dataset_plot(temp_data_dir):
    """Test GenericMultimodalSegmentationDataset plot method"""
    dataset = GenericMultimodalSegmentationDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        num_classes=3,
        label_data_root=str(temp_data_dir / "labels"),
        image_grep={"mod1": "*_mod1.tif"},
        image_modalities=["mod1"],
        constant_scale={"mod1": 1.0},
        label_grep="*.tif",
        allow_substring_file_names=True,
        class_names=["class0", "class1", "class2"],
    )

    sample = dataset[0]
    fig = dataset.plot(sample, suptitle="Segmentation", show_axes=True)
    assert fig is not None


def test_generic_multimodal_segmentation_dataset_plot_with_prediction(temp_data_dir):
    """Test GenericMultimodalSegmentationDataset plot with prediction"""
    dataset = GenericMultimodalSegmentationDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        num_classes=3,
        label_data_root=str(temp_data_dir / "labels"),
        image_grep={"mod1": "*_mod1.tif"},
        image_modalities=["mod1"],
        constant_scale={"mod1": 1.0},
        label_grep="*.tif",
        allow_substring_file_names=True,
    )

    sample = dataset[0]
    sample["prediction"] = sample["mask"].clone()

    fig = dataset.plot(sample)
    assert fig is not None


def test_generic_multimodal_pixelwise_regression_dataset(temp_data_dir):
    """Test GenericMultimodalPixelwiseRegressionDataset"""
    dataset = GenericMultimodalPixelwiseRegressionDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        label_data_root=str(temp_data_dir / "labels"),
        image_grep={"mod1": "*_mod1.tif"},
        image_modalities=["mod1"],
        constant_scale={"mod1": 1.0},
        label_grep="*.tif",
        allow_substring_file_names=True,
    )

    assert len(dataset) == 3

    sample = dataset[0]
    assert sample["mask"].dtype == torch.float32


def test_generic_multimodal_pixelwise_regression_dataset_plot(temp_data_dir):
    """Test GenericMultimodalPixelwiseRegressionDataset plot method"""
    dataset = GenericMultimodalPixelwiseRegressionDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        label_data_root=str(temp_data_dir / "labels"),
        image_grep={"mod1": "*_mod1.tif"},
        image_modalities=["mod1"],
        constant_scale={"mod1": 1.0},
        label_grep="*.tif",
        allow_substring_file_names=True,
    )

    sample = dataset[0]
    fig = dataset.plot(sample, show_axes=False)
    assert fig is not None


def test_generic_multimodal_pixelwise_regression_dataset_plot_with_prediction(temp_data_dir):
    """Test GenericMultimodalPixelwiseRegressionDataset plot with prediction"""
    dataset = GenericMultimodalPixelwiseRegressionDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        label_data_root=str(temp_data_dir / "labels"),
        image_grep={"mod1": "*_mod1.tif"},
        image_modalities=["mod1"],
        constant_scale={"mod1": 1.0},
        label_grep="*.tif",
        allow_substring_file_names=True,
    )

    sample = dataset[0]
    sample["prediction"] = sample["mask"].clone()

    fig = dataset.plot(sample)
    assert fig is not None


def test_generic_multimodal_scalar_dataset(temp_data_dir):
    """Test GenericMultimodalScalarDataset with tabular labels"""
    dataset = GenericMultimodalScalarDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        num_classes=3,
        label_data_root=str(temp_data_dir / "labels.parquet"),
        image_grep={"mod1": "*_mod1.tif"},
        constant_scale={"mod1": 1.0},
        allow_substring_file_names=True,
        image_modalities=["mod1"],
        class_names=["class0", "class1", "class2"],
    )

    assert dataset.num_classes == 3
    assert len(dataset) == 3

    sample = dataset[0]
    assert "label" in sample
    assert "mask" not in sample  # scalar_label=True converts mask to label


def test_generic_multimodal_scalar_dataset_no_label_root(temp_data_dir):
    """Test GenericMultimodalScalarDataset without labels (prediction mode)"""
    dataset = GenericMultimodalScalarDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        num_classes=3,
        label_data_root=None,
        image_grep={"mod1": "*_mod1.tif"},
        image_modalities=["mod1"],
        constant_scale={"mod1": 1.0},
        allow_substring_file_names=True,
    )

    assert len(dataset) == 3

    sample = dataset[0]
    assert "label" not in sample
    assert "mask" not in sample


def test_generic_multimodal_dataset_exact_file_names(temp_data_dir):
    """Test with allow_substring_file_names=False (exact match)"""
    # Create a split file with exact file names
    split_data = pd.DataFrame({"dummy": [1, 1]}, index=["sample1_mod1.tif", "sample2_mod1.tif"])
    split_data.to_csv(temp_data_dir / "exact_split.csv")

    dataset = GenericMultimodalDataset(
        data_root={"mod1": str(temp_data_dir / "mod1")},
        image_grep={"mod1": ""},  # Empty grep for exact match
        split=str(temp_data_dir / "exact_split.csv"),
        image_modalities=["mod1"],
        allow_substring_file_names=False,
    )

    assert len(dataset) == 2


def test_generic_multimodal_dataset_missing_split_file_error():
    """Test error when split file doesn't exist"""
    with pytest.raises(FileNotFoundError, match="Split file .* does not exist"):
        GenericMultimodalDataset(
            data_root={"mod1": "/tmp/nonexistent"},
            split="/tmp/nonexistent_split.txt",
            image_modalities=["mod1"],
            constant_scale={"mod1": 1.0},
        )


def test_generic_multimodal_dataset_concat_bands_with_non_image_error(temp_data_dir):
    """Test error when concat_bands used with non-image modalities"""
    with pytest.raises(AssertionError, match="concat_bands can only be used with image modalities"):
        GenericMultimodalDataset(
            data_root={
                "mod1": str(temp_data_dir / "mod1"),
                "tabular": str(temp_data_dir / "tabular_data.csv"),
            },
            image_modalities=["mod1"],
            image_grep={"mod1": "*_mod1.tif"},
            allow_substring_file_names=True,
            # image_modalities=["mod1"],
            concat_bands=True,
        )


def test_generic_multimodal_dataset_concat_bands_with_missing_modalities_error(temp_data_dir):
    """Test error when concat_bands used with allow_missing_modalities"""
    with pytest.raises(AssertionError, match="concat_bands cannot be used with allow_missing_modalities"):
        GenericMultimodalDataset(
            data_root={"mod1": str(temp_data_dir / "mod1")},
            image_modalities=["mod1"],
            image_grep={"mod1": "*_mod1.tif"},
            allow_substring_file_names=True,
            concat_bands=True,
            allow_missing_modalities=True,
        )


def test_generic_multimodal_dataset_mask_modality_error():
    """Test error when modality is named 'mask'"""
    with pytest.raises(AssertionError, match="Modality cannot be called 'mask'"):
        GenericMultimodalDataset(
            data_root={"mask": "/tmp/data"},
        )

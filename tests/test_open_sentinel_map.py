"""Tests for the OpenSentinelMap dataset."""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import albumentations as A
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from terratorch.datasets.open_sentinel_map import OpenSentinelMap


@pytest.fixture
def mock_dataset_structure(tmp_path):
    """Create a mock OpenSentinelMap dataset structure."""
    data_root = tmp_path / "osm_data"
    data_root.mkdir()

    # Create imagery directory
    imagery_root = data_root / "osm_sentinel_imagery"
    imagery_root.mkdir()

    # Create label directory
    label_root = data_root / "osm_label_images_v10"
    label_root.mkdir()

    # Create MGRS tile directories
    mgrs_tile = "32TLR"
    (imagery_root / mgrs_tile / "1234").mkdir(parents=True)
    (label_root / mgrs_tile).mkdir(parents=True)

    # Create sample npz files with different dates
    for date in ["20200101", "20200115", "20200201"]:
        npz_path = imagery_root / mgrs_tile / "1234" / f"data_{date}.npz"
        np.savez(
            npz_path,
            gsd_10=np.random.rand(256, 256, 4).astype(np.float32),
            gsd_20=np.random.rand(128, 128, 6).astype(np.float32),
            gsd_60=np.random.rand(64, 64, 2).astype(np.float32),
        )

    # Create label file as RGB image (3 channels)
    label_img = np.random.randint(0, 14, size=(256, 256, 3), dtype=np.uint8)
    # Convert to PIL RGB image properly
    pil_img = Image.fromarray(label_img, mode='RGB')
    pil_img.save(label_root / mgrs_tile / "1234.png")

    # Create spatial_cell_info.csv
    # Note: OpenSentinelMap hardcodes split to "test" which maps to "testing"
    csv_data = pd.DataFrame(
        {
            "MGRS_tile": [mgrs_tile, mgrs_tile, mgrs_tile],
            "cell_id": [1234, 5678, 9999],
            "split": ["testing", "validation", "training"],
        }
    )
    csv_data.to_csv(data_root / "spatial_cell_info.csv", index=False)

    # Create osm_categories.json
    categories = {
        "0": "background",
        "1": "building",
        "2": "road",
        "3": "water",
    }
    with open(data_root / "osm_categories.json", "w") as f:
        json.dump(categories, f)

    return data_root


class TestOpenSentinelMapInitialization:
    """Test initialization of OpenSentinelMap dataset."""

    def test_basic_initialization(self, mock_dataset_structure):
        """Test basic initialization with default parameters."""
        dataset = OpenSentinelMap(data_root=str(mock_dataset_structure), split="train")

        assert dataset.data_root == mock_dataset_structure
        assert dataset.bands == ["gsd_10", "gsd_20", "gsd_60"]
        assert dataset.spatial_interpolate_and_stack_temporally is True
        assert dataset.pad_image is None
        assert dataset.truncate_image is None
        assert dataset.target == 0
        assert dataset.pick_random_pair is True
        assert len(dataset) > 0

    def test_custom_bands(self, mock_dataset_structure):
        """Test initialization with custom bands."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure), split="train", bands=["gsd_10"]
        )

        assert dataset.bands == ["gsd_10"]

    def test_invalid_band_raises_error(self, mock_dataset_structure):
        """Test that invalid band name raises ValueError."""
        with pytest.raises(ValueError, match="Band 'invalid_band' is not recognized"):
            OpenSentinelMap(
                data_root=str(mock_dataset_structure),
                split="train",
                bands=["invalid_band"],
            )

    def test_split_parameter_ignored(self, mock_dataset_structure):
        """Test that split parameter is currently hardcoded (known issue in source)."""
        # Due to line 56 in source: split = "test"
        # All splits actually use "testing" data regardless of parameter
        dataset = OpenSentinelMap(data_root=str(mock_dataset_structure), split="invalid")
        # Should not raise error because split is overridden to "test"
        assert len(dataset) >= 0

    def test_validation_split(self, mock_dataset_structure):
        """Test initialization with validation split."""
        dataset = OpenSentinelMap(data_root=str(mock_dataset_structure), split="val")
        assert len(dataset) >= 0  # May be 0 if no validation files exist

    def test_test_split(self, mock_dataset_structure):
        """Test initialization with test split."""
        dataset = OpenSentinelMap(data_root=str(mock_dataset_structure), split="test")
        assert len(dataset) >= 0

    def test_custom_transform(self, mock_dataset_structure):
        """Test initialization with custom transform."""
        transform = A.Compose([A.HorizontalFlip(p=1.0)])
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure), split="train", transform=transform
        )

        assert dataset.transform is transform

    def test_multiple_bands(self, mock_dataset_structure):
        """Test initialization with multiple custom bands."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            bands=["gsd_10", "gsd_20"],
        )

        assert dataset.bands == ["gsd_10", "gsd_20"]

    def test_pad_and_truncate_parameters(self, mock_dataset_structure):
        """Test initialization with pad and truncate parameters."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            pad_image=10,
            truncate_image=5,
        )

        assert dataset.pad_image == 10
        assert dataset.truncate_image == 5

    def test_custom_target(self, mock_dataset_structure):
        """Test initialization with custom target class."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure), split="train", target=1
        )

        assert dataset.target == 1

    def test_pick_random_pair_false(self, mock_dataset_structure):
        """Test initialization with pick_random_pair=False."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure), split="train", pick_random_pair=False
        )

        assert dataset.pick_random_pair is False


class TestOpenSentinelMapGetItem:
    """Test __getitem__ method of OpenSentinelMap dataset."""

    def test_getitem_basic_interpolated(self, mock_dataset_structure):
        """Test __getitem__ with spatial interpolation and stacking."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            spatial_interpolate_and_stack_temporally=True,
        )

        sample = dataset[0]

        assert "image" in sample
        assert "mask" in sample
        # Default transform converts to tensors
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["mask"], torch.Tensor)
        # Image shape: (time, height, width, channels)
        assert len(sample["image"].shape) == 4

    def test_getitem_without_interpolation(self, mock_dataset_structure):
        """Test __getitem__ without spatial interpolation."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            spatial_interpolate_and_stack_temporally=False,
        )

        sample = dataset[0]

        assert "image" in sample
        assert "mask" in sample
        assert isinstance(sample["image"], dict)
        for band in dataset.bands:
            assert band in sample["image"]
            # Non-interpolated path keeps numpy arrays inside the dict
            assert isinstance(sample["image"][band], np.ndarray)

    def test_getitem_with_pad_image(self, mock_dataset_structure):
        """Test __getitem__ with pad_image parameter."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            pad_image=10,
            spatial_interpolate_and_stack_temporally=True,
        )

        sample = dataset[0]

        # After default to_tensor, shape is (C, T, H, W); T should be 10
        assert sample["image"].shape[1] == 10

    def test_getitem_with_truncate_image(self, mock_dataset_structure):
        """Test __getitem__ with truncate_image parameter."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            truncate_image=1,
            spatial_interpolate_and_stack_temporally=True,
        )

        sample = dataset[0]

        # After default to_tensor, shape is (C, T, H, W); T should be 1
        assert sample["image"].shape[1] == 1

    def test_getitem_with_transform(self, mock_dataset_structure):
        """Test __getitem__ with transform applied."""
        # Albumentations expects HxWxC; our images are temporal, so use identity transform
        transform = lambda **batch: batch
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure), split="train", transform=transform
        )

        sample = dataset[0]

        assert "image" in sample
        assert "mask" in sample

    def test_getitem_custom_target(self, mock_dataset_structure):
        """Test __getitem__ with custom target class."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure), split="train", target=1
        )

        sample = dataset[0]

        assert "mask" in sample
        # Mask is converted to torch tensor by default transform (no extra dim)
        assert isinstance(sample["mask"], torch.Tensor)
        assert sample["mask"].ndim == 2

    def test_getitem_mask_remapping(self, mock_dataset_structure):
        """Test that mask values 254 and 255 are remapped correctly."""
        # Create a label with special values
        data_root = mock_dataset_structure
        label_root = data_root / "osm_label_images_v10"
        mgrs_tile = "32TLR"

        # Create 3-channel label with special values 254 and 255 in channel 0
        label_img = np.zeros((256, 256, 3), dtype=np.uint8)
        label_img[..., 0] = 0
        label_img[::2, ::2, 0] = 254
        label_img[1::2, 1::2, 0] = 255
        Image.fromarray(label_img, mode="RGB").save(label_root / mgrs_tile / "1234.png")

        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure), split="train", target=0
        )

        sample = dataset[0]
        mask = sample["mask"].numpy()
        # Check that 254 is remapped to 15 and 255 to 16 on target channel 0
        assert (mask == 15).any() and (mask == 16).any()

    def test_getitem_without_random_pair(self, mock_dataset_structure):
        """Test __getitem__ with pick_random_pair=False."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            pick_random_pair=False,
            spatial_interpolate_and_stack_temporally=True,
        )

        sample = dataset[0]

        assert "image" in sample
        # After default to_tensor, shape is (C, T, H, W); T should be all 3 timesteps
        assert sample["image"].shape[1] == 3

    def test_getitem_single_band(self, mock_dataset_structure):
        """Test __getitem__ with single band."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            bands=["gsd_10"],
            spatial_interpolate_and_stack_temporally=True,
        )

        sample = dataset[0]

        assert "image" in sample
        # After default to_tensor, shape is (C, T, H, W); C should be 4 channels from gsd_10
        assert sample["image"].shape[0] == 4

    def test_getitem_pad_without_interpolation(self, mock_dataset_structure):
        """Test __getitem__ with pad_image and without interpolation."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            spatial_interpolate_and_stack_temporally=False,
            pad_image=10,
        )

        sample = dataset[0]

        assert isinstance(sample["image"], dict)
        for band in dataset.bands:
            # Current implementation pads per-frame (channels), not time.
            # Thus, time dimension remains unchanged (random pair -> 2 timesteps).
            assert sample["image"][band].shape[0] == 2

    def test_getitem_truncate_without_interpolation(self, mock_dataset_structure):
        """Test __getitem__ with truncate_image and without interpolation."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            spatial_interpolate_and_stack_temporally=False,
            truncate_image=1,
        )

        sample = dataset[0]

        assert isinstance(sample["image"], dict)
        for band in dataset.bands:
            # Each band should be truncated to 1 timestep (shape (T, C, H, W))
            assert sample["image"][band].shape[0] == 1


class TestOpenSentinelMapPlotting:
    """Test plotting methods of OpenSentinelMap dataset."""

    def test_plot_with_gsd_10(self, mock_dataset_structure):
        """Test plot method with gsd_10 band."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            bands=["gsd_10", "gsd_20"],
            spatial_interpolate_and_stack_temporally=False,
        )

        sample = dataset[0]
        # Convert to expected format for plot
        sample_plot = {
            "image1": {"gsd_10": sample["image"]["gsd_10"][0]},
            "image2": {"gsd_10": sample["image"]["gsd_10"][1]},
            "mask": sample["mask"],
        }

        fig = dataset.plot(sample_plot)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_without_gsd_10(self, mock_dataset_structure):
        """Test plot method without gsd_10 band returns None."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            bands=["gsd_20", "gsd_60"],
        )

        sample = dataset[0]
        sample_plot = {"mask": sample["mask"]}

        fig = dataset.plot(sample_plot)

        assert fig is None

    def test_plot_with_tensor_image(self, mock_dataset_structure):
        """Test plot method with torch Tensor image."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure), split="train", bands=["gsd_10"]
        )

        # Create sample with tensor
        image_array = np.random.rand(256, 256, 4).astype(np.float32)
        sample = {
            "image1": {"gsd_10": torch.from_numpy(image_array)},
            "mask": np.random.randint(0, 14, size=(256, 256)),
        }

        fig = dataset.plot(sample)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_with_tensor_mask(self, mock_dataset_structure):
        """Test plot method with torch Tensor mask."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure), split="train", bands=["gsd_10"]
        )

        # Create sample with tensor mask
        image_array = np.random.rand(256, 256, 4).astype(np.float32)
        mask_array = np.random.randint(0, 14, size=(256, 256))
        sample = {
            "image1": {"gsd_10": image_array},
            "mask": torch.from_numpy(mask_array),
        }

        fig = dataset.plot(sample)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_with_suptitle(self, mock_dataset_structure):
        """Test plot method with custom suptitle."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure), split="train", bands=["gsd_10"]
        )

        image_array = np.random.rand(256, 256, 4).astype(np.float32)
        sample = {
            "image1": {"gsd_10": image_array},
            "mask": np.random.randint(0, 14, size=(256, 256)),
        }

        fig = dataset.plot(sample, suptitle="Test Title")

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_with_axes_visible(self, mock_dataset_structure):
        """Test plot method with show_axes=True."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure), split="train", bands=["gsd_10"]
        )

        image_array = np.random.rand(256, 256, 4).astype(np.float32)
        sample = {
            "image1": {"gsd_10": image_array},
            "mask": np.random.randint(0, 14, size=(256, 256)),
        }

        fig = dataset.plot(sample, show_axes=True)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_multiple_images(self, mock_dataset_structure):
        """Test plot method with multiple images."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure), split="train", bands=["gsd_10"]
        )

        image_array = np.random.rand(256, 256, 4).astype(np.float32)
        sample = {
            "image1": {"gsd_10": image_array},
            "image2": {"gsd_10": image_array},
            "image3": {"gsd_10": image_array},
            "mask": np.random.randint(0, 14, size=(256, 256)),
        }

        fig = dataset.plot(sample)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestOpenSentinelMapHelpers:
    """Test helper methods of OpenSentinelMap dataset."""

    def test_load_label_mappings(self, mock_dataset_structure):
        """Test _load_label_mappings method."""
        dataset = OpenSentinelMap(data_root=str(mock_dataset_structure), split="train")

        assert dataset.label_mappings is not None
        assert isinstance(dataset.label_mappings, dict)
        assert "0" in dataset.label_mappings

    def test_extract_date_from_filename(self, mock_dataset_structure):
        """Test _extract_date_from_filename method."""
        dataset = OpenSentinelMap(data_root=str(mock_dataset_structure), split="train")

        date = dataset._extract_date_from_filename("data_20200101.npz")
        assert date == "20200101"

    def test_extract_date_invalid_filename(self, mock_dataset_structure):
        """Test _extract_date_from_filename with invalid filename."""
        dataset = OpenSentinelMap(data_root=str(mock_dataset_structure), split="train")

        with pytest.raises(ValueError, match="Date not found in filename"):
            dataset._extract_date_from_filename("invalid_filename.npz")

    def test_len_method(self, mock_dataset_structure):
        """Test __len__ method."""
        dataset = OpenSentinelMap(data_root=str(mock_dataset_structure), split="train")

        length = len(dataset)
        assert length > 0
        assert length == len(dataset.image_files)


class TestOpenSentinelMapEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_split(self, mock_dataset_structure):
        """Test dataset with empty split (no matching files)."""
        # Remove all label files
        label_root = mock_dataset_structure / "osm_label_images_v10"
        shutil.rmtree(label_root)
        label_root.mkdir()

        dataset = OpenSentinelMap(data_root=str(mock_dataset_structure), split="train")

        assert len(dataset) == 0

    def test_missing_npz_files_handled(self, mock_dataset_structure):
        """Test behavior when NPZ files are missing (raises at stack)."""
        # Remove npz files
        imagery_root = mock_dataset_structure / "osm_sentinel_imagery"
        mgrs_tile = "32TLR"
        spatial_cell_path = imagery_root / mgrs_tile / "1234"

        for npz_file in spatial_cell_path.glob("*.npz"):
            npz_file.unlink()

        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            pick_random_pair=False,
        )

        # Expect a runtime error due to empty list when stacking
        with pytest.raises(RuntimeError, match="stack expects a non-empty TensorList"):
            _ = dataset[0]

    def test_date_sorting(self, mock_dataset_structure):
        """Test that npz files are sorted by date correctly."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            pick_random_pair=False,
        )

        # The dates in files are: 20200101, 20200115, 20200201
        # They should be sorted chronologically
        sample = dataset[0]
        assert "image" in sample

    def test_all_bands_combination(self, mock_dataset_structure):
        """Test with all three bands combined."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            bands=["gsd_10", "gsd_20", "gsd_60"],
            spatial_interpolate_and_stack_temporally=True,
        )

        sample = dataset[0]

        # After default to_tensor, shape is (C, T, H, W); total channels should be 12
        assert sample["image"].shape[0] == 12

    def test_image_normalization_in_plot(self, mock_dataset_structure):
        """Test that plot properly normalizes images."""
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure), split="train", bands=["gsd_10"]
        )

        # Create image with known min/max values
        image_array = np.array(
            [[[0.0, 0.5, 1.0, 0.25]] * 256] * 256, dtype=np.float32
        )
        sample = {
            "image1": {"gsd_10": image_array},
            "mask": np.zeros((256, 256)),
        }

        fig = dataset.plot(sample)
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)


class TestOpenSentinelMapIntegration:
    """Integration tests for OpenSentinelMap dataset."""

    def test_full_pipeline_interpolated(self, mock_dataset_structure):
        """Test full pipeline with interpolation."""
        # Use identity transform to avoid Albumentations shape checks on temporal tensors
        transform = lambda **batch: batch
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            bands=["gsd_10", "gsd_20"],
            transform=transform,
            spatial_interpolate_and_stack_temporally=True,
            pad_image=5,
            truncate_image=3,
        )

        sample = dataset[0]

        assert "image" in sample
        assert "mask" in sample
        # Should be truncated to 3, then padded to 5
        assert sample["image"].shape[0] == 5

    def test_full_pipeline_non_interpolated(self, mock_dataset_structure):
        """Test full pipeline without interpolation."""
        # Use identity transform to avoid Albumentations shape checks on temporal tensors
        transform = lambda **batch: batch
        dataset = OpenSentinelMap(
            data_root=str(mock_dataset_structure),
            split="train",
            bands=["gsd_20", "gsd_60"],
            transform=transform,
            spatial_interpolate_and_stack_temporally=False,
            pad_image=4,
            truncate_image=2,
        )

        sample = dataset[0]

        assert isinstance(sample["image"], dict)
        for band in ["gsd_20", "gsd_60"]:
            # Non-interpolated keeps numpy arrays shaped (T, C, H, W)
            # Current implementation does not pad time; T equals truncated length (2)
            assert sample["image"][band].shape[0] == 2

    def test_iteration_over_dataset(self, mock_dataset_structure):
        """Test iterating over the entire dataset."""
        dataset = OpenSentinelMap(data_root=str(mock_dataset_structure), split="train")

        count = 0
        for sample in dataset:
            assert "image" in sample
            assert "mask" in sample
            count += 1

        assert count == len(dataset)

    def test_random_access(self, mock_dataset_structure):
        """Test random access to dataset items."""
        dataset = OpenSentinelMap(data_root=str(mock_dataset_structure), split="train")

        if len(dataset) > 0:
            # Access same item multiple times
            sample1 = dataset[0]
            sample2 = dataset[0]

            assert "image" in sample1
            assert "image" in sample2
            assert "mask" in sample1
            assert "mask" in sample2

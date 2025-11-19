"""Comprehensive tests for PASTIS dataset to maximize coverage.

Tests cover:
- Dataset initialization with various parameters
- Semantic and instance segmentation targets
- Multiple satellite configurations (S2, S1A, S1D)
- Normalization on/off
- Class mapping
- Image truncation and padding
- Date handling and conversions
- Fold selection
- Transform application
- Plot functionality
- Error conditions
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import torch

from terratorch.datasets.pastis import PASTIS


@pytest.fixture
def dummy_pastis_data_full(tmp_path) -> str:
    """Create comprehensive dummy PASTIS dataset with all satellite types and instance annotations."""
    data_root = tmp_path / "pastis_full"
    data_root.mkdir()
    
    # Create metadata with multiple dates for S2, S1A, S1D
    df = pd.DataFrame({
        "ID_PATCH": [1, 2, 3, 4, 5],
        "Fold": [1, 1, 2, 3, 5],
        "dates-S2": [
            json.dumps({"0": "20180701", "1": "20180801", "2": "20180901"}),
            json.dumps({"0": "20180715"}),
            json.dumps({"0": "20180901", "1": "20181001"}),
            json.dumps({"0": "20180901"}),
            json.dumps({"0": "20180601", "1": "20180901", "2": "20181201"}),
        ],
        "dates-S1A": [
            json.dumps({"0": "20180705", "1": "20180805"}),
            json.dumps({"0": "20180720"}),
            json.dumps({"0": "20180905"}),
            json.dumps({"0": "20180910"}),
            json.dumps({"0": "20180610", "1": "20180910"}),
        ],
        "dates-S1D": [
            json.dumps({"0": "20180710"}),
            json.dumps({"0": "20180725"}),
            json.dumps({"0": "20180910"}),
            json.dumps({"0": "20180915"}),
            json.dumps({"0": "20180615"}),
        ],
    })
    gdf = gpd.GeoDataFrame(df, geometry=[None] * 5)
    meta_path = data_root / "metadata.geojson"
    gdf.to_file(meta_path, driver="GeoJSON")
    
    # Create normalization files for all satellites
    for satellite in ["S2", "S1A", "S1D"]:
        norm_dict = {}
        for f in range(1, 6):
            if satellite == "S2":
                norm_dict[f"Fold_{f}"] = {"mean": [0.3, 0.4, 0.5], "std": [0.1, 0.15, 0.2]}
            else:
                norm_dict[f"Fold_{f}"] = {"mean": [0.2, 0.3], "std": [0.05, 0.1]}
        norm_path = data_root / f"NORM_{satellite}_patch.json"
        with open(norm_path, "w") as file:
            json.dump(norm_dict, file)
    
    # Create data directories and files
    for satellite in ["S2", "S1A", "S1D"]:
        data_dir = data_root / f"DATA_{satellite}"
        data_dir.mkdir()
        for patch_id in [1, 2, 3, 4, 5]:
            if satellite == "S2":
                # S2: (time, channels, h, w) - 3 channels
                dummy_array = np.random.rand(3, 3, 128, 128).astype(np.float32)
            else:
                # S1: (time, channels, h, w) - 2 channels
                dummy_array = np.random.rand(2, 2, 128, 128).astype(np.float32)
            file_path = data_dir / f"{satellite}_{patch_id}.npy"
            np.save(file_path, dummy_array)
    
    # Create semantic annotations
    annotations_dir = data_root / "ANNOTATIONS"
    annotations_dir.mkdir()
    for patch_id in [1, 2, 3, 4, 5]:
        # Semantic target with class labels 0-17
        dummy_target = np.random.randint(0, 18, size=(1, 128, 128), dtype=np.int32)
        file_path = annotations_dir / f"TARGET_{patch_id}.npy"
        np.save(file_path, dummy_target)
    
    # Create instance annotations
    instance_dir = data_root / "INSTANCE_ANNOTATIONS"
    instance_dir.mkdir()
    for patch_id in [1, 2, 3, 4, 5]:
        # Heatmap (centerness)
        heatmap = np.random.rand(128, 128).astype(np.float32)
        np.save(instance_dir / f"HEATMAP_{patch_id}.npy", heatmap)
        
        # Instance IDs (parcels)
        instance_ids = np.zeros((128, 128), dtype=np.int32)
        instance_ids[:40, :40] = 1
        instance_ids[40:80, 40:80] = 2
        instance_ids[80:120, 80:120] = 3
        np.save(instance_dir / f"INSTANCES_{patch_id}.npy", instance_ids)
        
        # Zones (Voronoi partitioning)
        zones = np.zeros((128, 128), dtype=np.int32)
        zones[:64, :64] = 1
        zones[:64, 64:] = 2
        zones[64:, :64] = 3
        zones[64:, 64:] = 0
        np.save(instance_dir / f"ZONES_{patch_id}.npy", zones)
    
    return str(data_root)


class TestPASTISInitialization:
    """Test dataset initialization with various parameter combinations."""
    
    def test_basic_semantic_target(self, dummy_pastis_data_full):
        """Test basic initialization with semantic target."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            norm=True,
            target="semantic",
            satellites=["S2"],
        )
        assert len(dataset) == 5
        assert dataset.target == "semantic"
        assert dataset.satellites == ["S2"]
        assert dataset.norm is not None
    
    def test_instance_target(self, dummy_pastis_data_full):
        """Test initialization with instance target."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            norm=False,
            target="instance",
            satellites=["S2"],
        )
        assert dataset.target == "instance"
        assert dataset.norm is None
    
    def test_multiple_satellites(self, dummy_pastis_data_full):
        """Test initialization with multiple satellite configurations."""
        # S2 only
        dataset = PASTIS(data_root=dummy_pastis_data_full, satellites=["S2"])
        assert dataset.satellites == ["S2"]
        
        # S2 + S1A
        dataset = PASTIS(data_root=dummy_pastis_data_full, satellites=["S2", "S1A"])
        assert set(dataset.satellites) == {"S2", "S1A"}
        
        # All satellites
        dataset = PASTIS(data_root=dummy_pastis_data_full, satellites=["S2", "S1A", "S1D"])
        assert set(dataset.satellites) == {"S2", "S1A", "S1D"}
    
    def test_fold_selection(self, dummy_pastis_data_full):
        """Test selecting specific folds."""
        # Single fold
        dataset = PASTIS(data_root=dummy_pastis_data_full, folds=[1], satellites=["S2"])
        assert len(dataset) == 2  # Patches 1 and 2 are in fold 1
        
        # Multiple folds
        dataset = PASTIS(data_root=dummy_pastis_data_full, folds=[1, 2], satellites=["S2"])
        assert len(dataset) == 3  # Patches 1, 2, 3
        
        # All folds (None)
        dataset = PASTIS(data_root=dummy_pastis_data_full, folds=None, satellites=["S2"])
        assert len(dataset) == 5
    
    def test_class_mapping(self, dummy_pastis_data_full):
        """Test custom class mapping."""
        # Map all classes to binary (background vs crop)
        class_mapping = {i: 0 if i == 0 else 1 for i in range(18)}
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            target="semantic",
            class_mapping=class_mapping,
            satellites=["S2"],
        )
        assert dataset.class_mapping is not None
    
    def test_custom_reference_date(self, dummy_pastis_data_full):
        """Test custom reference date."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            reference_date="2018-01-01",
            satellites=["S2"],
        )
        expected_date = datetime(2018, 1, 1, tzinfo=timezone.utc)
        assert dataset.reference_date == expected_date
    
    def test_custom_date_interval(self, dummy_pastis_data_full):
        """Test custom date interval."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            date_interval=(-100, 300),
            satellites=["S2"],
        )
        assert len(dataset.date_range) == 400
        assert dataset.date_range[0] == -100
        assert dataset.date_range[-1] == 299
    
    def test_invalid_target_raises_error(self, dummy_pastis_data_full):
        """Test that invalid target raises ValueError."""
        with pytest.raises(ValueError, match="Target 'invalid' not recognized"):
            PASTIS(data_root=dummy_pastis_data_full, target="invalid", satellites=["S2"])
    
    def test_invalid_satellite_raises_error(self, dummy_pastis_data_full):
        """Test that invalid satellite raises ValueError."""
        with pytest.raises(ValueError, match="Satellite 'INVALID' not recognized"):
            PASTIS(data_root=dummy_pastis_data_full, satellites=["INVALID"])


class TestPASTISGetItem:
    """Test __getitem__ functionality with various configurations."""
    
    def test_semantic_target_getitem(self, dummy_pastis_data_full):
        """Test getting item with semantic target."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            norm=True,
            target="semantic",
            satellites=["S2"],
        )
        sample = dataset[0]
        
        assert "image" in sample
        assert "mask" in sample
        assert "dates" in sample
        assert "S2" in sample
        
        # Check shapes
        assert sample["image"].shape[0] == 3  # timesteps
        assert sample["image"].shape[-1] == 3  # channels (transposed)
        assert sample["mask"].shape == (128, 128)
        assert "S2" in sample["dates"]
    
    def test_instance_target_getitem(self, dummy_pastis_data_full):
        """Test getting item with instance target."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            norm=False,
            target="instance",
            satellites=["S2"],
        )
        sample = dataset[0]
        
        assert "mask" in sample
        # Instance target has 7 channels:
        # heatmap, instance_ids, pixel_to_object_mapping, size(h,w), object_semantic, pixel_semantic
        assert sample["mask"].shape[-1] == 7
        assert sample["mask"].dtype == np.float32
    
    def test_multiple_satellites_getitem(self, dummy_pastis_data_full):
        """Test getting item with multiple satellites."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            satellites=["S2", "S1A", "S1D"],
        )
        sample = dataset[0]
        
        assert "S2" in sample
        assert "S1A" in sample
        assert "S1D" in sample
        assert "S2" in sample["dates"]
        assert "S1A" in sample["dates"]
        assert "S1D" in sample["dates"]
    
    def test_normalization_applied(self, dummy_pastis_data_full):
        """Test that normalization is applied correctly."""
        dataset_norm = PASTIS(
            data_root=dummy_pastis_data_full,
            norm=True,
            satellites=["S2"],
        )
        dataset_no_norm = PASTIS(
            data_root=dummy_pastis_data_full,
            norm=False,
            satellites=["S2"],
        )
        
        sample_norm = dataset_norm[0]
        sample_no_norm = dataset_no_norm[0]
        
        # Normalized data should have different values
        assert not np.allclose(sample_norm["S2"], sample_no_norm["S2"])
    
    def test_class_mapping_applied(self, dummy_pastis_data_full):
        """Test that class mapping is applied to semantic target."""
        class_mapping = {i: 0 if i < 5 else 1 for i in range(18)}
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            target="semantic",
            class_mapping=class_mapping,
            satellites=["S2"],
        )
        sample = dataset[0]
        
        # All values should be 0 or 1 after mapping
        unique_vals = np.unique(sample["mask"])
        assert all(val in [0, 1] for val in unique_vals)
    
    def test_truncate_image(self, dummy_pastis_data_full):
        """Test image truncation."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            truncate_image=2,
            satellites=["S2"],
        )
        sample = dataset[0]
        
        # Should truncate to last 2 timesteps
        assert sample["S2"].shape[0] == 2
        assert len(sample["dates"]["S2"]) == 2
    
    def test_pad_image(self, dummy_pastis_data_full):
        """Test image padding."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            pad_image=10,
            satellites=["S2"],
        )
        sample = dataset[1]  # Patch 2 has only 1 timestep
        
        # Should pad to 10 timesteps
        assert sample["S2"].shape[0] == 10
        assert len(sample["dates"]["S2"]) == 10
    
    def test_truncate_when_larger(self, dummy_pastis_data_full):
        """Test that truncation only happens when data is larger."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            truncate_image=5,  # Data has 3 timesteps, so no truncation
            satellites=["S2"],
        )
        sample = dataset[0]
        
        # Should keep original 3 timesteps
        assert sample["S2"].shape[0] == 3
    
    def test_pad_when_smaller(self, dummy_pastis_data_full):
        """Test that padding only happens when data is smaller."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            pad_image=2,  # Data has 3 timesteps, so no padding
            satellites=["S2"],
        )
        sample = dataset[0]
        
        # Should keep original 3 timesteps
        assert sample["S2"].shape[0] == 3
    
    def test_transform_applied(self, dummy_pastis_data_full):
        """Test that transform is applied to sample."""
        def custom_transform(**kwargs):
            # Modify image
            if "image" in kwargs:
                kwargs["image"] = kwargs["image"] * 2
            kwargs["custom_key"] = "custom_value"
            return kwargs
        
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            transform=custom_transform,
            satellites=["S2"],
        )
        sample = dataset[0]
        
        assert "custom_key" in sample
        assert sample["custom_key"] == "custom_value"
    
    def test_instance_annotations_with_zero_instance(self, dummy_pastis_data_full):
        """Test instance target processing with zero instance IDs."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            target="instance",
            satellites=["S2"],
        )
        sample = dataset[0]
        
        # The instance processing should handle instance_id == 0 correctly
        mask = sample["mask"]
        assert mask.shape[-1] == 7
        
        # Check that size and semantic annotations are computed
        # For non-zero instances
        assert mask[..., 3:5].max() > 0  # Size should be non-zero for instances


class TestPASTISDateHandling:
    """Test date handling and conversion."""
    
    def test_get_dates(self, dummy_pastis_data_full):
        """Test get_dates method."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            reference_date="2018-09-01",
            satellites=["S2"],
        )
        
        # Get dates for patch 1 (ID_PATCH=1)
        dates = dataset.get_dates(1, "S2")
        
        # Should return dates relative to reference date
        assert isinstance(dates, np.ndarray)
        assert len(dates) > 0
    
    def test_date_conversion_from_string(self, dummy_pastis_data_full):
        """Test that string dates in metadata are converted correctly."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            reference_date="2018-09-01",
            satellites=["S2"],
        )
        
        sample = dataset[0]
        dates = sample["dates"]["S2"]
        
        # Dates should be torch tensors
        assert isinstance(dates, torch.Tensor)
        
        # Dates should be relative to reference date (in days)
        assert dates.dtype == torch.int64 or dates.dtype == torch.long


class TestPASTISPlotting:
    """Test plot functionality."""
    
    def test_plot_with_s2_data(self, dummy_pastis_data_full):
        """Test plotting with S2 RGB data."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            norm=False,
            satellites=["S2"],
            pad_image=6,  # Ensure multiple rows for 2D subplot grid
        )
        sample = dataset[0]
        
        # Convert S2 to torch tensor for plot
        sample["S2"] = torch.from_numpy(sample["S2"])
        sample["target"] = torch.from_numpy(sample["mask"])
        
        fig = dataset.plot(sample, suptitle="Test Plot", show_axes=True)
        
        assert fig is not None
    
    def test_plot_without_s2_data(self, dummy_pastis_data_full):
        """Test plotting without S2 data returns None with warning."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            satellites=["S2"],
        )
        sample = dataset[0]
        
        # Remove S2 from sample
        sample_no_s2 = {k: v for k, v in sample.items() if k != "S2"}
        sample_no_s2["target"] = torch.from_numpy(sample["mask"])
        
        with pytest.warns(UserWarning, match="No RGB image"):
            result = dataset.plot(sample_no_s2)
            assert result is None
    
    def test_plot_with_target(self, dummy_pastis_data_full):
        """Test plotting with target mask."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            target="semantic",
            satellites=["S2"],
            pad_image=6,  # Ensure multiple rows for 2D subplot grid
        )
        sample = dataset[0]
        
        # Convert to torch for plot function
        sample["S2"] = torch.from_numpy(sample["S2"])
        sample["target"] = torch.from_numpy(sample["mask"])
        
        fig = dataset.plot(sample, show_axes=False)
        assert fig is not None
    
    def test_plot_normalization(self, dummy_pastis_data_full):
        """Test that RGB normalization in plot handles edge cases."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            satellites=["S2"],
            pad_image=6,  # Ensure multiple rows for 2D subplot grid
        )
        sample = dataset[0]
        
        # Create sample with constant RGB values (denom == 0 case)
        s2_data = sample["S2"]
        s2_data[:, :, :, 0] = 0.5  # Set all R channel to same value
        sample["S2"] = torch.from_numpy(s2_data)
        sample["target"] = torch.from_numpy(sample["mask"])
        
        # Should handle the case where denom == 0
        fig = dataset.plot(sample)
        assert fig is not None
    
    def test_plot_with_target_none(self, dummy_pastis_data_full):
        """Test plotting when target is None."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            satellites=["S2"],
            pad_image=6,  # Ensure multiple rows for 2D subplot grid
        )
        sample = dataset[0]
        
        sample["S2"] = torch.from_numpy(sample["S2"])
        sample["target"] = None
        
        fig = dataset.plot(sample)
        assert fig is not None
    
    def test_plot_with_more_images_than_cols(self, dummy_pastis_data_full):
        """Test plotting with many timesteps to check grid layout."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            satellites=["S2"],
            pad_image=10,  # Create 10 timesteps
        )
        sample = dataset[0]
        
        sample["S2"] = torch.from_numpy(sample["S2"])
        sample["target"] = torch.from_numpy(sample["mask"])
        
        fig = dataset.plot(sample)
        assert fig is not None


class TestPASTISEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_len_method(self, dummy_pastis_data_full):
        """Test __len__ returns correct count."""
        dataset = PASTIS(data_root=dummy_pastis_data_full, folds=[1], satellites=["S2"])
        assert len(dataset) == 2
        
        dataset_all = PASTIS(data_root=dummy_pastis_data_full, satellites=["S2"])
        assert len(dataset_all) == 5
    
    def test_normalization_with_selected_folds(self, dummy_pastis_data_full):
        """Test that normalization uses correct fold statistics."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            norm=True,
            folds=[1, 2],
            satellites=["S2"],
        )
        
        # Normalization should average stats from folds 1 and 2
        assert "S2" in dataset.norm
        assert len(dataset.norm["S2"]) == 2  # (mean, std)
    
    def test_multiple_satellites_different_channels(self, dummy_pastis_data_full):
        """Test handling satellites with different channel counts."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            norm=True,
            satellites=["S2", "S1A"],
        )
        
        sample = dataset[0]
        
        # S2 has 3 channels, S1A has 2 channels
        assert sample["S2"].shape[1] == 3
        assert sample["S1A"].shape[1] == 2
    
    def test_date_table_construction(self, dummy_pastis_data_full):
        """Test that date tables are constructed correctly for all satellites."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            satellites=["S2", "S1A", "S1D"],
        )
        
        assert "S2" in dataset.date_tables
        assert "S1A" in dataset.date_tables
        assert "S1D" in dataset.date_tables
        
        # Each patch should have date table entry
        for patch_id in [1, 2, 3, 4, 5]:
            assert patch_id in dataset.date_tables["S2"]
    
    def test_instance_semantic_annotation_consistency(self, dummy_pastis_data_full):
        """Test that instance target includes correct semantic annotations."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            target="instance",
            satellites=["S2"],
        )
        
        sample = dataset[0]
        mask = sample["mask"]
        
        # Check channel structure:
        # 0: heatmap, 1: instance_ids, 2: zones, 3-4: size, 5: object_semantic, 6: pixel_semantic
        assert mask.shape[-1] == 7
        
        # Pixel semantic should match the semantic target
        pixel_semantic = mask[..., 6]
        assert pixel_semantic.min() >= 0
        assert pixel_semantic.max() < 18
    
    def test_class_mapping_with_instance_target(self, dummy_pastis_data_full):
        """Test class mapping is applied to instance target annotations."""
        class_mapping = {i: i // 2 for i in range(18)}  # Group classes
        
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            target="instance",
            class_mapping=class_mapping,
            satellites=["S2"],
        )
        
        sample = dataset[0]
        mask = sample["mask"]
        
        # Pixel and object semantic annotations should use mapped classes
        pixel_semantic = mask[..., 6]
        object_semantic = mask[..., 5]
        
        # Check that mapping was applied
        unique_pixel = np.unique(pixel_semantic)
        unique_object = np.unique(object_semantic)
        
        assert len(unique_pixel) <= 9  # At most 9 mapped classes (18 // 2)


class TestPASTISIntegration:
    """Integration tests with realistic workflows."""
    
    def test_full_pipeline_semantic(self, dummy_pastis_data_full):
        """Test full pipeline for semantic segmentation."""
        def augmentation(**kwargs):
            # Simple augmentation
            if "image" in kwargs:
                kwargs["image"] = kwargs["image"] + 0.01
            return kwargs
        
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            norm=True,
            target="semantic",
            folds=[1, 2],
            reference_date="2018-09-01",
            class_mapping={i: i % 10 for i in range(18)},
            transform=augmentation,
            truncate_image=5,
            pad_image=6,
            satellites=["S2"],
        )
        
        assert len(dataset) == 3
        sample = dataset[0]
        
        # Check all expected keys
        assert "image" in sample
        assert "mask" in sample
        assert "dates" in sample
        assert "S2" in sample
        
        # Check padding was applied
        assert sample["S2"].shape[0] == 6
    
    def test_full_pipeline_instance(self, dummy_pastis_data_full):
        """Test full pipeline for instance segmentation."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            norm=False,
            target="instance",
            folds=[1],
            satellites=["S2", "S1A"],
            truncate_image=3,
        )
        
        assert len(dataset) == 2
        sample = dataset[0]
        
        # Check instance-specific outputs
        assert sample["mask"].shape[-1] == 7
        assert "S2" in sample
        assert "S1A" in sample
    
    def test_iteration_over_dataset(self, dummy_pastis_data_full):
        """Test iterating over the full dataset."""
        dataset = PASTIS(
            data_root=dummy_pastis_data_full,
            satellites=["S2"],
        )
        
        samples = []
        for i in range(len(dataset)):
            sample = dataset[i]
            samples.append(sample)
        
        assert len(samples) == 5
        
        # Check all samples have required keys
        for sample in samples:
            assert "image" in sample
            assert "mask" in sample
            assert "dates" in sample

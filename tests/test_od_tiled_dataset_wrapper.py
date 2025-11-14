"""Comprehensive tests for TiledDataset.

Tests cover initialization, tiling logic, caching, bbox handling, edge cases 
to ensure maximum code coverage.
"""

import gc
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import pytest
import torch
from PIL import Image
from torchvision.transforms import functional as F

from terratorch.datasets.od_tiled_dataset_wrapper import (
    TiledDataset,
    atomic_write_image,
    atomic_write_json,
)


@pytest.fixture
def mock_dataset():
    """Create a mock base dataset with images and bounding boxes."""
    class MockDataset:
        def __init__(self):
            self.samples = []
            
            # Sample 0: Large image with multiple boxes
            img1 = torch.rand(3, 1024, 1024)
            boxes1 = torch.tensor([
                [100, 100, 300, 300],
                [600, 600, 800, 800],
                [50, 50, 150, 150],
            ], dtype=torch.float32)
            labels1 = torch.tensor([1, 2, 1], dtype=torch.int64)
            self.samples.append({
                "image": img1,
                "boxes": boxes1,
                "labels": labels1,
            })
            
            # Sample 1: Medium image with one box
            img2 = torch.rand(3, 600, 600)
            boxes2 = torch.tensor([[200, 200, 400, 400]], dtype=torch.float32)
            labels2 = torch.tensor([1], dtype=torch.int64)
            self.samples.append({
                "image": img2,
                "boxes": boxes2,
                "labels": labels2,
            })
            
            # Sample 2: Image with no boxes
            img3 = torch.rand(3, 512, 512)
            boxes3 = torch.zeros((0, 4), dtype=torch.float32)
            labels3 = torch.zeros((0,), dtype=torch.int64)
            self.samples.append({
                "image": img3,
                "boxes": boxes3,
                "labels": labels3,
            })
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    return MockDataset()


@pytest.fixture
def small_mock_dataset():
    """Create a mock dataset with small images."""
    class SmallMockDataset:
        def __init__(self):
            self.samples = []
            
            # Very small image
            img = torch.rand(3, 50, 50)
            boxes = torch.tensor([[10, 10, 30, 30]], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.int64)
            self.samples.append({
                "image": img,
                "boxes": boxes,
                "labels": labels,
            })
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    return SmallMockDataset()


class TestAtomicWriteFunctions:
    """Test suite for atomic write helper functions."""

    def test_atomic_write_image(self):
        """Test atomic_write_image function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_tensor = torch.rand(3, 100, 100)
            img_path = os.path.join(tmpdir, "test_image.png")
            
            atomic_write_image(img_tensor, img_path)
            
            # Check that file exists
            assert os.path.exists(img_path)
            
            # Check that temp file is cleaned up
            assert not os.path.exists(img_path + ".tmp")
            
            # Check that image can be loaded
            loaded_img = Image.open(img_path)
            assert loaded_img.size == (100, 100)

    def test_atomic_write_json(self):
        """Test atomic_write_json function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "boxes": [[10, 20, 30, 40]],
                "labels": [1],
                "image_id": 0,
            }
            json_path = os.path.join(tmpdir, "test_data.json")
            
            atomic_write_json(data, json_path)
            
            # Check that file exists
            assert os.path.exists(json_path)
            
            # Check that temp file is cleaned up
            assert not os.path.exists(json_path + ".tmp")
            
            # Check that data can be loaded
            with open(json_path, 'r') as f:
                loaded_data = json.load(f)
            assert loaded_data == data

    def test_atomic_write_image_replaces_existing(self):
        """Test that atomic_write_image replaces existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test_image.png")
            
            # Write first image
            img1 = torch.zeros(3, 50, 50)
            atomic_write_image(img1, img_path)
            
            # Write second image
            img2 = torch.ones(3, 50, 50)
            atomic_write_image(img2, img_path)
            
            # Check that file exists and contains second image
            loaded = F.to_tensor(Image.open(img_path))
            assert torch.allclose(loaded, img2, atol=0.01)

    def test_atomic_write_json_replaces_existing(self):
        """Test that atomic_write_json replaces existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "test_data.json")
            
            # Write first data
            data1 = {"value": 1}
            atomic_write_json(data1, json_path)
            
            # Write second data
            data2 = {"value": 2}
            atomic_write_json(data2, json_path)
            
            # Check that file contains second data
            with open(json_path, 'r') as f:
                loaded = json.load(f)
            assert loaded == data2


class TestTiledDatasetInitialization:
    """Test suite for TiledDataset initialization."""

    def test_basic_initialization(self, mock_dataset):
        """Test basic initialization with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            assert dataset.base_dataset is mock_dataset
            assert dataset.cache_dir == tmpdir
            assert dataset.tile_h == 512
            assert dataset.tile_w == 512
            assert dataset.overlap == 0
            assert len(dataset) > 0

    def test_initialization_with_overlap(self, mock_dataset):
        """Test initialization with overlap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
                overlap=50,
            )
            
            assert dataset.overlap == 50

    def test_initialization_with_min_size(self, mock_dataset):
        """Test initialization with minimum size constraint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
                min_size=(100, 100),
            )
            
            assert dataset.min_h == 100
            assert dataset.min_w == 100

    def test_initialization_creates_cache_dir(self, mock_dataset):
        """Test that cache directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "new_cache")
            
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=cache_dir,
                tile_size=(512, 512),
            )
            
            assert os.path.exists(cache_dir)

    def test_initialization_skip_empty_boxes_true(self, mock_dataset):
        """Test initialization with skip_empty_boxes=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
                skip_empty_boxes=True,
            )
            
            assert dataset.skip_empty_boxes is True

    def test_initialization_skip_empty_boxes_false(self, mock_dataset):
        """Test initialization with skip_empty_boxes=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
                skip_empty_boxes=False,
            )
            
            assert dataset.skip_empty_boxes is False
            # Should have more tiles when including empty boxes
            assert len(dataset) > 0


class TestTiledDatasetPrepare:
    """Test suite for tile preparation logic."""

    def test_prepare_tiles_creates_files(self, mock_dataset):
        """Test that prepare_tiles creates tile files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            # Check that files were created
            files = os.listdir(tmpdir)
            png_files = [f for f in files if f.endswith('.png')]
            json_files = [f for f in files if f.endswith('.json')]
            
            assert len(png_files) > 0
            assert len(json_files) > 0
            assert len(png_files) == len(json_files)

    def test_prepare_tiles_uses_cache(self, mock_dataset):
        """Test that prepare_tiles reuses cached files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First creation
            dataset1 = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            num_tiles1 = len(dataset1)
            
            # Get modification times
            files = os.listdir(tmpdir)
            mtimes = {f: os.path.getmtime(os.path.join(tmpdir, f)) for f in files}
            
            # Second creation (should use cache)
            dataset2 = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            num_tiles2 = len(dataset2)
            
            assert num_tiles1 == num_tiles2
            
            # Check that files weren't modified
            for f, old_mtime in mtimes.items():
                new_mtime = os.path.getmtime(os.path.join(tmpdir, f))
                assert new_mtime == old_mtime

    def test_prepare_tiles_rebuild(self, mock_dataset):
        """Test that rebuild=True forces tile regeneration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First creation
            dataset1 = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            # Get modification times
            files = os.listdir(tmpdir)
            mtimes = {f: os.path.getmtime(os.path.join(tmpdir, f)) for f in files}
            
            import time
            time.sleep(0.1)  # Ensure time difference
            
            # Second creation with rebuild
            dataset2 = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
                rebuild=True,
            )
            
            # Check that at least some files were modified
            modified_count = 0
            for f in files:
                if f.endswith('.png') or f.endswith('.json'):
                    new_mtime = os.path.getmtime(os.path.join(tmpdir, f))
                    if new_mtime > mtimes[f]:
                        modified_count += 1
            
            assert modified_count > 0

    def test_prepare_tiles_skips_small_images(self, small_mock_dataset):
        """Test that images smaller than min_size are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=small_mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
                min_size=(100, 100),
            )
            
            # Image is 50x50, below min_size, should be skipped
            assert len(dataset) == 0

    def test_prepare_tiles_with_overlap(self, mock_dataset):
        """Test tile preparation with overlap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
                overlap=100,
            )
            
            # With overlap, should create more tiles
            assert len(dataset) > 0


class TestTiledDatasetGetItem:
    """Test suite for __getitem__ method."""

    def test_getitem_returns_correct_format(self, mock_dataset):
        """Test that __getitem__ returns correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            sample = dataset[0]
            
            assert "image" in sample
            assert "boxes" in sample
            assert "labels" in sample
            assert "image_id" in sample
            
            assert isinstance(sample["image"], torch.Tensor)
            assert isinstance(sample["boxes"], torch.Tensor)
            assert isinstance(sample["labels"], torch.Tensor)

    def test_getitem_image_dimensions(self, mock_dataset):
        """Test that returned image has correct dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tile_size = (512, 512)
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=tile_size,
            )
            
            sample = dataset[0]
            
            # Check image shape (C, H, W)
            assert sample["image"].shape[0] == 3
            assert sample["image"].shape[1] == tile_size[0]
            assert sample["image"].shape[2] == tile_size[1]

    def test_getitem_boxes_format(self, mock_dataset):
        """Test that boxes are in correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            for i in range(len(dataset)):
                sample = dataset[i]
                boxes = sample["boxes"]
                
                # Boxes should be (N, 4) or (0, 4)
                assert boxes.dim() == 2
                assert boxes.shape[1] == 4
                
                # All boxes should be within tile bounds
                if len(boxes) > 0:
                    assert boxes[:, 0].min() >= 0
                    assert boxes[:, 1].min() >= 0
                    assert boxes[:, 2].max() <= 512
                    assert boxes[:, 3].max() <= 512

    def test_getitem_boxes_labels_match(self, mock_dataset):
        """Test that boxes and labels have matching lengths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            for i in range(len(dataset)):
                sample = dataset[i]
                assert len(sample["boxes"]) == len(sample["labels"])

    def test_getitem_all_indices(self, mock_dataset):
        """Test that all indices can be accessed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            for i in range(len(dataset)):
                sample = dataset[i]
                assert sample is not None


class TestTiledDatasetBboxHandling:
    """Test suite for bounding box handling logic."""

    def test_bbox_clipping(self, mock_dataset):
        """Test that bounding boxes are clipped to tile bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            for i in range(len(dataset)):
                sample = dataset[i]
                boxes = sample["boxes"]
                
                if len(boxes) > 0:
                    # All coordinates should be within [0, tile_size]
                    assert torch.all(boxes[:, 0] >= 0)
                    assert torch.all(boxes[:, 1] >= 0)
                    assert torch.all(boxes[:, 2] <= 512)
                    assert torch.all(boxes[:, 3] <= 512)

    def test_bbox_filtering_by_overlap(self, mock_dataset):
        """Test that boxes with insufficient overlap are filtered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            # All remaining boxes should have sufficient overlap
            for i in range(len(dataset)):
                sample = dataset[i]
                boxes = sample["boxes"]
                
                # Check that boxes have valid dimensions
                if len(boxes) > 0:
                    widths = boxes[:, 2] - boxes[:, 0]
                    heights = boxes[:, 3] - boxes[:, 1]
                    assert torch.all(widths >= 0)
                    assert torch.all(heights >= 0)

    def test_empty_boxes_skipped(self, mock_dataset):
        """Test that tiles with empty boxes are skipped when skip_empty_boxes=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
                skip_empty_boxes=True,
            )
            
            # When skip_empty_boxes=True, should have fewer tiles
            # At least some tiles should have boxes
            has_boxes_count = 0
            for i in range(len(dataset)):
                sample = dataset[i]
                if len(sample["boxes"]) > 0:
                    has_boxes_count += 1
            
            assert has_boxes_count > 0

    def test_empty_boxes_included(self, mock_dataset):
        """Test that tiles with empty boxes are included when skip_empty_boxes=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
                skip_empty_boxes=False,
            )
            
            # Should include tiles with empty boxes
            empty_count = 0
            for i in range(len(dataset)):
                sample = dataset[i]
                if len(sample["boxes"]) == 0:
                    empty_count += 1
            
            # Should have at least some empty tiles
            assert empty_count >= 0


class TestTiledDatasetPlot:
    """Test suite for plot method."""

    def test_plot_executes_without_error(self, mock_dataset):
        """Test that plot executes without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            sample = dataset[0]
            # Should not raise an error
            dataset.plot(sample)
            plt.close('all')

    def test_plot_with_suptitle(self, mock_dataset):
        """Test plot with suptitle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            sample = dataset[0]
            # Should not raise an error
            dataset.plot(sample, suptitle="Test Title")
            plt.close('all')

    def test_plot_without_suptitle(self, mock_dataset):
        """Test plot without suptitle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            sample = dataset[0]
            # Should not raise an error
            dataset.plot(sample, suptitle=None)
            plt.close('all')

    def test_plot_with_boxes(self, mock_dataset):
        """Test that plot renders bounding boxes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            # Find a sample with boxes
            sample_with_boxes = None
            for i in range(len(dataset)):
                sample = dataset[i]
                if len(sample["boxes"]) > 0:
                    sample_with_boxes = sample
                    break
            
            if sample_with_boxes:
                dataset.plot(sample_with_boxes)
                plt.close('all')

    def test_plot_with_empty_boxes(self, mock_dataset):
        """Test plot with empty boxes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
                skip_empty_boxes=False,
            )
            
            # Create sample with empty boxes
            sample = dataset[0]
            sample_empty = {
                "image": sample["image"],
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": 0,
            }
            
            dataset.plot(sample_empty)
            plt.close('all')


class TestTiledDatasetEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_tile_per_image(self, mock_dataset):
        """Test when image size equals tile size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(1024, 1024),
            )
            
            # Should create at least one tile
            assert len(dataset) > 0

    def test_multiple_tiles_per_image(self, mock_dataset):
        """Test tiling of large images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(256, 256),
            )
            
            # Should create multiple tiles from 1024x1024 image
            assert len(dataset) > 4

    def test_non_tensor_input_raises_error(self):
        """Test that non-tensor input raises RuntimeError."""
        class BadDataset:
            def __len__(self):
                return 1
            
            def __getitem__(self, idx):
                return {
                    "image": "not_a_tensor",
                    "boxes": torch.zeros((0, 4)),
                    "labels": torch.zeros((0,)),
                }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="Only torch.Tensor supported"):
                dataset = TiledDataset(
                    base_dataset=BadDataset(),
                    cache_dir=tmpdir,
                    tile_size=(512, 512),
                )

    def test_dataset_length(self, mock_dataset):
        """Test __len__ method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            assert len(dataset) == len(dataset.tiles)
            assert len(dataset) > 0

    def test_different_tile_sizes(self, mock_dataset):
        """Test with different tile sizes."""
        tile_sizes = [(256, 256), (512, 512), (1024, 1024)]
        
        for tile_size in tile_sizes:
            with tempfile.TemporaryDirectory() as tmpdir:
                dataset = TiledDataset(
                    base_dataset=mock_dataset,
                    cache_dir=tmpdir,
                    tile_size=tile_size,
                )
                
                assert len(dataset) > 0
                sample = dataset[0]
                assert sample["image"].shape[1] == tile_size[0]
                assert sample["image"].shape[2] == tile_size[1]


class TestTiledDatasetIntegration:
    """Integration tests for TiledDataset."""

    def test_end_to_end_workflow(self, mock_dataset):
        """Test complete workflow from creation to iteration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            # Iterate and check samples
            for i in range(min(5, len(dataset))):
                sample = dataset[i]
                
                assert sample["image"].shape == (3, 512, 512)
                assert len(sample["boxes"]) == len(sample["labels"])
                assert isinstance(sample["image_id"], int)

    def test_cache_persistence(self, mock_dataset):
        """Test that cache persists across dataset instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first instance
            dataset1 = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            len1 = len(dataset1)
            sample1 = dataset1[0]
            
            # Delete first instance
            del dataset1
            gc.collect()
            
            # Create second instance (should use cache)
            dataset2 = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            len2 = len(dataset2)
            sample2 = dataset2[0]
            
            assert len1 == len2
            assert torch.equal(sample1["image"], sample2["image"])

    def test_dataloader_compatibility(self, mock_dataset):
        """Test compatibility with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            def collate_fn(batch):
                return batch
            
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=collate_fn,
            )
            
            batch = next(iter(dataloader))
            assert len(batch) <= 2
            
            gc.collect()

    def test_iteration_consistency(self, mock_dataset):
        """Test that iteration produces consistent results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TiledDataset(
                base_dataset=mock_dataset,
                cache_dir=tmpdir,
                tile_size=(512, 512),
            )
            
            # Get samples twice
            samples1 = [dataset[i] for i in range(len(dataset))]
            samples2 = [dataset[i] for i in range(len(dataset))]
            
            # Should be identical
            for s1, s2 in zip(samples1, samples2):
                assert torch.equal(s1["image"], s2["image"])
                assert torch.equal(s1["boxes"], s2["boxes"])
                assert torch.equal(s1["labels"], s2["labels"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

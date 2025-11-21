"""Comprehensive tests for ElephantCocoDataset.

Tests cover initialization, getitem, bbox handling, edge cases to ensure maximum code coverage.
"""

import gc
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from terratorch.datasets.od_aed_elephant import ElephantCocoDataset


@pytest.fixture
def temp_coco_dataset():
    """Create a temporary COCO dataset structure with images and annotations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create images directory
        img_dir = tmpdir / "images"
        img_dir.mkdir()
        
        # Create sample images
        for i in range(5):
            img = Image.new('RGB', (640, 480), color=(i*50, 100, 150))
            img.save(img_dir / f"image_{i}.jpg")
        
        # Create COCO annotations
        annotations = {
            "images": [
                {"id": 0, "file_name": "image_0.jpg", "width": 640, "height": 480},
                {"id": 1, "file_name": "image_1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "image_2.jpg", "width": 640, "height": 480},
                {"id": 3, "file_name": "image_3.jpg", "width": 640, "height": 480},
                {"id": 4, "file_name": "image_4.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                # Image 0: normal box
                {"id": 0, "image_id": 0, "category_id": 1, "bbox": [100, 100, 200, 150], "area": 30000, "iscrowd": 0},
                # Image 1: multiple boxes
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 50, 100, 100], "area": 10000, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [300, 200, 150, 100], "area": 15000, "iscrowd": 0},
                # Image 2: zero width box (should be filtered)
                {"id": 3, "image_id": 2, "category_id": 1, "bbox": [100, 100, 0, 150], "area": 0, "iscrowd": 0},
                # Image 3: zero height box (should be filtered)
                {"id": 4, "image_id": 3, "category_id": 1, "bbox": [100, 100, 200, 0], "area": 0, "iscrowd": 0},
                # Image 4: negative dimensions (should be filtered)
                {"id": 5, "image_id": 4, "category_id": 1, "bbox": [100, 100, -50, 150], "area": 0, "iscrowd": 0},
            ],
            "categories": [
                {"id": 1, "name": "elephant"},
                {"id": 2, "name": "other"},
            ]
        }
        
        ann_file = tmpdir / "annotations.json"
        with open(ann_file, 'w') as f:
            json.dump(annotations, f)
        
        yield str(img_dir), str(ann_file)


@pytest.fixture
def temp_empty_annotations():
    """Create a dataset with images but empty annotations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        img_dir = tmpdir / "images"
        img_dir.mkdir()
        
        # Create sample image
        img = Image.new('RGB', (640, 480), color=(100, 100, 100))
        img.save(img_dir / "image_0.jpg")
        
        # Empty annotations
        annotations = {
            "images": [
                {"id": 0, "file_name": "image_0.jpg", "width": 640, "height": 480},
            ],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "elephant"},
            ]
        }
        
        ann_file = tmpdir / "annotations.json"
        with open(ann_file, 'w') as f:
            json.dump(annotations, f)
        
        yield str(img_dir), str(ann_file)


class TestElephantCocoDatasetInitialization:
    """Test suite for ElephantCocoDataset initialization."""

    def test_basic_initialization(self, temp_coco_dataset):
        """Test basic initialization without transform."""
        img_folder, ann_file = temp_coco_dataset
        
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        assert dataset is not None
        assert len(dataset) == 5
        assert dataset.transform is None

    def test_initialization_with_transform(self, temp_coco_dataset):
        """Test initialization with custom transform."""
        img_folder, ann_file = temp_coco_dataset
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ColorJitter(brightness=0.2)
        ])
        
        dataset = ElephantCocoDataset(img_folder, ann_file, transform=transform)
        
        assert dataset.transform is not None
        assert len(dataset) == 5

    def test_initialization_with_none_transform(self, temp_coco_dataset):
        """Test initialization with explicit None transform."""
        img_folder, ann_file = temp_coco_dataset
        
        dataset = ElephantCocoDataset(img_folder, ann_file, transform=None)
        
        assert dataset.transform is None


class TestElephantCocoDatasetGetItem:
    """Test suite for __getitem__ method."""

    def test_getitem_single_box(self, temp_coco_dataset):
        """Test getting item with single bounding box."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        sample = dataset[0]
        
        assert "image" in sample
        assert "boxes" in sample
        assert "labels" in sample
        assert "image_id" in sample
        
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["boxes"], torch.Tensor)
        assert isinstance(sample["labels"], torch.Tensor)
        assert isinstance(sample["image_id"], torch.Tensor)
        
        # Check dimensions
        assert sample["image"].shape[0] == 3  # RGB channels
        assert sample["boxes"].shape == torch.Size([1, 4])
        assert sample["labels"].shape == torch.Size([1])

    def test_getitem_multiple_boxes(self, temp_coco_dataset):
        """Test getting item with multiple bounding boxes."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        sample = dataset[1]
        
        # Image 1 has 2 valid boxes
        assert sample["boxes"].shape[0] == 2
        assert sample["labels"].shape[0] == 2
        
        # Check box format (x1, y1, x2, y2)
        boxes = sample["boxes"]
        for box in boxes:
            assert box[2] > box[0]  # x2 > x1
            assert box[3] > box[1]  # y2 > y1

    def test_getitem_zero_width_box(self, temp_coco_dataset):
        """Test that boxes with zero width are filtered out."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        sample = dataset[2]  # Image 2 has zero-width box
        
        # Should have no boxes
        assert sample["boxes"].shape == torch.Size([0, 4])
        assert sample["labels"].shape == torch.Size([0])

    def test_getitem_zero_height_box(self, temp_coco_dataset):
        """Test that boxes with zero height are filtered out."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        sample = dataset[3]  # Image 3 has zero-height box
        
        # Should have no boxes
        assert sample["boxes"].shape == torch.Size([0, 4])
        assert sample["labels"].shape == torch.Size([0])

    def test_getitem_negative_dimensions(self, temp_coco_dataset):
        """Test that boxes with negative dimensions are filtered out."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        sample = dataset[4]  # Image 4 has negative width
        
        # Should have no boxes
        assert sample["boxes"].shape == torch.Size([0, 4])
        assert sample["labels"].shape == torch.Size([0])

    def test_getitem_empty_annotations(self, temp_empty_annotations):
        """Test getting item when no annotations exist."""
        img_folder, ann_file = temp_empty_annotations
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        sample = dataset[0]
        
        # Should have empty tensors
        assert sample["boxes"].shape == torch.Size([0, 4])
        assert sample["labels"].shape == torch.Size([0])
        assert sample["boxes"].dtype == torch.float32
        assert sample["labels"].dtype == torch.int64

    def test_getitem_with_transform(self, temp_coco_dataset):
        """Test that transform is applied to image."""
        img_folder, ann_file = temp_coco_dataset
        transform = T.Resize((224, 224))
        dataset = ElephantCocoDataset(img_folder, ann_file, transform=transform)
        
        sample = dataset[0]
        
        # After ToTensor, shape is (C, H, W)
        # Transform should have resized it
        assert sample["image"].shape[1] == 224
        assert sample["image"].shape[2] == 224

    def test_getitem_image_format(self, temp_coco_dataset):
        """Test that image is correctly converted to tensor."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        sample = dataset[0]
        
        # Check tensor properties
        assert sample["image"].dtype == torch.float32
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0
        assert sample["image"].shape[0] == 3  # RGB

    def test_getitem_box_coordinates(self, temp_coco_dataset):
        """Test that box coordinates are correctly converted from COCO format."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        sample = dataset[0]
        
        # COCO format: [100, 100, 200, 150] (x, y, w, h)
        # Expected: [100, 100, 300, 250] (x1, y1, x2, y2)
        box = sample["boxes"][0]
        
        assert box[0] == 100  # x1
        assert box[1] == 100  # y1
        assert box[2] == 300  # x2 = x1 + w
        assert box[3] == 250  # y2 = y1 + h

    def test_getitem_labels_format(self, temp_coco_dataset):
        """Test that labels are correctly extracted."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        sample = dataset[1]  # Image with 2 boxes
        
        # Check labels
        assert sample["labels"][0] == 1  # First category
        assert sample["labels"][1] == 2  # Second category
        assert sample["labels"].dtype == torch.int64

    def test_getitem_image_id(self, temp_coco_dataset):
        """Test that image_id is correctly set."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        for i in range(len(dataset)):
            sample = dataset[i]
            # image_id should match the index
            assert sample["image_id"].item() == i


class TestElephantCocoDatasetEdgeCases:
    """Test edge cases and special scenarios."""

    def test_multiple_access_same_item(self, temp_coco_dataset):
        """Test accessing the same item multiple times."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        sample1 = dataset[0]
        sample2 = dataset[0]
        
        # Should return identical data
        assert torch.equal(sample1["boxes"], sample2["boxes"])
        assert torch.equal(sample1["labels"], sample2["labels"])

    def test_all_valid_indices(self, temp_coco_dataset):
        """Test that all indices can be accessed."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        for i in range(len(dataset)):
            sample = dataset[i]
            assert "image" in sample
            assert "boxes" in sample
            assert "labels" in sample

    def test_box_tensor_dtype(self, temp_coco_dataset):
        """Test that boxes have correct dtype."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        sample = dataset[0]
        
        assert sample["boxes"].dtype == torch.float32

    def test_label_tensor_dtype(self, temp_coco_dataset):
        """Test that labels have correct dtype."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        sample = dataset[0]
        
        assert sample["labels"].dtype == torch.int64

    def test_empty_boxes_shape(self, temp_coco_dataset):
        """Test that empty boxes have correct shape."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        sample = dataset[2]  # Has no valid boxes
        
        # Should be (0, 4) not (0,) or something else
        assert sample["boxes"].shape == torch.Size([0, 4])
        assert sample["boxes"].dim() == 2

    def test_empty_labels_shape(self, temp_coco_dataset):
        """Test that empty labels have correct shape."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        sample = dataset[2]  # Has no valid boxes
        
        # Should be (0,)
        assert sample["labels"].shape == torch.Size([0])
        assert sample["labels"].dim() == 1

    def test_box_validity_constraints(self, temp_coco_dataset):
        """Test that all returned boxes satisfy validity constraints."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        for i in range(len(dataset)):
            sample = dataset[i]
            boxes = sample["boxes"]
            
            if len(boxes) > 0:
                # All boxes should have positive width and height
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]
                
                assert torch.all(widths > 0)
                assert torch.all(heights > 0)


class TestElephantCocoDatasetTransforms:
    """Test suite for various transforms."""

    def test_resize_transform(self, temp_coco_dataset):
        """Test with resize transform."""
        img_folder, ann_file = temp_coco_dataset
        transform = T.Resize((300, 400))
        dataset = ElephantCocoDataset(img_folder, ann_file, transform=transform)
        
        sample = dataset[0]
        
        assert sample["image"].shape[1] == 300
        assert sample["image"].shape[2] == 400

    def test_color_jitter_transform(self, temp_coco_dataset):
        """Test with color jitter transform."""
        img_folder, ann_file = temp_coco_dataset
        transform = T.ColorJitter(brightness=0.5, contrast=0.5)
        dataset = ElephantCocoDataset(img_folder, ann_file, transform=transform)
        
        sample = dataset[0]
        
        # Should still have valid image
        assert sample["image"].shape[0] == 3
        assert torch.isfinite(sample["image"]).all()

    def test_compose_transforms(self, temp_coco_dataset):
        """Test with composed transforms."""
        img_folder, ann_file = temp_coco_dataset
        transform = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ColorJitter(brightness=0.2)
        ])
        dataset = ElephantCocoDataset(img_folder, ann_file, transform=transform)
        
        sample = dataset[0]
        
        assert sample["image"].shape[1] == 224
        assert sample["image"].shape[2] == 224

    def test_grayscale_transform(self, temp_coco_dataset):
        """Test with grayscale transform."""
        img_folder, ann_file = temp_coco_dataset
        transform = T.Grayscale(num_output_channels=3)
        dataset = ElephantCocoDataset(img_folder, ann_file, transform=transform)
        
        sample = dataset[0]
        
        # Should still have 3 channels after conversion
        assert sample["image"].shape[0] == 3


class TestElephantCocoDatasetIntegration:
    """Integration tests for ElephantCocoDataset."""

    def test_dataset_length(self, temp_coco_dataset):
        """Test that dataset length matches number of images."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        assert len(dataset) == 5

    def test_iteration_over_dataset(self, temp_coco_dataset):
        """Test iterating over entire dataset."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        count = 0
        for sample in dataset:
            assert "image" in sample
            assert "boxes" in sample
            assert "labels" in sample
            assert "image_id" in sample
            count += 1
        
        assert count == len(dataset)

    def test_dataloader_compatibility(self, temp_coco_dataset):
        """Test that dataset works with DataLoader."""
        from torch.utils.data import DataLoader
        
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        # Custom collate function for variable-sized boxes
        def collate_fn(batch):
            return batch
        
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False,
            collate_fn=collate_fn
        )
        
        batch = next(iter(dataloader))
        assert len(batch) == 2
        
        gc.collect()

    def test_consistency_across_epochs(self, temp_coco_dataset):
        """Test that data is consistent across multiple epochs."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        # Get samples from first epoch
        samples_epoch1 = [dataset[i] for i in range(len(dataset))]
        
        # Get samples from second epoch
        samples_epoch2 = [dataset[i] for i in range(len(dataset))]
        
        # Should be identical
        for s1, s2 in zip(samples_epoch1, samples_epoch2):
            assert torch.equal(s1["boxes"], s2["boxes"])
            assert torch.equal(s1["labels"], s2["labels"])

    def test_memory_cleanup(self, temp_coco_dataset):
        """Test that dataset can be properly cleaned up."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        # Access some samples
        for i in range(len(dataset)):
            _ = dataset[i]
        
        # Delete and cleanup
        del dataset
        gc.collect()

    def test_boxes_labels_correspondence(self, temp_coco_dataset):
        """Test that boxes and labels have matching lengths."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        for i in range(len(dataset)):
            sample = dataset[i]
            assert len(sample["boxes"]) == len(sample["labels"])

    def test_all_images_accessible(self, temp_coco_dataset):
        """Test that all images can be loaded without errors."""
        img_folder, ann_file = temp_coco_dataset
        dataset = ElephantCocoDataset(img_folder, ann_file)
        
        for i in range(len(dataset)):
            sample = dataset[i]
            # Check that image was loaded
            assert sample["image"].numel() > 0
            assert torch.isfinite(sample["image"]).all()


class TestElephantCocoDatasetBoundaryConditions:
    """Test boundary conditions and special cases."""

    def test_single_pixel_box(self):
        """Test handling of very small boxes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            img_dir = tmpdir / "images"
            img_dir.mkdir()
            
            # Create image
            img = Image.new('RGB', (640, 480))
            img.save(img_dir / "image_0.jpg")
            
            # Box with width=1, height=1
            annotations = {
                "images": [
                    {"id": 0, "file_name": "image_0.jpg", "width": 640, "height": 480},
                ],
                "annotations": [
                    {"id": 0, "image_id": 0, "category_id": 1, "bbox": [100, 100, 1, 1], "area": 1, "iscrowd": 0},
                ],
                "categories": [{"id": 1, "name": "elephant"}]
            }
            
            ann_file = tmpdir / "annotations.json"
            with open(ann_file, 'w') as f:
                json.dump(annotations, f)
            
            dataset = ElephantCocoDataset(str(img_dir), str(ann_file))
            sample = dataset[0]
            
            # Should accept 1x1 box as valid
            assert len(sample["boxes"]) == 1

    def test_large_box(self):
        """Test handling of large boxes spanning entire image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            img_dir = tmpdir / "images"
            img_dir.mkdir()
            
            img = Image.new('RGB', (640, 480))
            img.save(img_dir / "image_0.jpg")
            
            # Box covering entire image
            annotations = {
                "images": [
                    {"id": 0, "file_name": "image_0.jpg", "width": 640, "height": 480},
                ],
                "annotations": [
                    {"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 640, 480], "area": 307200, "iscrowd": 0},
                ],
                "categories": [{"id": 1, "name": "elephant"}]
            }
            
            ann_file = tmpdir / "annotations.json"
            with open(ann_file, 'w') as f:
                json.dump(annotations, f)
            
            dataset = ElephantCocoDataset(str(img_dir), str(ann_file))
            sample = dataset[0]
            
            assert len(sample["boxes"]) == 1
            box = sample["boxes"][0]
            assert box[0] == 0
            assert box[1] == 0
            assert box[2] == 640
            assert box[3] == 480


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Comprehensive tests for mVHR10 dataset.

Tests cover initialization, splits, getitem, plotting, edge cases to ensure maximum code coverage.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest
import torch
import numpy as np
from PIL import Image

from terratorch.datasets.m_VHR10 import mVHR10


# Autouse fixture to mock the parent VHR10 dataset so tests don't depend on
# the real NWPU VHR-10 dataset being present locally. This lets us exercise
# the logic inside mVHR10 (__init__, __getitem__, plot) deterministically.
@pytest.fixture(autouse=True)
def mock_vhr10(monkeypatch):
    from torchgeo.datasets import VHR10

    # Fake parent __init__ providing the minimal attributes mVHR10 relies on.
    def fake_init(self, root=None, split="positive", transforms=None, **kwargs):  # noqa: D401
        self.root = root
        self.split = split
        self.transforms = transforms
        # Provide 10 deterministic ids (0..9); mVHR10 will shuffle and slice.
        self.ids = list(range(10))
        # Minimal category list for color mapping in plot.
        self.categories = ["airplane", "ship", "tank"]

    def fake_load_image(self, id_):
        # Return a dummy RGB image tensor.
        return torch.rand(3, 600, 800)

    def fake_load_target(self, id_):
        # Even ids have one annotation, odd ids have none to exercise both paths.
        if id_ % 2 == 0:
            return {
                "annotations": [
                    {
                        "bbox": [100, 100, 200, 150],  # x, y, w, h format
                        "category_id": 0,
                        "segmentation": [[100, 100, 300, 100, 300, 250, 100, 250]],
                    }
                ]
            }
        return {"annotations": []}

    def fake_coco_convert(self, sample):
        anns = sample["label"].get("annotations", [])
        if not anns:
            return sample
        boxes = []
        labels = []
        masks = []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
            masks.append(torch.ones(600, 800))
        sample["label"] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": masks,
        }
        return sample

    monkeypatch.setattr(VHR10, "__init__", fake_init)
    monkeypatch.setattr(VHR10, "_load_image", fake_load_image)
    monkeypatch.setattr(VHR10, "_load_target", fake_load_target)
    monkeypatch.setattr(VHR10, "coco_convert", fake_coco_convert)

    # Patch lazy_import used in plot() to avoid importing heavy skimage.
    # Provide a dummy object with measure.find_contours.
    class DummyMeasure:
        def find_contours(self, mask, level):  # noqa: D401
            # Simple square contour
            return [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])]

    class DummySkimage:
        measure = DummyMeasure()

    def fake_lazy_import(name):  # noqa: D401
        return DummySkimage()

    monkeypatch.setattr("terratorch.datasets.m_VHR10.lazy_import", fake_lazy_import)


@pytest.fixture
def temp_vhr10_dataset():
    """Create a temporary VHR10-like dataset structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create positive directory with images
        positive_dir = tmpdir / "positive image set"
        positive_dir.mkdir(parents=True)
        
        # Create 10 sample images
        for i in range(1, 11):
            img = Image.new('RGB', (800, 600), color=(i*20, 100, 150))
            img.save(positive_dir / f"{i:03d}.jpg")
        
        # Create annotations directory
        ann_dir = tmpdir / "ground truth"
        ann_dir.mkdir()
        
        # Create COCO annotations for positive images
        annotations = {
            "images": [
                {"id": i, "file_name": f"{i:03d}.jpg", "width": 800, "height": 600}
                for i in range(1, 11)
            ],
            "annotations": [
                {"id": i, "image_id": i, "category_id": 1, "bbox": [100, 100, 200, 150], 
                 "area": 30000, "iscrowd": 0, "segmentation": [[100, 100, 300, 100, 300, 250, 100, 250]]}
                for i in range(1, 6)
            ],
            "categories": [
                {"id": 1, "name": "airplane"},
                {"id": 2, "name": "ship"},
                {"id": 3, "name": "storage tank"},
                {"id": 4, "name": "baseball diamond"},
                {"id": 5, "name": "tennis court"},
                {"id": 6, "name": "basketball court"},
                {"id": 7, "name": "ground track field"},
                {"id": 8, "name": "harbor"},
                {"id": 9, "name": "bridge"},
                {"id": 10, "name": "vehicle"},
            ]
        }
        
        ann_file = ann_dir / "NWPU VHR-10 dataset.json"
        with open(ann_file, 'w') as f:
            json.dump(annotations, f)
        
        # Create negative directory
        negative_dir = tmpdir / "negative image set"
        negative_dir.mkdir()
        
        for i in range(1, 6):
            img = Image.new('RGB', (800, 600), color=(50, i*30, 100))
            img.save(negative_dir / f"neg_{i}.jpg")
        
        yield str(tmpdir)


class TestMVHR10Initialization:
    """Test initialization and split creation."""
    
    def test_basic_train_split(self, temp_vhr10_dataset):
        """Test basic initialization with train split."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            second_level_split_proportions=(0.6, 0.2, 0.2),
            download=False
        )
        
        assert len(dataset) > 0
        assert dataset.second_level_split == "train"
        # Train split should have 60% of data
        assert len(dataset.ids) == 6  # 60% of 10 images
    
    def test_val_split(self, temp_vhr10_dataset):
        """Test validation split."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="val",
            second_level_split_proportions=(0.6, 0.2, 0.2),
            download=False
        )
        
        assert dataset.second_level_split == "val"
        # Val split should have 20% of data
        assert len(dataset.ids) == 2  # 20% of 10 images
    
    def test_test_split(self, temp_vhr10_dataset):
        """Test test split."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="test",
            second_level_split_proportions=(0.6, 0.2, 0.2),
            download=False
        )
        
        assert dataset.second_level_split == "test"
        # Test split should have 20% of data
        assert len(dataset.ids) == 2  # 20% of 10 images
    
    def test_custom_split_proportions(self, temp_vhr10_dataset):
        """Test custom split proportions."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            second_level_split_proportions=(0.8, 0.1, 0.1),
            download=False
        )
        
        # Train split should have 80% of data
        assert len(dataset.ids) == 8  # 80% of 10 images
    
    def test_custom_output_tags(self, temp_vhr10_dataset):
        """Test custom output tags."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            boxes_output_tag="custom_boxes",
            labels_output_tag="custom_labels",
            masks_output_tag="custom_masks",
            scores_output_tag="custom_scores",
            download=False
        )
        
        assert dataset.boxes_output_tag == "custom_boxes"
        assert dataset.labels_output_tag == "custom_labels"
        assert dataset.masks_output_tag == "custom_masks"
        assert dataset.scores_output_tag == "custom_scores"
    
    def test_negative_split(self, temp_vhr10_dataset):
        """Test negative split (images without annotations)."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="negative",
            second_level_split="train",
            download=False
        )
        
        assert len(dataset) > 0
    
    def test_invalid_second_level_split(self, temp_vhr10_dataset):
        """Test that invalid second level split raises assertion error."""
        with pytest.raises(AssertionError):
            mVHR10(
                root=temp_vhr10_dataset,
                split="positive",
                second_level_split="invalid",
                download=False
            )
    
    def test_invalid_proportions_length(self, temp_vhr10_dataset):
        """Test that invalid proportions length raises assertion error."""
        with pytest.raises(AssertionError):
            mVHR10(
                root=temp_vhr10_dataset,
                split="positive",
                second_level_split="train",
                second_level_split_proportions=(0.5, 0.5),  # Only 2 elements
                download=False
            )
    
    def test_invalid_proportions_sum(self, temp_vhr10_dataset):
        """Test that proportions not summing to 1 raises assertion error."""
        with pytest.raises(AssertionError):
            mVHR10(
                root=temp_vhr10_dataset,
                split="positive",
                second_level_split="train",
                second_level_split_proportions=(0.5, 0.3, 0.1),  # Sums to 0.9
                download=False
            )


class TestMVHR10GetItem:
    """Test __getitem__ method."""
    
    def test_getitem_with_annotations(self, temp_vhr10_dataset):
        """Test getting item with annotations."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            download=False
        )
        
        sample = dataset[0]
        
        assert 'image' in sample
        assert 'boxes' in sample
        assert 'labels' in sample
        assert 'masks' in sample
        
        # Check image tensor
        assert isinstance(sample['image'], torch.Tensor)
        assert sample['image'].shape[0] == 3  # RGB channels
        
        # Check boxes tensor
        assert isinstance(sample['boxes'], torch.Tensor)
        if len(sample['boxes']) > 0:
            assert sample['boxes'].shape[1] == 4  # [x1, y1, x2, y2]
    
    def test_getitem_custom_output_tags(self, temp_vhr10_dataset):
        """Test getting item with custom output tags."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            boxes_output_tag="my_boxes",
            labels_output_tag="my_labels",
            masks_output_tag="my_masks",
            download=False
        )
        
        sample = dataset[0]
        
        assert 'my_boxes' in sample
        assert 'my_labels' in sample
        assert 'my_masks' in sample
        assert 'label' not in sample  # Should be removed when labels_output_tag != 'label'
    
    def test_getitem_with_transforms(self, temp_vhr10_dataset):
        """Test getting item with transforms."""
        def dummy_transform(sample):
            sample['transformed'] = True
            return sample
        
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            transforms=dummy_transform,
            download=False
        )
        
        sample = dataset[0]
        assert sample.get('transformed', False) == True
    
    def test_getitem_empty_annotations(self, temp_vhr10_dataset):
        """Test getting item when annotation list is empty."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="test",  # Use test split which might have fewer annotations
            download=False
        )
        
        # Mock an image with empty annotations
        with patch.object(dataset, '_load_target') as mock_target:
            mock_target.return_value = {'annotations': []}
            
            sample = dataset[0]
            
            # Should still have image but transforms might not be applied
            assert 'image' in sample


class TestMVHR10Plot:
    """Test plot method."""
    
    @pytest.fixture
    def sample_with_annotations(self):
        """Create a sample with annotations for plotting."""
        return {
            'image': torch.rand(3, 600, 800),
            'boxes': torch.tensor([[100, 100, 300, 250], [400, 300, 600, 500]]),
            'labels': torch.tensor([0, 1]),
            'masks': [torch.ones(600, 800), torch.ones(600, 800)],
        }
    
    @pytest.fixture
    def sample_with_predictions(self):
        """Create a sample with predictions for plotting."""
        return {
            'image': torch.rand(3, 600, 800),
            'boxes': torch.tensor([[100, 100, 300, 250]]),
            'labels': torch.tensor([0]),
            'masks': [torch.ones(600, 800)],
            'prediction_labels': torch.tensor([0, 1]),
            'prediction_scores': torch.tensor([0.8, 0.6]),
            'prediction_boxes': torch.tensor([[110, 110, 310, 260], [410, 310, 610, 510]]),
            'prediction_masks': torch.ones(2, 1, 600, 800),
        }
    
    def test_plot_positive_split_boxes(self, temp_vhr10_dataset, sample_with_annotations):
        """Test plotting with boxes only."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            download=False
        )
        
        fig = dataset.plot(sample_with_annotations, show_feats='boxes')
        assert fig is not None
        assert len(fig.axes) == 1
    
    def test_plot_positive_split_masks(self, temp_vhr10_dataset, sample_with_annotations):
        """Test plotting with masks only."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            download=False
        )
        
        fig = dataset.plot(sample_with_annotations, show_feats='masks')
        assert fig is not None
    
    def test_plot_positive_split_both(self, temp_vhr10_dataset, sample_with_annotations):
        """Test plotting with both boxes and masks."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            download=False
        )
        
        fig = dataset.plot(sample_with_annotations, show_feats='both')
        assert fig is not None
    
    def test_plot_with_predictions(self, temp_vhr10_dataset, sample_with_predictions):
        """Test plotting with predictions."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            download=False
        )
        
        fig = dataset.plot(sample_with_predictions, show_feats='both')
        assert fig is not None
        assert len(fig.axes) == 2  # Ground truth and predictions
    
    def test_plot_with_predictions_boxes_only(self, temp_vhr10_dataset, sample_with_predictions):
        """Test plotting predictions with boxes only."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            download=False
        )
        
        # Remove prediction masks to test boxes-only path
        sample = sample_with_predictions.copy()
        del sample['prediction_masks']
        
        fig = dataset.plot(sample, show_feats='boxes')
        assert fig is not None
    
    def test_plot_with_predictions_masks_only(self, temp_vhr10_dataset, sample_with_predictions):
        """Test plotting predictions with masks only."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            download=False
        )
        
        # Remove prediction boxes to test masks-only path
        sample = sample_with_predictions.copy()
        del sample['prediction_boxes']
        
        fig = dataset.plot(sample, show_feats='masks')
        assert fig is not None
    
    def test_plot_with_low_confidence_predictions(self, temp_vhr10_dataset, sample_with_predictions):
        """Test plotting with predictions below confidence threshold."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            download=False
        )
        
        # Set high confidence threshold to filter predictions
        fig = dataset.plot(sample_with_predictions, confidence_score=0.9)
        assert fig is not None
    
    def test_plot_negative_split(self, temp_vhr10_dataset):
        """Test plotting negative split (no annotations)."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="negative",
            second_level_split="train",
            download=False
        )
        
        sample = {'image': torch.rand(3, 600, 800)}
        fig = dataset.plot(sample)
        assert fig is not None
        assert len(fig.axes) == 1
    
    def test_plot_with_suptitle(self, temp_vhr10_dataset, sample_with_annotations):
        """Test plotting with custom suptitle."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            download=False
        )
        
        fig = dataset.plot(sample_with_annotations, suptitle="Custom Title")
        assert fig is not None
    
    def test_plot_without_titles(self, temp_vhr10_dataset, sample_with_annotations):
        """Test plotting without titles."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            download=False
        )
        
        fig = dataset.plot(sample_with_annotations, show_titles=False)
        assert fig is not None
    
    def test_plot_custom_alpha_values(self, temp_vhr10_dataset, sample_with_annotations):
        """Test plotting with custom alpha values."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            download=False
        )
        
        fig = dataset.plot(sample_with_annotations, box_alpha=0.5, mask_alpha=0.3)
        assert fig is not None
    
    def test_plot_invalid_show_feats(self, temp_vhr10_dataset, sample_with_annotations):
        """Test that invalid show_feats raises assertion error."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            download=False
        )
        
        with pytest.raises(AssertionError):
            dataset.plot(sample_with_annotations, show_feats='invalid')
    
    def test_plot_without_masks_in_sample(self, temp_vhr10_dataset):
        """Test plotting when masks are not in sample."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            download=False
        )
        
        sample = {
            'image': torch.rand(3, 600, 800),
            'boxes': torch.tensor([[100, 100, 300, 250]]),
            'labels': torch.tensor([0]),
        }
        
        # Should plot without error even though masks key is missing
        fig = dataset.plot(sample, show_feats='boxes')
        assert fig is not None


class TestMVHR10EdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_split_consistency_across_runs(self, temp_vhr10_dataset):
        """Test that splits are consistent across multiple initializations."""
        dataset1 = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            second_level_split_proportions=(0.6, 0.2, 0.2),
            download=False
        )
        
        dataset2 = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            second_level_split_proportions=(0.6, 0.2, 0.2),
            download=False
        )
        
        # IDs should be the same due to fixed random seed (123)
        assert dataset1.ids == dataset2.ids
    
    def test_all_splits_cover_all_data(self, temp_vhr10_dataset):
        """Test that train/val/test splits cover all data without overlap."""
        proportions = (0.6, 0.2, 0.2)
        
        train_dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            second_level_split_proportions=proportions,
            download=False
        )
        
        val_dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="val",
            second_level_split_proportions=proportions,
            download=False
        )
        
        test_dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="test",
            second_level_split_proportions=proportions,
            download=False
        )
        
        # All splits should sum to total
        total_len = len(train_dataset.ids) + len(val_dataset.ids) + len(test_dataset.ids)
        assert total_len == 10  # Total images
        
        # Check no overlap
        all_ids = set(train_dataset.ids) | set(val_dataset.ids) | set(test_dataset.ids)
        assert len(all_ids) == 10
    
    def test_single_image_dataset(self, temp_vhr10_dataset):
        """Test behavior with very small dataset."""
        # Use proportions that result in single images per split
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            second_level_split_proportions=(0.7, 0.2, 0.1),
            download=False
        )
        
        # Should handle small datasets gracefully
        assert len(dataset) > 0
    
    def test_getitem_keeps_label_when_tag_is_label(self, temp_vhr10_dataset):
        """Test that 'label' key is kept when labels_output_tag is 'label'."""
        dataset = mVHR10(
            root=temp_vhr10_dataset,
            split="positive",
            second_level_split="train",
            labels_output_tag="label",  # Use default 'label' tag
            download=False
        )
        
        sample = dataset[0]
        
        # When labels_output_tag is 'label', the key should remain
        assert 'label' in sample

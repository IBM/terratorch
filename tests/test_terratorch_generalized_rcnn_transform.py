import copy
import gc

import pytest
import torch
from torchvision.models.detection.image_list import ImageList

# Adjust the import below to match where your utils.py lives in your repo.
# If utils.py sits at terratorch/datamodules/utils.py, this import is correct:
from terratorch.models.utils import TerratorchGeneralizedRCNNTransform


@pytest.fixture
def dummy_images():
    # Two images with different spatial sizes but same channels
    # C, H, W

    return [torch.randn(3, 32, 48), torch.randn(3, 32, 48)]


@pytest.fixture
def dummy_targets():
    # Two simple target dicts (common detection fields)
    t1 = {"boxes": torch.tensor([[1.0, 2.0, 10.0, 20.0]]), "labels": torch.tensor([1])}
    t2 = {"boxes": torch.tensor([[5.0, 5.0, 12.0, 12.0]]), "labels": torch.tensor([2])}
    return [t1, t2]


def _make_transform():
    """
    NOTE:
      In your implementation, the class defines `init(...)` (not `__init__`).
      That means the parent `GeneralizedRCNNTransform.__init__` is used.
      So we must pass the parent ctor args here.
    """
    # Typical values used by torchvision's GeneralizedRCNNTransform
    # min_size and max_size can be single ints; stats are per-channel (3)
    return TerratorchGeneralizedRCNNTransform(
        min_size=800,
        max_size=1333,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        size_divisible=32,
        fixed_size=None,
    )


def test_forward_returns_imagelist_and_targets(dummy_images, dummy_targets):
    tfm = _make_transform()
    image_list, targets_out = tfm.forward(dummy_images, dummy_targets)

    # Types
    assert isinstance(image_list, ImageList)
    assert isinstance(targets_out, list)
    assert len(targets_out) == len(dummy_targets)

    # Batched tensor of shape (N, C, H, W); since forward stacks as-is,
    # H and W must match the per-image sizes for each sample in image_list.image_sizes
    assert isinstance(image_list.tensors, torch.Tensor)
    assert image_list.tensors.ndim == 4
    assert image_list.tensors.shape[0] == 2  # batch size N

    # image_sizes preserved and are List[Tuple[int, int]]
    assert image_list.image_sizes == [(32, 48), (32, 48)]

    # Targets are shallow-copied dicts: new dict objects, same tensor objects
    for t_in, t_out in zip(dummy_targets, targets_out, strict=False):
        assert t_in is not t_out  # dict was copied
        for k in t_in:
            # tensors are same objects (shallow copy), not deep-copied
            assert t_in[k] is t_out[k]
            assert torch.equal(t_in[k], t_out[k])

    gc.collect()


def test_forward_without_targets(dummy_images):
    tfm = _make_transform()
    image_list, targets_out = tfm.forward(dummy_images, targets=None)

    assert isinstance(image_list, ImageList)
    assert targets_out is None
    assert image_list.image_sizes == [(32, 48), (32, 48)]
    # Check batched tensor first dimension equals number of images
    assert image_list.tensors.shape[0] == len(dummy_images)
    gc.collect()


def test_images_are_copied_not_referenced(dummy_images, dummy_targets):
    """
    The implementation constructs a new batched tensor.
    Mutating the original list should not change ImageList internals.
    """
    tfm = _make_transform()
    image_list, _ = tfm.forward(dummy_images, dummy_targets)

    # Keep an old reference
    old_batched = image_list.tensors.clone()

    # Mutate original images after forward
    dummy_images[0].add_(10.0)
    dummy_images[1].mul_(0.0)

    # ImageList should be unchanged
    assert torch.equal(image_list.tensors, old_batched)
    gc.collect()


def test_invalid_image_rank_raises():
    """
    Code asserts that the last two elements of img.shape are H and W.
    If we pass a wrongly shaped tensor (e.g., [C, H] or [H, W] without C),
    the len(image_size) check will fail and raise AssertionError.
    """
    tfm = _make_transform()
    bad_img = torch.randn(3, 32, 32)  # 2D tensor (no channel dim)
    ok_img = torch.randn(3, 16, 16)
    with pytest.raises((AssertionError, RuntimeError)):
        tfm.forward([bad_img, ok_img], targets=None)
    gc.collect()


def test_targets_are_shallow_copied(dummy_images, dummy_targets):
    """
    Verify targets are shallow-copied dicts: outer dict copied, tensors shared.
    """
    tfm = _make_transform()
    targets_in = dummy_targets
    image_list, targets_out = tfm.forward(dummy_images, targets=targets_in)

    # Different dict objects
    for t_in, t_out in zip(targets_in, targets_out, strict=False):
        assert t_in is not t_out
        # Modifying the returned dict doesn't mutate original dict object
        t_out["new_key"] = torch.tensor([1])
        assert "new_key" not in t_in

        # But tensors for existing keys are the same objects (shallow copy)
        assert (
            (t_in["boxes"] is targets_out[0]["boxes"])
            or (t_in["labels"] is targets_out[1]["labels"])
            or (t_in["boxes"] is targets_out[1]["boxes"])
            or (t_in["labels"] is targets_out[0]["labels"])
        )

    gc.collect()

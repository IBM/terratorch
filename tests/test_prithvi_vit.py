import pytest
import torch
from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from terratorch.models.backbones.select_patch_embed_weights import select_patch_embed_weights
from terratorch.registry import BACKBONE_REGISTRY, FULL_MODEL_REGISTRY

import gc

NUM_CHANNELS = 6

@pytest.fixture
def input_224():
    return torch.ones((1, NUM_CHANNELS, 224, 224))

@pytest.mark.parametrize("patch_size", [4, 8, 16])
@pytest.mark.parametrize("patch_size_time,num_frames", [(1, 1), (1, 2), (1, 3), (2, 2), (3,3)])
def test_prithvi_vit_patch_embed_loading_compatible(patch_size, patch_size_time, num_frames):
    model = BACKBONE_REGISTRY.build(
        "prithvi_eo_v1_100",
        pretrained=False,
        num_frames=num_frames,
        patch_size=[patch_size_time, 16, 16],
    )

    weights = BACKBONE_REGISTRY.build(
        "prithvi_eo_v1_100",
        pretrained=False,
        num_frames=num_frames,
        patch_size=[patch_size_time, 16, 16],
    ).state_dict()

    select_patch_embed_weights(weights, model, PRETRAINED_BANDS, PRETRAINED_BANDS)

    gc.collect()

@pytest.mark.parametrize("patch_size_time,patch_size_time_other", [(1, 2), (2, 4)])
def test_prithvi_vit_patch_embed_loading_time_patch_size_other(patch_size_time,patch_size_time_other):
    model = BACKBONE_REGISTRY.build(
        "prithvi_eo_v1_100",
        pretrained=False,
        num_frames=4,
        patch_size=[patch_size_time, 16, 16],
    )

    weights = BACKBONE_REGISTRY.build(
        "prithvi_eo_v1_100",
        pretrained=False,
        num_frames=4,
        patch_size=[patch_size_time_other, 16, 16],
    ).state_dict()

    # assert warning produced
    with pytest.warns(UserWarning):
        select_patch_embed_weights(weights, model, PRETRAINED_BANDS, PRETRAINED_BANDS)

    gc.collect()

@pytest.mark.parametrize("patch_size,patch_size_other", [(2, 4), (4, 8), (16, 4)])
def test_prithvi_vit_patch_embed_loading_not_compatible_patch(patch_size, patch_size_other):
    model = BACKBONE_REGISTRY.build(
        "prithvi_eo_v1_100",
        pretrained=False,
        num_frames=1,
        patch_size=patch_size,
    )

    weights = BACKBONE_REGISTRY.build(
        "prithvi_eo_v1_100",
        pretrained=False,
        num_frames=1,
        patch_size=patch_size_other,
    ).state_dict()

    with pytest.warns(UserWarning):
        select_patch_embed_weights(weights, model, PRETRAINED_BANDS, PRETRAINED_BANDS)

    gc.collect()

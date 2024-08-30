import pytest
import timm

import terratorch
from terratorch.models.backbones.prithvi_select_patch_embed_weights import prithvi_select_patch_embed_weights
from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS


@pytest.mark.parametrize("patch_size", [4, 8, 16])
@pytest.mark.parametrize("tubelet_size,num_frames", [(1, 1), (1, 2), (1, 3), (2, 2), (3,3)])
def test_prithvi_vit_patch_embed_loading_compatible(patch_size, tubelet_size, num_frames):
    model = timm.create_model(
        "prithvi_vit_100",
        pretrained=False,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        features_only=True,
    )

    weights = timm.create_model(
        "prithvi_vit_100",
        pretrained=False,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        features_only=True,
    ).state_dict()

    prithvi_select_patch_embed_weights(weights, model, PRETRAINED_BANDS, PRETRAINED_BANDS)

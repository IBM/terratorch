import pytest
import timm

from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from terratorch.models.backbones.select_patch_embed_weights import select_patch_embed_weights


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

    select_patch_embed_weights(weights, model, PRETRAINED_BANDS, PRETRAINED_BANDS)

@pytest.mark.parametrize("tubelet_size,tubelet_size_other", [(1, 2), (2, 4)])
def test_prithvi_vit_patch_embed_loading_not_compatible_tubelet(tubelet_size, tubelet_size_other):
    model = timm.create_model(
        "prithvi_vit_100",
        pretrained=False,
        num_frames=4,
        patch_size=16,
        tubelet_size=tubelet_size,
        features_only=True,
    )

    weights = timm.create_model(
        "prithvi_vit_100",
        pretrained=False,
        num_frames=4,
        patch_size=16,
        tubelet_size=tubelet_size_other,
        features_only=True,
    ).state_dict()

    # assert warning produced
    with pytest.warns(UserWarning):
        select_patch_embed_weights(weights, model, PRETRAINED_BANDS, PRETRAINED_BANDS)

@pytest.mark.parametrize("patch_size,patch_size_other", [(2, 4), (4, 8), (16, 4)])
def test_prithvi_vit_patch_embed_loading_not_compatible_patch(patch_size, patch_size_other):
    model = timm.create_model(
        "prithvi_vit_100",
        pretrained=False,
        num_frames=1,
        patch_size=patch_size,
        tubelet_size=1,
        features_only=True,
    )

    weights = timm.create_model(
        "prithvi_vit_100",
        pretrained=False,
        num_frames=1,
        patch_size=patch_size_other,
        tubelet_size=1,
        features_only=True,
    ).state_dict()

    with pytest.warns(UserWarning):
        select_patch_embed_weights(weights, model, PRETRAINED_BANDS, PRETRAINED_BANDS)

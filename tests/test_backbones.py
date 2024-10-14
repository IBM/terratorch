# Copyright contributors to the Terratorch project

import pytest
import timm
import torch

from terratorch.models.backbones import scalemae
from terratorch.registry import BACKBONE_REGISTRY

NUM_CHANNELS = 6
NUM_FRAMES = 4


@pytest.fixture
def input_224():
    return torch.ones((1, NUM_CHANNELS, 224, 224))


@pytest.fixture
def input_512():
    return torch.ones((1, NUM_CHANNELS, 512, 512))


@pytest.fixture
def input_224_multitemporal():
    return torch.ones((1, NUM_CHANNELS, NUM_FRAMES, 224, 224))


@pytest.fixture
def input_386():
    return torch.ones((1, NUM_CHANNELS, 386, 386))


@pytest.mark.parametrize("model_name", ["prithvi_vit_100", "prithvi_vit_300", "prithvi_swin_B"])
@pytest.mark.parametrize("test_input", ["input_224", "input_512"])
def test_can_create_backbones_from_timm(model_name, test_input, request):
    backbone = timm.create_model(model_name, pretrained=False)
    input_tensor = request.getfixturevalue(test_input)
    backbone(input_tensor)


@pytest.mark.parametrize("model_name", ["prithvi_vit_100", "prithvi_vit_300", "prithvi_swin_B"])
@pytest.mark.parametrize("test_input", ["input_224", "input_512"])
def test_can_create_backbones_from_timm_features_only(model_name, test_input, request):
    backbone = timm.create_model(model_name, pretrained=False, features_only=True)
    input_tensor = request.getfixturevalue(test_input)
    backbone(input_tensor)

@pytest.mark.parametrize("model_name", ["prithvi_vit_100", "prithvi_vit_300", "prithvi_swin_B"])
@pytest.mark.parametrize("prefix", ["", "timm_"])
def test_can_create_timm_backbones_from_registry(model_name, input_224, prefix):
    backbone = BACKBONE_REGISTRY.build(prefix+model_name, pretrained=False)
    backbone(input_224)


@pytest.mark.parametrize("model_name", ["prithvi_vit_100", "prithvi_vit_300"])
def test_vit_models_accept_multitemporal(model_name, input_224_multitemporal):
    backbone = timm.create_model(model_name, pretrained=False, num_frames=NUM_FRAMES)
    backbone(input_224_multitemporal)


@pytest.mark.parametrize("model_name", ["prithvi_vit_100", "prithvi_vit_300"])
@pytest.mark.parametrize("patch_size", [8, 16])
@pytest.mark.parametrize("tubelet_size", [1, 2, 4])
def test_vit_models_different_patch_tubelet_sizes(model_name, patch_size, tubelet_size, input_224_multitemporal):
    backbone = timm.create_model(
        model_name,
        pretrained=False,
        num_frames=NUM_FRAMES,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        features_only=True,
    )
    embedding = backbone(input_224_multitemporal)
    processed_embedding = backbone.prepare_features_for_image_model(embedding)

    expected_h_w = 224 // patch_size
    expected_t = NUM_FRAMES // tubelet_size

    for e in processed_embedding:
        assert (
            e.shape[-2] == expected_h_w
        ), f"Height {e.shape[-2]} did not match expected height {expected_h_w}"

        assert (
            e.shape[-1] == expected_h_w
        ), f"Width {e.shape[-1]} did not match expected width {expected_h_w}"

        assert (
            e.shape[1] == expected_t * backbone.embed_dim
        ), f"Expected embedding dimension to be of size effective time {expected_t} x embedding dimension\
        {backbone.embed_dim} = {expected_t * backbone.embed_dim} but was {e.shape[1]}"


@pytest.mark.parametrize("model_name", ["prithvi_vit_100", "prithvi_vit_300"])
def test_out_indices(model_name, input_224):
    out_indices = [2, 4, 8, 10]
    backbone = timm.create_model(model_name, pretrained=False, features_only=True, out_indices=out_indices)
    assert backbone.feature_info.out_indices == out_indices

    output = backbone(input_224)
    full_output = backbone.forward_features(input_224)

    for filtered_index, full_index in enumerate(out_indices):
        assert torch.allclose(full_output[full_index], output[filtered_index])

@pytest.mark.parametrize("model_name", ["vit_base_patch16", "vit_large_patch16"])
def test_scale_mae(model_name):
    out_indices = [2, 4, 8, 10]

    # default should have 3 channels
    backbone = scalemae.create_model(model_name, out_indices=out_indices)
    input_tensor = torch.ones((1, 3, 224, 224))
    output = backbone(input_tensor)

    assert len(output) == len(out_indices)

@pytest.mark.parametrize("model_name", ["vit_base_patch16", "vit_large_patch16"])
@pytest.mark.parametrize("bands", [2, 4, 6])
def test_scale_mae_new_channels(model_name, bands):

    backbone = scalemae.create_model(model_name, bands=list(range(bands)))
    input_tensor = torch.ones((1, bands, 224, 224))
    backbone(input_tensor)


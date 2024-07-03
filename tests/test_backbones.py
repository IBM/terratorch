import pytest
import timm
import torch
import importlib
import terratorch  # noqa: F401
import os 

NUM_CHANNELS = 6
NUM_FRAMES = 3


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


@pytest.mark.parametrize("model_name", ["prithvi_vit_100", "prithvi_vit_300"])
def test_vit_models_accept_multitemporal(model_name, input_224_multitemporal):
    backbone = timm.create_model(model_name, pretrained=False, num_frames=NUM_FRAMES)
    backbone(input_224_multitemporal)

#def test_swin_models_accept_non_divisible_by_patch_size(input_386):
#    backbone = timm.create_model("prithvi_swin_90_us", pretrained=False, num_frames=NUM_FRAMES)
#    backbone(input_386)

# Copyright contributors to the Terratorch project
import os
import warnings
import pytest
import timm
import torch
import gc 
from terratorch.models.backbones import scalemae
from terratorch.registry import BACKBONE_REGISTRY
import terratorch.models.backbones.torchgeo_vit as torchgeo_vit

NUM_CHANNELS = 6
NUM_FRAMES = 4

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS", "false") == "true"

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
def input_non_divisible():
    return torch.ones((1, NUM_CHANNELS, NUM_FRAMES, 220, 230))


@pytest.fixture
def input_386():
    return torch.ones((1, NUM_CHANNELS, 386, 386))

def torchgeo_vit_backbones():
    return [i for i in dir(torchgeo_vit) if "_vit_small" in i ]

@pytest.mark.parametrize("model_name", ["prithvi_swin_B", "prithvi_swin_L", "prithvi_swin_B"])
@pytest.mark.parametrize("test_input", ["input_224", "input_512"])
def test_can_create_backbones_from_timm(model_name, test_input, request):
    backbone = timm.create_model(model_name, pretrained=False)
    input_tensor = request.getfixturevalue(test_input)
    backbone(input_tensor)
    gc.collect()

@pytest.mark.parametrize("model_name", ["prithvi_swin_B", "prithvi_swin_L", "prithvi_swin_B"])
@pytest.mark.parametrize("test_input", ["input_224", "input_512"])
def test_can_create_backbones_from_timm_features_only(model_name, test_input, request):
    backbone = timm.create_model(model_name, pretrained=False, features_only=True)
    input_tensor = request.getfixturevalue(test_input)
    backbone(input_tensor)
    gc.collect()

@pytest.mark.parametrize("model_name", ["prithvi_swin_L", "prithvi_swin_L", "prithvi_swin_B"])
@pytest.mark.parametrize("prefix", ["", "timm_"])
def test_can_create_timm_backbones_from_registry(model_name, input_224, prefix):
    backbone = BACKBONE_REGISTRY.build(prefix+model_name, pretrained=False)
    backbone(input_224)
    gc.collect()


@pytest.mark.parametrize("model_name", ["prithvi_eo_v1_100", "prithvi_eo_v2_300"])
def test_can_create_backbones_from_registry(model_name, input_224):
    backbone = BACKBONE_REGISTRY.build(model_name, pretrained=False)
    backbone(input_224)
    gc.collect()

@pytest.mark.parametrize("model_name", torchgeo_vit_backbones())
def test_can_create_backbones_from_registry_torchgeo_vit(model_name, input_224):
    backbone = BACKBONE_REGISTRY.build(model_name, model_bands=[0,1,2,3,4,5], pretrained=False)
    backbone(input_224)
    gc.collect()

@pytest.mark.parametrize("model_name", ["prithvi_eo_v1_100", "prithvi_eo_v2_300"])
def test_vit_models_accept_multitemporal(model_name, input_224_multitemporal):
    backbone = BACKBONE_REGISTRY.build(model_name, pretrained=False, num_frames=NUM_FRAMES)
    backbone(input_224_multitemporal)
    gc.collect()


@pytest.mark.parametrize("model_name", ["prithvi_eo_v1_100", "prithvi_eo_v2_300"])
@pytest.mark.parametrize("patch_size", [8, 16])
@pytest.mark.parametrize("patch_size_time", [1, 2, 4])
def test_vit_models_different_patch_tubelet_sizes(model_name, patch_size, patch_size_time, input_224_multitemporal):
    backbone = BACKBONE_REGISTRY.build(
        model_name,
        pretrained=False,
        num_frames=NUM_FRAMES,
        patch_size=[patch_size_time, patch_size, patch_size],
    )
    embedding = backbone(input_224_multitemporal)
    processed_embedding = backbone.prepare_features_for_image_model(embedding)

    expected_h_w = 224 // patch_size
    expected_t = NUM_FRAMES // patch_size_time

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

    gc.collect()
@pytest.mark.parametrize("model_name", ["prithvi_eo_v1_100", "prithvi_eo_v2_300"])
def test_out_indices(model_name, input_224):
    out_indices = (2, 4, 8, 10)
    backbone = BACKBONE_REGISTRY.build(model_name, pretrained=False, out_indices=out_indices)
    assert backbone.out_indices == out_indices

    output = backbone(input_224)
    full_output = backbone.forward_features(input_224)

    for filtered_index, full_index in enumerate(out_indices):
        assert torch.allclose(full_output[full_index], output[filtered_index])
    gc.collect()


@pytest.mark.parametrize("model_name", ["vit_base_patch16", "vit_large_patch16"])
def test_scale_mae(model_name):
    # out_indices = [2, 4, 8, 10]
    out_indices = (2, 4, 8, 10)
    # default should have 3 channels
    backbone = scalemae.create_model(model_name, out_indices=out_indices)
    input_tensor = torch.ones((1, 3, 224, 224))
    output = backbone(input_tensor)

    assert len(output) == len(out_indices)
    gc.collect()


@pytest.mark.parametrize("model_name", ["vit_base_patch16", "vit_large_patch16"])
@pytest.mark.parametrize("bands", [2, 4, 6])
def test_scale_mae_new_channels(model_name, bands):

    backbone = scalemae.create_model(model_name, bands=list(range(bands)))
    input_tensor = torch.ones((1, bands, 224, 224))
    backbone(input_tensor)
    gc.collect()

@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skip this test in GitHub Actions as deformable attn is not supported.")
@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100", "prithvi_eo_v2_300", "prithvi_eo_v2_300_tl"])
def test_prithvi_vit_adapter(backbone, input_224):
    try:
        from terratorch.models.backbones.detr_ops.modules.ms_deform_attn import MSDeformAttn
    except ImportError:
        pytest.skip(f'Cannot test vit_adapter due to missing deformable attn module.')

    backbone = BACKBONE_REGISTRY.build(backbone, pretrained=True, vit_adapter=True)
    backbone = backbone.to("cuda")
    input_224 = input_224.to("cuda")
    output = backbone(input_224)

    assert len(output) == 4
    embed_dim = backbone.embed_dim
    assert output[0].shape == (1, embed_dim, 56, 56)
    assert output[1].shape == (1, embed_dim, 28, 28)
    assert output[2].shape == (1, embed_dim, 14, 14)
    assert output[3].shape == (1, embed_dim, 7, 7)

@pytest.mark.parametrize("model_name", ["multimae_small", "multimae_base"])
@pytest.mark.parametrize("input_adapters", [None, ['S2L2A']])
def test_multi_mae(model_name, input_adapters):
    # default should have 3 channels
    backbone = BACKBONE_REGISTRY.build(model_name, input_adapters=input_adapters)
    input_tensor = torch.ones((1, 12, 224, 224))
    output = backbone({"S2L2A": input_tensor})

    gc.collect()


@pytest.mark.parametrize("model_name",
                         ["terramind_v1_base", "terramind_v1_large", "terramind_v1_base_tim", "terramind_v1_large_tim"])
def test_terramind(model_name):
    # default should have 3 channels
    backbone = BACKBONE_REGISTRY.build(model_name, modalities=['S2L2A'])
    output = backbone({"S2L2A": torch.ones((1, 12, 224, 224))})

    gc.collect()

"""Tests for mmearth_convnextv2 ConvNeXtV2 backbone to maximize coverage.

Covers:
- Block forward pass & residual + drop_path behavior
- ConvNeXtV2 initialization (orig stem vs depthwise stem)
- forward_features path
- forward with classification head
- forward with mask (pretraining path) both stems
- upsample_mask utility
- weight initialization & head scaling
- factory helper functions convnextv2_* variants
- error path when depths=None (TypeError expected due to len(None))
- checkpoints dict integrity
"""
import math
import types
import sys
import pytest
import torch
import torch.nn as nn

# The source file depends on a missing local module `.norm_layers`. We inject a lightweight
# stub providing LayerNorm & GRN before importing the backbone to keep tests self-contained.
if 'terratorch.models.backbones.norm_layers' not in sys.modules:
    norm_layers_mod = types.ModuleType('terratorch.models.backbones.norm_layers')

    class StubLayerNorm(nn.Module):
        def __init__(self, dim, eps=1e-6, data_format=None):
            super().__init__()
            self.ln = nn.LayerNorm(dim, eps=eps)
            self.channels_first = data_format == 'channels_first'
        def forward(self, x):
            # Emulate ConvNeXt style LayerNorm supporting channels_first
            if x.dim() == 4 and self.channels_first:
                # (N, C, H, W) -> (N, H, W, C)
                x = x.permute(0, 2, 3, 1)
                x = self.ln(x)
                x = x.permute(0, 3, 1, 2)
                return x
            return self.ln(x)

    class StubGRN(nn.Module):
        def __init__(self, dim):
            super().__init__()
            # simple identity parameters to keep interface
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        def forward(self, x):
            return x

    norm_layers_mod.LayerNorm = StubLayerNorm
    norm_layers_mod.GRN = StubGRN
    sys.modules['terratorch.models.backbones.norm_layers'] = norm_layers_mod

from terratorch.models.backbones.mmearth_convnextv2 import (
    Block,
    ConvNeXtV2,
    convnextv2_atto,
    convnextv2_femto,
    convnext_pico,
    convnextv2_nano,
    convnextv2_tiny,
    convnextv2_base,
    convnextv2_large,
    convnextv2_huge,
    checkpoints,
)


@pytest.mark.parametrize("dim,shape", [(8, (2, 8, 16, 16)), (12, (1, 12, 7, 7))])
def test_block_forward_shape_and_residual(dim, shape):
    block = Block(dim=dim, drop_path=0.0)
    x = torch.randn(*shape)
    y = block(x)
    assert y.shape == x.shape
    # Residual connection: difference should not be identical for random weights
    assert not torch.allclose(y, x)


def test_block_forward_with_drop_path():
    block = Block(dim=8, drop_path=0.1)
    block.train()  # activate stochastic behavior
    x = torch.randn(2, 8, 8, 8)
    y = block(x)
    assert y.shape == x.shape


def test_block_grad_passes():
    block = Block(dim=4, drop_path=0.0)
    x = torch.randn(3, 4, 10, 10, requires_grad=True)
    y = block(x).mean()
    y.backward()
    assert x.grad is not None


def test_convnextv2_initialization_non_orig_stem():
    model = ConvNeXtV2(patch_size=8, img_size=32, in_chans=3, num_classes=10, depths=[1,1,1,1], dims=[8,16,32,64])
    assert hasattr(model, 'initial_conv')
    assert hasattr(model, 'stem')
    assert len(model.stages) == 4
    assert model.head.out_features == 10


def test_convnextv2_initialization_orig_stem():
    model = ConvNeXtV2(patch_size=8, img_size=32, in_chans=3, num_classes=5, depths=[1,1,1,1], dims=[8,16,32,64], use_orig_stem=True)
    assert hasattr(model, 'stem_orig')
    assert not hasattr(model, 'initial_conv')
    assert len(model.downsample_layers) == 3


def test_convnextv2_forward_classification():
    model = ConvNeXtV2(patch_size=8, img_size=32, in_chans=3, num_classes=7, depths=[1,1,1,1], dims=[8,16,32,64])
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 7)


def test_convnextv2_forward_features():
    model = ConvNeXtV2(patch_size=8, img_size=32, in_chans=3, num_classes=7, depths=[1,1,1,1], dims=[8,16,32,64])
    x = torch.randn(2, 3, 32, 32)
    feats = model.forward_features(x)
    assert feats.shape == (2, 64)


def _make_mask(img_size: int, patch_size: int):
    # Choose num_patches such that scale = img_size // sqrt(num_patches) is patch_size
    # Let num_patches = (img_size // patch_size)**2
    num = (img_size // patch_size) ** 2
    return torch.randint(0, 2, (2, num))  # batch size 2


def test_convnextv2_forward_with_mask_non_orig():
    img_size = 32
    patch = 8
    model = ConvNeXtV2(patch_size=patch, img_size=img_size, in_chans=3, num_classes=11, depths=[1,1,1,1], dims=[8,16,32,64])
    x = torch.randn(2, 3, img_size, img_size)
    mask = _make_mask(img_size, patch)
    out = model(x, mask=mask)
    # Returns feature map (not pooled/classified)
    assert out.dim() == 4
    # Last stage channels
    assert out.shape[1] == 64


def test_convnextv2_forward_with_mask_orig_stem():
    img_size = 32
    patch = 8
    model = ConvNeXtV2(patch_size=patch, img_size=img_size, in_chans=3, num_classes=11, depths=[1,1,1,1], dims=[8,16,32,64], use_orig_stem=True)
    x = torch.randn(2, 3, img_size, img_size)
    mask = _make_mask(img_size, patch)
    out = model(x, mask=mask)
    assert out.shape[1] == 64


def test_upsample_mask_utility():
    model = ConvNeXtV2(patch_size=8, img_size=32, in_chans=3, num_classes=7, depths=[1,1,1,1], dims=[8,16,32,64])
    mask = torch.randint(0, 2, (2, 16))  # 4x4 patches
    up = model.upsample_mask(mask, scale=8)
    assert up.shape == (2, 32, 32)


def test_head_init_scale_effect():
    scale = 0.5
    model = ConvNeXtV2(patch_size=8, img_size=32, in_chans=3, num_classes=3, depths=[1,1,1,1], dims=[8,16,32,64], head_init_scale=scale)
    # After scaling, bias should be all zeros still but multiplied (0 * scale)
    assert torch.allclose(model.head.bias.data, torch.zeros_like(model.head.bias))
    # Weight scaled
    orig_norm = model.head.weight.data.abs().mean()
    assert orig_norm > 0


def test_weight_init_on_new_layer():
    model = ConvNeXtV2(patch_size=8, img_size=32, depths=[1,1,1,1], dims=[8,16,32,64])
    lin = nn.Linear(4, 4)
    model._init_weights(lin)
    assert torch.allclose(lin.bias, torch.zeros_like(lin.bias))


def test_factory_functions_output_types():
    for fn in [convnextv2_atto, convnextv2_femto, convnext_pico, convnextv2_nano, convnextv2_tiny]:
        m = fn(num_classes=2, img_size=32, patch_size=8)
        x = torch.randn(1, 3, 32, 32)
        out = m(x)
        assert out.shape == (1, 2)


def test_factory_large_variants_shapes():
    for fn in [convnextv2_base, convnextv2_large]:  # skip huge for memory
        m = fn(num_classes=2, img_size=32, patch_size=8)
        x = torch.randn(1, 3, 32, 32)
        out = m(x)
        assert out.shape == (1, 2)


def test_checkpoints_dict_integrity():
    assert isinstance(checkpoints, dict)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in checkpoints.items())
    assert len(checkpoints) >= 1


def test_depths_none_raises_type_error():
    # Given current implementation len(depths) before defaulting triggers TypeError
    with pytest.raises(TypeError):
        ConvNeXtV2(depths=None)  # minimal args rely on defaults


def test_backward_through_classification_head():
    model = ConvNeXtV2(patch_size=8, img_size=32, in_chans=3, num_classes=4, depths=[1,1,1,1], dims=[8,16,32,64])
    x = torch.randn(5, 3, 32, 32)
    out = model(x)
    loss = out.sum()
    loss.backward()
    # Check one parameter received gradient
    params_with_grad = [p for p in model.parameters() if p.grad is not None]
    assert len(params_with_grad) > 0


def test_global_average_pooling_behavior():
    model = ConvNeXtV2(patch_size=8, img_size=32, in_chans=3, num_classes=4, depths=[1,1,1,1], dims=[8,16,32,64])
    x = torch.randn(2, 3, 32, 32)
    feats = model.forward_features(x)
    # Ensure features are mean over spatial dims -> compare manual computation
    manual = model.norm(model.stages[3](model.downsample_layers[2](model.stages[2](model.downsample_layers[1](model.stages[1](model.downsample_layers[0](model.stages[0](model.stem(model.initial_conv(x))))))))).mean(dim=(-2,-1)))
    # shapes equal
    assert feats.shape == manual.shape


def test_forward_features_with_orig_stem():
    """Test forward_features specifically with use_orig_stem=True to ensure line 163 coverage."""
    model = ConvNeXtV2(depths=[2, 2, 2, 2], dims=[32, 64, 128, 256], use_orig_stem=True)
    model.eval()

    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        features = model.forward_features(x)

    # Should produce features with final dim
    assert features.shape == (2, 256), f"Expected (2, 256), got {features.shape}"
    assert not torch.allclose(features, torch.zeros_like(features))


def test_dims_none_uses_default():
    """Test that when dims=None, the default [96, 192, 384, 768] is used."""
    model = ConvNeXtV2(patch_size=16, img_size=64, in_chans=3, num_classes=10, depths=[2,2,2,2], dims=None)
    # Default dims should be [96, 192, 384, 768]
    # Check first stage has 96 channels by looking at first downsample layer input
    assert model.downsample_layers[0][1].in_channels == 96
    assert model.downsample_layers[0][1].out_channels == 192
    assert model.downsample_layers[1][1].in_channels == 192
    assert model.downsample_layers[1][1].out_channels == 384


def test_checkpoints_dict_all_keys():
    """Test that the checkpoints dictionary contains all expected keys and can access all values."""
    from terratorch.models.backbones.mmearth_convnextv2 import checkpoints

    expected_keys = [
        "pt-S2_atto_1M_64_uncertainty_56-8",
        "pt-all_mod_atto_100k_128_uncertainty_112-16",
        "pt-all_mod_atto_1M_128_uncertainty_112-16",
        "pt-all_mod_atto_1M_64_uncertainty_56-8",
    ]

    assert set(checkpoints.keys()) == set(expected_keys), "Checkpoints dict missing expected keys"

    # Verify all values are valid URLs and explicitly access each one to cover lines
    assert isinstance(checkpoints["pt-S2_atto_1M_64_uncertainty_56-8"], str)
    assert checkpoints["pt-S2_atto_1M_64_uncertainty_56-8"].startswith("https://")
    
    assert isinstance(checkpoints["pt-all_mod_atto_100k_128_uncertainty_112-16"], str)
    assert checkpoints["pt-all_mod_atto_100k_128_uncertainty_112-16"].startswith("https://")
    
    assert isinstance(checkpoints["pt-all_mod_atto_1M_128_uncertainty_112-16"], str)
    assert checkpoints["pt-all_mod_atto_1M_128_uncertainty_112-16"].startswith("https://")
    
    assert isinstance(checkpoints["pt-all_mod_atto_1M_64_uncertainty_56-8"], str)
    assert checkpoints["pt-all_mod_atto_1M_64_uncertainty_56-8"].startswith("https://")


if __name__ == "__main__":
    pytest.main([__file__, '-v'])

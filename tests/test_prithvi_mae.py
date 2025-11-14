import pytest
import torch
import numpy as np
from torch import nn
import terratorch.models.backbones.prithvi_mae as mae

from terratorch.models.backbones.prithvi_mae import (
    get_1d_sincos_pos_embed_from_grid,
    get_3d_sincos_pos_embed,
    _get_1d_sincos_embed_from_grid_torch,
    _interpolate_pos_encoding,
    PatchEmbed,
    TemporalEncoder,
    LocationEncoder,
    PrithviViT,
)


def test_get_1d_sincos_pos_embed_from_grid():
    pos = np.arange(5)
    emb = get_1d_sincos_pos_embed_from_grid(8, pos)
    assert emb.shape == (5, 8)


def test_get_1d_sincos_pos_embed_from_grid_torch():
    pos = torch.arange(5, dtype=torch.float32)
    emb = _get_1d_sincos_embed_from_grid_torch(8, pos)
    assert emb.shape == (5, 8)
    assert emb.dtype == torch.float32


def test_get_3d_sincos_pos_embed():
    emb = get_3d_sincos_pos_embed(32, (2, 2, 2), add_cls_token=True)
    assert emb.shape == (1 + 8, 32)


def test_interpolate_pos_encoding_changes_shape():
    embed_dim = 32
    pos = torch.zeros(1, 1 + 2*2*2, embed_dim)
    out = _interpolate_pos_encoding(
        pos_embed=pos,
        grid_size=[2,2,2],
        patch_size=[1,1,1],
        shape=(2,4,4),   # change only H,W → interpolation path
        embed_dim=embed_dim,
    )
    assert out.shape[1] != pos.shape[1]


def test_patch_embed_forward_warns():
    x = torch.randn(1, 3, 1, 20, 20)  # not divisible → warning
    patch = PatchEmbed(
        input_size=(1, 20, 20),
        patch_size=(1, 16, 16),
        in_chans=3,
        embed_dim=32,
        norm_layer=nn.LayerNorm,
    )
    with pytest.warns(UserWarning):
        out = patch(x)
    assert out.shape[0] == 1


def test_patch_embed_forward_ok():
    x = torch.randn(1, 3, 1, 16, 16)
    patch = PatchEmbed(
        input_size=(1, 16, 16),
        patch_size=(1, 16, 16),
        in_chans=3,
        embed_dim=32,
    )
    out = patch(x)
    assert out.shape == (1, 1, 32)


def test_temporal_encoder():
    enc = TemporalEncoder(32, trainable_scale=True)
    coords = torch.tensor([[[2020, 150]]], dtype=torch.float32)  # B=1,T=1
    out = enc(coords, tokens_per_frame=2)
    assert out.shape == (1, 2, 32)


def test_location_encoder():
    enc = LocationEncoder(32, trainable_scale=True)
    coords = torch.tensor([[47.5, 8.5]], dtype=torch.float32)
    out = enc(coords)
    assert out.shape == (1, 1, 32)


def test_prithvi_forward_no_vpt():
    model = PrithviViT(
        img_size=16,
        patch_size=(1, 8, 8),
        num_frames=1,
        in_chans=3,
        embed_dim=32,
        depth=2,
        num_heads=4,
        coords_encoding=["time", "location"],
        coords_scale_learn=True,
    )
    x = torch.randn(1, 3, 16, 16)
    temporal = torch.tensor([[[2020, 100]]], dtype=torch.float32)
    location = torch.tensor([[47.5, 8.5]], dtype=torch.float32)

    out, mask, ids_restore = model(
        x,
        temporal_coords=temporal,
        location_coords=location,
        mask_ratio=0.5,
    )
    assert out.ndim == 3
    assert mask.ndim == 2
    assert ids_restore.ndim == 2


def test_prithvi_forward_features():
    model = PrithviViT(
        img_size=16,
        patch_size=(1, 8, 8),
        num_frames=1,
        in_chans=3,
        embed_dim=32,
        depth=3,
        num_heads=4,
    )
    x = torch.randn(1, 3, 16, 16)
    features = model.forward_features(x)
    assert len(features) == 3
    assert features[-1].shape[-1] == 32


def test_prithvi_random_masking_deterministic():
    model = PrithviViT(
        img_size=16,
        patch_size=(1, 8, 8),
        num_frames=1,
        in_chans=3,
        embed_dim=32,
        depth=1,
        num_heads=4,
    )
    seq = torch.randn(1, 4, 32)
    noise = torch.tensor([[0.1, 0.5, 0.9, 0.2]])
    out, mask, ids = model.random_masking(seq, mask_ratio=0.5, noise=noise)

    assert out.shape[1] == 2
    assert mask.shape == (1, 4)
    assert ids.shape == (1, 4)


def test_prithvi_vpt_forward():
    model = PrithviViT(
        img_size=16,
        patch_size=(1, 8, 8),
        num_frames=1,
        in_chans=3,
        embed_dim=32,
        depth=2,
        num_heads=4,
        vpt=True,
        vpt_n_tokens=2,
        vpt_dropout=0.1,
    )
    x = torch.randn(1, 3, 16, 16)
    out, mask, ids = model(x)
    assert out.ndim == 3

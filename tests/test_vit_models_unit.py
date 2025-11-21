import math
import torch
import pytest
from functools import partial

from terratorch.models.backbones.terramind.tokenizer.models.vit_models import (
    pair,
    build_2d_sincos_posemb,
    trunc_normal_,
    drop_path,
    DropPath,
    Mlp,
    Attention,
    CrossAttention,
    Block,
    DecoderBlock,
    LayerNorm,
    ConvNeXtBlock,
    ViTEncoder,
    ViTDecoder,
    vit_s_enc,
    vit_b_enc,
    vit_l_enc,
    vit_s_dec,
    vit_b_dec,
    vit_l_dec,
)


def test_pair():
    assert pair(4) == (4, 4)
    assert pair((2, 3)) == (2, 3)


def test_build_2d_sincos_posemb_shape():
    emb = build_2d_sincos_posemb(2, 2, embed_dim=32)
    assert emb.shape == (1, 32, 2, 2)


def test_trunc_normal_warns():
    t = torch.empty(10)
    # Force warning: mean very far from a,b range
    with pytest.warns(UserWarning):
        trunc_normal_(t, mean=10.0, std=1.0, a=-2.0, b=2.0)
    assert (t >= -2).all() and (t <= 2).all()


def test_drop_path_function_training():
    x = torch.ones(4, 3)
    out = drop_path(x, drop_prob=0.5, training=True)
    # Some rows may be scaled but shape preserved
    assert out.shape == x.shape


def test_drop_path_module_eval():
    m = DropPath(0.5)
    m.eval()
    x = torch.randn(2, 3, 4)
    assert torch.equal(m(x), x)


def test_drop_path_extra_repr():
    m = DropPath(0.3)
    assert 'p=0.3' in m.extra_repr()


def test_mlp_forward():
    mlp = Mlp(16, hidden_features=32, out_features=8, drop=0.1)
    x = torch.randn(4, 16)
    y = mlp(x)
    assert y.shape == (4, 8)


def test_attention_forward():
    attn = Attention(dim=32, num_heads=4)
    x = torch.randn(2, 5, 32)
    y = attn(x)
    assert y.shape == (2, 5, 32)


def test_cross_attention_forward():
    attn = CrossAttention(dim=32, num_heads=4)
    x = torch.randn(2, 5, 32)
    context = torch.randn(2, 7, 32)
    y = attn(x, context)
    assert y.shape == (2, 5, 32)


def test_block_forward():
    block = Block(dim=32, num_heads=4, mlp_ratio=2.0, drop_path=0.1)
    x = torch.randn(2, 5, 32)
    y = block(x)
    assert y.shape == x.shape


def test_decoder_block_forward():
    block = DecoderBlock(dim=32, num_heads=4, mlp_ratio=2.0, drop_path=0.1)
    x = torch.randn(2, 5, 32)
    context = torch.randn(2, 7, 32)
    y = block(x, context)
    assert y.shape == x.shape


def test_layernorm_channels_last():
    ln = LayerNorm(8, data_format="channels_last")
    x = torch.randn(2, 4, 4, 8)  # (N,H,W,C)
    y = ln(x)
    assert y.shape == x.shape


def test_layernorm_channels_first():
    ln = LayerNorm(8, data_format="channels_first")
    x = torch.randn(2, 8, 4, 4)  # (N,C,H,W)
    y = ln(x)
    assert y.shape == x.shape


def test_layernorm_invalid_format_raises():
    with pytest.raises(NotImplementedError):
        LayerNorm(8, data_format="invalid_format")


def test_convnext_block_forward():
    block = ConvNeXtBlock(dim=8, drop_path=0.1)
    x = torch.randn(2, 8, 8, 8)
    y = block(x)
    assert y.shape == x.shape


def test_convnext_block_no_gamma():
    # layer_scale_init_value <= 0 means no gamma parameter
    block = ConvNeXtBlock(dim=8, layer_scale_init_value=0.0)
    x = torch.randn(1, 8, 4, 4)
    y = block(x)
    assert y.shape == x.shape
    assert block.gamma is None

# Encoder tests

def test_vit_encoder_basic_forward():
    enc = ViTEncoder(in_channels=3, patch_size=4, resolution=8, dim_tokens=32, depth=2, num_heads=4)
    x = torch.randn(2, 3, 8, 8)
    y = enc(x)
    assert y.shape == (2, 32, 2, 2)


def test_vit_encoder_return_intermediates():
    enc = ViTEncoder(in_channels=3, patch_size=4, resolution=8, dim_tokens=32, depth=2, num_heads=4)
    x = torch.randn(1, 3, 8, 8)
    outs = enc(x, return_intermediates=True)
    assert isinstance(outs, list) and len(outs) == 2
    assert outs[-1].shape == (1, 32, 2, 2)


def test_vit_encoder_one_hot_from_indexes():
    enc = ViTEncoder(in_channels=5, patch_size=4, resolution=8, dim_tokens=32, depth=1, num_heads=4)
    x = torch.randint(0, 5, (2, 8, 8))  # shape [B,H,W]
    y = enc(x)
    assert y.shape == (2, 32, 2, 2)


def test_vit_encoder_one_hot_from_single_channel():
    enc = ViTEncoder(in_channels=4, patch_size=4, resolution=8, dim_tokens=32, depth=1, num_heads=4)
    x = torch.randint(0, 4, (2, 1, 8, 8))
    y = enc(x)
    assert y.shape == (2, 32, 2, 2)


def test_vit_encoder_patch_proj_disabled():
    # Provide already token grid shape [B,C,N_H,N_W]
    enc = ViTEncoder(in_channels=3, patch_size=4, resolution=8, dim_tokens=32, depth=1, num_heads=4, patch_proj=False)
    x = torch.randn(2, 3, 2, 2)
    y = enc(x)
    assert y.shape == (2, 32, 2, 2)


def test_vit_encoder_post_mlp():
    enc = ViTEncoder(in_channels=3, patch_size=4, resolution=8, dim_tokens=32, depth=1, num_heads=4, post_mlp=True)
    x = torch.randn(1, 3, 8, 8)
    y = enc(x)
    assert y.shape == (1, 32, 2, 2)


def test_vit_encoder_learnable_pos_emb():
    enc = ViTEncoder(in_channels=3, patch_size=4, resolution=8, dim_tokens=32, depth=1, num_heads=4, 
                     sincos_pos_emb=False, learnable_pos_emb=True)
    x = torch.randn(1, 3, 8, 8)
    y = enc(x)
    assert y.shape == (1, 32, 2, 2)
    assert enc.pos_emb.requires_grad


def test_vit_encoder_invalid_channels_raises():
    enc = ViTEncoder(in_channels=3, patch_size=4, resolution=8, dim_tokens=32, depth=1, num_heads=4)
    # shape [B,C,H,W] but C != in_channels and not one-hot convertible
    x = torch.randn(1, 2, 8, 8)
    with pytest.raises(ValueError):
        enc(x)

# Checkpoint load test

def test_vit_encoder_checkpoint_loading(tmp_path):
    N_H = 2; N_W = 2; dim = 32
    ckpt = {
        'model': {
            'pos_embed': torch.randn(1, 1 + N_H * N_W, dim),
            'patch_embed.proj.weight': torch.randn(dim, 3, 4, 4),
            'patch_embed.proj.bias': torch.randn(dim),
        }
    }
    ckpt_file = tmp_path / 'ckpt.pt'
    torch.save(ckpt, ckpt_file)
    enc = ViTEncoder(in_channels=3, patch_size=4, resolution=8, dim_tokens=dim, depth=1, num_heads=4, ckpt_path=str(ckpt_file))
    x = torch.randn(1, 3, 8, 8)
    y = enc(x)
    assert y.shape == (1, dim, 2, 2)

# Decoder tests

def test_vit_decoder_basic_forward():
    dec = ViTDecoder(out_channels=3, patch_size=4, resolution=8, dim_tokens=32, depth=2, num_heads=4)
    x = torch.randn(2, 32, 2, 2)
    y = dec(x)
    assert y.shape == (2, 3, 8, 8)


def test_vit_decoder_patch_proj_disabled():
    dec = ViTDecoder(out_channels=3, patch_size=4, resolution=8, dim_tokens=32, depth=1, num_heads=4, patch_proj=False)
    x = torch.randn(1, 32, 2, 2)
    y = dec(x)
    assert y.shape == (1, 3, 2, 2)


def test_vit_decoder_post_mlp():
    dec = ViTDecoder(out_channels=3, patch_size=4, resolution=8, dim_tokens=32, depth=1, num_heads=4, post_mlp=True)
    x = torch.randn(1, 32, 2, 2)
    y = dec(x)
    assert y.shape == (1, 3, 8, 8)


def test_vit_decoder_out_conv():
    dec = ViTDecoder(out_channels=4, patch_size=4, resolution=8, dim_tokens=32, depth=1, num_heads=4, out_conv=True)
    x = torch.randn(1, 32, 2, 2)
    y = dec(x)
    assert y.shape == (1, 4, 8, 8)


def test_vit_decoder_learnable_pos_emb():
    dec = ViTDecoder(out_channels=3, patch_size=4, resolution=8, dim_tokens=32, depth=1, num_heads=4,
                     sincos_pos_emb=False, learnable_pos_emb=True)
    x = torch.randn(1, 32, 2, 2)
    y = dec(x)
    assert y.shape == (1, 3, 8, 8)
    assert dec.pos_emb.requires_grad

# Preset constructors

def test_preset_vit_s_enc_instantiation():
    m = vit_s_enc(in_channels=3, patch_size=4, resolution=8)
    assert m.get_num_layers() == 8


def test_preset_vit_s_dec_instantiation():
    m = vit_s_dec(out_channels=3, patch_size=4, resolution=8)
    assert m.get_num_layers() == 8


def test_preset_vit_b_enc_instantiation():
    m = vit_b_enc(in_channels=3, patch_size=4, resolution=8)
    assert m.get_num_layers() == 12


def test_preset_vit_b_dec_instantiation():
    m = vit_b_dec(out_channels=3, patch_size=4, resolution=8)
    assert m.get_num_layers() == 12


def test_preset_vit_l_enc_instantiation():
    m = vit_l_enc(in_channels=3, patch_size=4, resolution=8)
    assert m.get_num_layers() == 24


def test_preset_vit_l_dec_instantiation():
    m = vit_l_dec(out_channels=3, patch_size=4, resolution=8)
    assert m.get_num_layers() == 24

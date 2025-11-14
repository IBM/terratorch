"""Max coverage tests for uvit.py core modules and UViT model.
Small tensor sizes chosen for speed.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from terratorch.models.backbones.terramind.tokenizer.models.uvit import (
    modulate,
    pair,
    build_2d_sincos_posemb,
    drop_path,
    DropPath,
    Mlp,
    Attention,
    CrossAttention,
    Block,
    DecoderBlock,
    TransformerConcatCond,
    TransformerXattnCond,
    UViT,
    uvit_b_p4_f16,
    uvit_b_p4_f16_longskip,
    uvit_b_p4_f8,
)


def test_pair():
    assert pair(4) == (4,4)
    assert pair((2,3)) == (2,3)


def test_modulate():
    x = torch.ones(2,5,4)
    shift = torch.zeros(2,4)
    scale = torch.ones(2,4)
    y = modulate(x, shift, scale)
    assert torch.allclose(y, x * 2)


def test_build_2d_sincos_posemb_shape():
    pos = build_2d_sincos_posemb(4,4,embed_dim=16)
    assert pos.shape == (1,16,4,4)


def test_drop_path_training():
    x = torch.randn(3,4)
    out = drop_path(x, drop_prob=0.0, training=True)
    assert torch.allclose(out, x)


def test_droppath_module():
    m = DropPath(0.0)
    m.train()
    x = torch.randn(2,3,4)
    assert torch.allclose(m(x), x)


def test_mlp_with_temb():
    mlp = Mlp(8, temb_dim=6, hidden_features=16)
    x = torch.randn(2,5,8)
    temb = torch.randn(2,6)
    out = mlp(x, temb)
    assert out.shape == (2,5,8)


def test_attention_mask():
    attn = Attention(8, num_heads=2)
    x = torch.randn(2,5,8)
    mask = torch.zeros(2,5,5,dtype=torch.bool)
    mask[:,2:] = True  # mask some positions
    out = attn(x, mask=mask)
    assert out.shape == x.shape


def test_cross_attention_mask():
    attn = CrossAttention(8, dim_context=8, num_heads=2)
    q = torch.randn(2,4,8)
    context = torch.randn(2,5,8)
    mask = torch.ones(2,4,5,dtype=torch.bool)
    out = attn(q, context, mask=mask)
    assert out.shape == (2,4,8)


def test_block_variants():
    x = torch.randn(2,6,8)
    temb = torch.randn(2,12)
    # with gate & modulation
    blk = Block(dim=8, num_heads=2, temb_dim=12, skip=True, temb_in_mlp=True)
    out = blk(x, temb=temb, skip_connection=x)
    assert out.shape == x.shape
    # without temb
    blk2 = Block(dim=8, num_heads=2, temb_dim=None, skip=False, temb_in_mlp=False, temb_gate=False, temb_after_norm=False)
    out2 = blk2(x)
    assert out2.shape == x.shape


def test_decoder_block_variants():
    x = torch.randn(2,6,8)
    context = torch.randn(2,6,8)
    temb = torch.randn(2,10)
    dblk = DecoderBlock(dim=8, num_heads=2, temb_dim=10, dim_context=8, skip=True, temb_in_mlp=True)
    out = dblk(x, context, temb=temb, sa_mask=None, xa_mask=None, skip_connection=x)
    assert out.shape == x.shape
    dblk2 = DecoderBlock(dim=8, num_heads=2, temb_dim=None, dim_context=8, skip=False, temb_in_mlp=False, temb_gate=False, temb_after_norm=False)
    out2 = dblk2(x, context)
    assert out2.shape == x.shape


def test_transformer_concat_cond_basic():
    model = TransformerConcatCond(unet_dim=16, cond_dim=4, mid_layers=3, mid_num_heads=2, mid_dim=16, hw_posemb=4)
    x = torch.randn(2,16,4,4)
    temb = torch.randn(2,512)
    cond = torch.randn(2,4,2,2)  # will interpolate
    out = model(x, temb, cond)
    assert out.shape == x.shape


def test_transformer_concat_cond_mask_and_longskip():
    model = TransformerConcatCond(unet_dim=8, cond_dim=4, mid_layers=3, mid_num_heads=2, mid_dim=8, hw_posemb=4, use_long_skip=True)
    x = torch.randn(1,8,4,4)
    temb = torch.randn(1,512)
    cond = torch.randn(1,4,4,4)
    mask = torch.zeros(1,4,4,dtype=torch.bool)
    mask[:,2:,2:] = True
    out = model(x, temb, cond, cond_mask=mask)
    assert out.shape == x.shape


def test_transformer_xattn_cond_basic():
    model = TransformerXattnCond(unet_dim=8, cond_dim=8, mid_layers=3, mid_num_heads=2, mid_dim=8, hw_posemb=4)
    x = torch.randn(1,8,4,4)
    temb = torch.randn(1,512)
    cond = torch.randn(1,8,4,4)
    out = model(x, temb, cond)
    assert out.shape == x.shape


def test_transformer_xattn_cond_mask_longskip():
    model = TransformerXattnCond(unet_dim=8, cond_dim=8, mid_layers=3, mid_num_heads=2, mid_dim=8, hw_posemb=4, use_long_skip=True)
    x = torch.randn(1,8,4,4)
    temb = torch.randn(1,512)
    cond = torch.randn(1,8,4,4)
    mask = torch.ones(1,4,4,dtype=torch.bool)  # mask all, attention should ignore
    out = model(x, temb, cond, cond_mask=mask)
    assert out.shape == x.shape


def test_uvit_forward_concat_positional():
    model = UViT(patch_size=4, block_out_channels=(16,32), layers_per_block=1, mid_layers=4, mid_num_heads=4, mid_dim=64, cond_type='concat', cond_dim=8, sample_size=32, norm_num_groups=8)
    x = torch.randn(2,3,32,32)
    cond = torch.randn(2,8,16,16)
    out = model(x, timestep=torch.tensor([10]), condition=cond)
    assert out.shape[2:] == x.shape[2:]


def test_uvit_forward_xattn_fourier():
    model = UViT(patch_size=4, block_out_channels=(16,32), layers_per_block=1, mid_layers=4, mid_num_heads=4, mid_dim=64, cond_type='xattn', cond_dim=32, time_embedding_type='fourier', sample_size=32, norm_num_groups=8)
    x = torch.randn(1,3,32,32)
    cond = torch.randn(1,32,32,32)
    out = model(x, timestep=5, condition=cond)
    assert out.shape == x.shape


def test_uvit_forward_res_embedding():
    model = UViT(patch_size=4, block_out_channels=(16,32), layers_per_block=1, mid_layers=2, mid_num_heads=2, mid_dim=32, cond_type='concat', cond_dim=8, res_embedding=True, sample_size=32, norm_num_groups=8)
    x = torch.randn(1,3,32,32)
    cond = torch.randn(1,8,8,8)
    out = model(x, timestep=3, condition=cond, orig_res=(32,32))
    assert out.shape == x.shape


def test_uvit_factory_b_p4_f16():
    model = uvit_b_p4_f16(sample_size=64, in_channels=3, out_channels=3, cond_dim=32)
    x = torch.randn(1,3,64,64)
    cond = torch.randn(1,32,64,64)
    out = model(x, timestep=1, condition=cond)
    assert out.shape == x.shape


def test_uvit_factory_b_p4_f16_longskip():
    model = uvit_b_p4_f16_longskip(sample_size=64, in_channels=3, out_channels=3, cond_dim=32)
    x = torch.randn(1,3,64,64)
    cond = torch.randn(1,32,64,64)
    out = model(x, timestep=2, condition=cond)
    assert out.shape == x.shape


def test_uvit_factory_b_p4_f8():
    model = uvit_b_p4_f8(sample_size=32, in_channels=3, out_channels=3, cond_dim=32)
    x = torch.randn(1,3,32,32)
    cond = torch.randn(1,32,32,32)
    out = model(x, timestep=4, condition=cond)
    assert out.shape == x.shape


def test_weight_init_embedding_std():
    model = UViT(patch_size=4, block_out_channels=(16,32), layers_per_block=1, mid_layers=2, mid_num_heads=2, mid_dim=32, cond_type='concat', cond_dim=8, norm_num_groups=8)
    # Find an embedding module (time embedding has linear, but we rely on init_std property existing)
    assert hasattr(model, 'init_std') and math.isclose(model.init_std, 0.02, rel_tol=1e-6)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

import pytest
import torch
from terratorch.models.backbones.clay_v1.modules import (
    FeedForward,
    Attention,
    Transformer,
    Encoder,
    EmbeddingEncoder,
    DynamicEmbedding,
    WavesTransformer,
    Datacuber,
    WAVELENGTHS
)

# ----------- FeedForward -----------
def test_feedforward_forward():
    x = torch.randn(2, 10, 16)
    ff = FeedForward(dim=16, hidden_dim=32)
    out = ff(x)
    assert out.shape == x.shape

# ----------- Attention -----------
def test_attention_forward(monkeypatch):
    x = torch.randn(2, 5, 16)
    attn = Attention(dim=16, heads=2, dim_head=8)

    # patch use_fused_attn to False to hit else branch
    monkeypatch.setattr("timm.layers.use_fused_attn", lambda: False)
    out = attn(x)
    assert out.shape == x.shape

# ----------- Transformer -----------
def test_transformer_forward():
    x = torch.randn(2, 5, 16)
    trans = Transformer(dim=16, depth=2, heads=2, dim_head=8, mlp_dim=32)
    out = trans(x)
    # out is list of tensor per layer
    assert isinstance(out, list)
    assert out[-1].shape == x.shape

def test_transformer_vpt_forward():
    x = torch.randn(2, 5, 16)
    trans = Transformer(dim=16, depth=2, heads=2, dim_head=8, mlp_dim=32, vpt=True, vpt_n_tokens=3)
    out = trans(x)

# ----------- Encoder -----------
def get_datacube(batch=2, C=3, H=4, W=4):
    cube = torch.randn(batch, C, H, W)
    time = torch.randn(batch, 2)
    latlon = torch.randn(batch, 2)
    gsd = torch.tensor(1.0)
    waves = torch.randn(C, 128)
    return {"pixels": cube, "time": time, "latlon": latlon, "gsd": gsd, "waves": waves}

def test_encoder_forward():
    C = 16          # match model channels
    datacube = get_datacube(batch=2, C=C, H=4, W=4)

    enc = Encoder(
        mask_ratio=0.5,
        patch_size=2,
        shuffle=True,
        dim=16,
        depth=1,
        heads=2,
        dim_head=8,
        mlp_ratio=2.0,
    )

    # create a waves tensor that matches the encoder's expected wave_dim
    waves = torch.randn(datacube.shape[1], enc.dim)  # channels x dim
    out, unmasked_idx, masked_idx, mask = enc(datacube, waves)
    assert out.shape[0] == datacube.shape[0]


# ----------- EmbeddingEncoder -----------
def test_embedding_encoder_forward():
    datacube = get_datacube(batch=2, C=16, H=4, W=4)

    enc = EmbeddingEncoder(
        img_size=4,
        patch_size=2,
        dim=16,
        depth=1,
        heads=2,
        dim_head=8,
        mlp_ratio=2.0
    )

    waves = torch.randn(datacube.shape[1], enc.dim)  # channels x dim
    out = enc(datacube, waves)
    assert out.shape[0] == datacube.shape[0]


# ----------- WavesTransformer -----------
def test_waves_transformer_forward():
    x = torch.randn(10, 128)
    wt = WavesTransformer(wave_dim=128, output_dim=32, num_latent_tokens=2, embed_dim=16, is_decoder=False)
    weights, bias = wt(x)
    assert weights.shape[0] == 10
    assert bias.shape[0] == 16

# ----------- DynamicEmbedding -----------
def test_dynamic_embedding_forward():
    batch = torch.randn(2, 16, 4, 4)
    waves = torch.randn(batch.shape[1], 16)  # must match embed_dim / wave_dim inside the module

    de = DynamicEmbedding(
        wave_dim=16,
        num_latent_tokens=2,
        patch_size=2,
        embed_dim=16,
        is_decoder=False
    )

    out, waves_out = de(batch, waves)
    assert out.shape[0] == batch.shape[0]
    assert waves_out.shape[1] == waves.shape[1]
    
# ----------- Datacuber -----------
def test_datacuber_forward():
    x = torch.randn(2, 7, 4, 4)
    dc = Datacuber(bands=list(WAVELENGTHS.keys())[:7])
    out = dc(x)


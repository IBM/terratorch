import types
import io
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from terratorch.models.backbones.terramind.tokenizer import vqvae as vq


class DummyMLP(nn.Module):
    def __init__(self, dim_in=None, dim_out=None, mode="enc"):
        super().__init__()
        self.mode = mode
        # For encoder path, VQ expects encoder.dim_out
        # For decoder MLP path, VQVAE expects decoder.dim_in
        if mode == "enc":
            in_ch = dim_in if dim_in is not None else 8
            self.dim_out = dim_out if dim_out is not None else in_ch
            self.proj = nn.Conv2d(in_ch, self.dim_out, 1)
        else:
            self.dim_in = dim_in if dim_in is not None else 16
            self.proj = nn.Identity()

    def forward(self, x):
        return self.proj(x)


def dummy_build_mlp(model_id: str, dim_in=None, dim_out=None):
    # Route by model_id: anything with "enc" builds encoder-like, others decoder-like
    if "enc" in model_id:
        return DummyMLP(dim_in=dim_in, dim_out=dim_out, mode="enc")
    return DummyMLP(dim_in=dim_in, dim_out=dim_out, mode="dec")


class DummyQuantizer(nn.Module):
    def __init__(self, dim=32, codebook_size=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size

    def forward(self, x):
        # Return input as quant, a small loss, and zero tokens
        b, d, h, w = x.shape
        loss = torch.tensor(0.5)
        tokens = torch.zeros((b, h, w), dtype=torch.long)
        return x, loss, tokens

    __call__ = forward

    def indices_to_embedding(self, tokens: torch.LongTensor):
        # Map tokens back to zeros latent embedding with expected dim
        b, h, w = tokens.shape
        return torch.zeros((b, self.dim, h, w), dtype=torch.float32)


@pytest.fixture(autouse=True)
def patch_lightweight_modules(monkeypatch):
    # Patch build_mlp and quantizers to lightweight dummies
    monkeypatch.setattr(vq, "build_mlp", dummy_build_mlp)
    monkeypatch.setattr(vq, "VectorQuantizerLucid", DummyQuantizer)
    monkeypatch.setattr(vq, "Memcodes", DummyQuantizer)
    monkeypatch.setattr(vq, "FiniteScalarQuantizer", DummyQuantizer)
    yield


def test_vq_init_and_encode_tokenize_mlp_fsq():
    model = vq.VQ(
        image_size=16,
        n_channels=4,
        enc_type="MLP_tiny_enc",
        quant_type="fsq",
        latent_dim=8,
        norm_codes=False,
    )
    x = torch.randn(2, 4, 16, 16)
    quant, code_loss, tokens = model.encode(x)
    assert quant.shape[0] == 2 and quant.shape[1] == model.latent_dim
    assert code_loss.item() == pytest.approx(0.5)
    assert tokens.shape[-2:] == (quant.shape[-2], quant.shape[-1])

    out_tokens = model.tokenize(x)
    assert out_tokens.shape == tokens.shape


def test_prepare_input_undo_std(monkeypatch):
    # Patch denormalize to a known function
    monkeypatch.setattr(vq, "denormalize", lambda t: t * 0.5)
    model = vq.VQ(image_size=8, n_channels=3, enc_type="MLP_tiny_enc", quant_type="fsq", undo_std=True)
    x = torch.ones(1, 3, 8, 8)
    y = model.prepare_input(x)
    # 2*denorm(x)-1 = 2*(0.5)-1 = 0
    assert torch.allclose(y, torch.zeros_like(y))


def test_prepare_input_cls_emb():
    model = vq.VQ(image_size=8, n_channels=5, n_labels=7, enc_type="MLP_tiny_enc", quant_type="fsq")
    # Class indices within [0, n_labels)
    x = torch.randint(0, 7, (2, 8, 8))
    y = model.prepare_input(x)
    # becomes B C H W with C=n_channels
    assert y.shape == (2, 5, 8, 8)


def test_to_rgb_outputs_in_0_1_range():
    model = vq.VQ(image_size=8, n_channels=4, n_labels=4, enc_type="MLP_tiny_enc", quant_type="fsq")
    # Vary across pixels to avoid div by zero
    x = torch.rand(1, 4, 8, 8)
    rgb = model.to_rgb(x)
    assert rgb.shape == (1, 3, 8, 8)
    assert 0.0 <= rgb.min().item() <= 1.0
    assert 0.0 <= rgb.max().item() <= 1.0


def test_train_freeze_enc_behavior():
    model = vq.VQ(image_size=8, n_channels=3, enc_type="MLP_tiny_enc", quant_type="fsq", freeze_enc=True)
    # train(True) should not switch frozen modules
    model.train(True)
    # encoder is frozen (no grad) and eval mode
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert not model.encoder.training


def test_train_invalid_mode_raises():
    model = vq.VQ(image_size=8, n_channels=3, enc_type="MLP_tiny_enc", quant_type="fsq")
    with pytest.raises(ValueError, match="training mode is expected to be boolean"):
        model.train(mode="yes")


def test_init_from_ckpt_key_renaming(tmp_path, monkeypatch):
    # Build a checkpoint with various keys to trigger renaming branches
    sd = {
        "quant_conv.0.weight": torch.randn(8, 8, 1, 1),
        "quant_conv.0.bias": torch.randn(8),
        "post_quant_conv.0.weight": torch.randn(8, 8, 1, 1),
        "post_quant_conv.0.bias": torch.randn(8),
        "decoder.weight": torch.randn(1),  # in ignore list
    }
    ckpt_path = tmp_path / "ckpt_a.pth"
    torch.save({"state_dict": sd}, ckpt_path)

    model = vq.VQ(image_size=8, n_channels=3, enc_type="MLP_tiny_enc", quant_type="fsq")

    # Monkeypatch load_state_dict to avoid strict shape checks
    calls = {}

    def fake_load_state_dict(self, new_sd, strict=False):
        calls["called"] = True
        # expected renames applied
        assert "quant_proj.weight" in new_sd and "quant_proj.bias" in new_sd
        assert "post_quant_proj.weight" in new_sd and "post_quant_proj.bias" in new_sd
        # old keys removed
        assert "quant_conv.0.weight" not in new_sd and "post_quant_conv.0.weight" not in new_sd
        return types.SimpleNamespace()

    monkeypatch.setattr(vq.VQ, "load_state_dict", fake_load_state_dict, raising=True)

    model.init_from_ckpt(str(ckpt_path), ignore_keys=["decoder"])  # ignore some keys
    assert calls.get("called", False)


def test_init_from_ckpt_alt_keys(tmp_path, monkeypatch):
    # Alternate key names without .0
    sd = {
        "quant_conv.weight": torch.randn(8, 8, 1, 1),
        "quant_conv.bias": torch.randn(8),
        "post_quant_conv.weight": torch.randn(8, 8, 1, 1),
        "post_quant_conv.bias": torch.randn(8),
    }
    ckpt_path = tmp_path / "ckpt_b.pth"
    torch.save({"model": sd}, ckpt_path)

    model = vq.VQ(image_size=8, n_channels=3, enc_type="MLP_tiny_enc", quant_type="fsq")

    def fake_load_state_dict(self, new_sd, strict=False):
        assert "quant_proj.weight" in new_sd and "post_quant_proj.weight" in new_sd
        return types.SimpleNamespace()

    monkeypatch.setattr(vq.VQ, "load_state_dict", fake_load_state_dict, raising=True)

    model.init_from_ckpt(str(ckpt_path))


def test_vq_decode_tokens_calls_decode_quant(monkeypatch):
    # Use VQVAE and spy on decode_quant; ensure image_size derived from tokens*patch_size
    model = vq.VQVAE(
        image_size=32,
        n_channels=3,
        enc_type="MLP_tiny_enc",
        dec_type="MLP_tiny_dec",
        quant_type="fsq",
        latent_dim=8,
        patch_size=4,
    )

    called = {}

    def fake_decode_quant(self, quant, **kwargs):
        called["image_size"] = kwargs.get("image_size")
        # Return dummy image tensor
        b, d, h, w = quant.shape
        return torch.zeros((b, self.n_channels, h * self.patch_size, w * self.patch_size))

    monkeypatch.setattr(vq.VQVAE, "decode_quant", fake_decode_quant, raising=True)

    tokens = torch.zeros((2, 5, 7), dtype=torch.long)
    out = model.decode_tokens(tokens)
    assert called["image_size"] == (5 * 4, 7 * 4)
    assert out.shape == (2, model.n_channels, 5 * 4, 7 * 4)


def test_vqvae_forward_and_autoencode(monkeypatch):
    model = vq.VQVAE(
        image_size=16,
        n_channels=3,
        enc_type="MLP_tiny_enc",
        dec_type="MLP_tiny_dec",
        quant_type="fsq",
        latent_dim=8,
    )
    # Ensure decode_quant returns an image-shaped tensor
    def fake_decode_quant(self, quant, **kwargs):
        b, d, h, w = quant.shape
        return torch.zeros((b, self.n_channels, self.image_size, self.image_size))

    monkeypatch.setattr(vq.VQVAE, "decode_quant", fake_decode_quant, raising=True)
    x = torch.randn(2, 3, 16, 16)
    dec, code_loss = model(x)
    assert dec.shape == x.shape
    assert code_loss.shape == ()

    rec = model.autoencode(x)
    assert rec.shape == x.shape


def test_divae_importerror_when_diffusers_missing():
    # If diffusers is installed, skip this test, otherwise expect ImportError
    try:
        import diffusers  # noqa: F401
        pytest.skip("diffusers installed; skipping ImportError assertion")
    except Exception:
        with pytest.raises(ImportError):
            _ = vq.DiVAE()


def test_divae_sample_mask_functionality():
    # Bypass __init__ to avoid heavy deps
    d = object.__new__(vq.DiVAE)
    quant = torch.randn(3, 8, 4, 5)
    mask = vq.DiVAE.sample_mask(d, quant, low=0, high=10)
    assert mask.shape == (3, 4, 5)
    # Check boolean
    assert mask.dtype == torch.bool


def test_vq_init_lucid_memcodes_and_invalid_quant():
    # lucid path
    m_lucid = vq.VQ(image_size=8, n_channels=3, enc_type="MLP_tiny_enc", quant_type="lucid", latent_dim=6)
    x = torch.randn(1, 3, 8, 8)
    q_l, loss_l, tok_l = m_lucid.encode(x)
    assert q_l.shape[1] == 6 and tok_l.shape[-2:] == q_l.shape[-2:]
    # memcodes path
    m_mem = vq.VQ(image_size=8, n_channels=3, enc_type="MLP_tiny_enc", quant_type="memcodes", latent_dim=5)
    q_m, loss_m, tok_m = m_mem.encode(x)
    assert q_m.shape[1] == 5 and tok_m.shape[-2:] == q_m.shape[-2:]
    # invalid quant type
    with pytest.raises(ValueError):
        _ = vq.VQ(image_size=8, n_channels=3, enc_type="MLP_tiny_enc", quant_type="badtype")


def test_vq_invalid_enc_type():
    with pytest.raises(NotImplementedError):
        _ = vq.VQ(image_size=8, n_channels=3, enc_type="unknown_enc", quant_type="fsq")


def test_vqvae_invalid_dec_type():
    with pytest.raises(NotImplementedError):
        _ = vq.VQVAE(image_size=8, n_channels=3, enc_type="MLP_tiny_enc", dec_type="unknown_dec", quant_type="fsq")


def test_vqvae_decode_quant_original_implementation():
    # Exercise real decode_quant path (post_quant_proj + decoder) using MLP decoder
    model = vq.VQVAE(image_size=8, n_channels=3, enc_type="MLP_tiny_enc", dec_type="MLP_tiny_dec", quant_type="fsq", latent_dim=4)
    # quant embedding (latent_dim) -> post_quant_proj -> decoder
    quant = torch.randn(2, model.latent_dim, 2, 2)
    out = model.decode_quant(quant)
    # Decoder here is a dummy MLP returning identity with dec_dim channels
    assert out.shape == (2, model.dec_dim, 2, 2)


def test_vq_decode_tokens_explicit_image_size(monkeypatch):
    # Provide explicit image_size override differing from token-derived size
    model = vq.VQVAE(image_size=32, n_channels=3, enc_type="MLP_tiny_enc", dec_type="MLP_tiny_dec", quant_type="fsq", patch_size=4, latent_dim=8)
    # Spy on decode_quant to capture image_size passed
    called = {}
    def fake_decode_quant(self, quant, **kwargs):
        called['image_size'] = kwargs.get('image_size')
        b, d, h, w = quant.shape
        return torch.zeros((b, self.n_channels, called['image_size'][0], called['image_size'][1]))
    monkeypatch.setattr(vq.VQVAE, 'decode_quant', fake_decode_quant, raising=True)
    tokens = torch.zeros((1, 3, 5), dtype=torch.long)
    out = model.decode_tokens(tokens, image_size=(20, 22))
    assert called['image_size'] == (20, 22)
    assert out.shape == (1, 3, 20, 22)


def test_divae_forward_and_train_forward_with_masking(monkeypatch):
    # Stub diffusers presence
    sys.modules.setdefault('diffusers', types.ModuleType('diffusers'))
    # Patch modules used inside DiVAE to lightweight dummies
    class DummyDecoder(nn.Module):
        def __init__(self, in_channels=None, out_channels=None, cond_channels=None, image_size=None, **kwargs):
            super().__init__()
            self.out_channels = out_channels
            self.image_size = image_size
        def forward(self, input_noised, timesteps, quant, cond_mask=None, orig_res=None):
            b = input_noised.shape[0]
            return torch.zeros(b, self.out_channels, self.image_size, self.image_size)
        def __call__(self, *a, **k):
            return self.forward(*a, **k)
    class DummyPipeline:
        def __init__(self, model=None, scheduler=None, n_channels=None):
            self.model = model
            self.n_channels = n_channels
        def __call__(self, quant, timesteps=10, generator=None, image_size=None, verbose=False, scheduler_timesteps_mode=None, orig_res=None):
            b = quant.shape[0]
            sz = image_size if isinstance(image_size, int) else (image_size if image_size is not None else self.model.image_size)
            if isinstance(sz, tuple):
                h, w = sz
            else:
                h = w = sz
            return torch.zeros(b, self.n_channels, h, w)
    class DummyScheduler:
        def __init__(self, **kwargs):
            pass
    monkeypatch.setattr(vq, 'unet', types.SimpleNamespace(unet_patched=DummyDecoder))
    monkeypatch.setattr(vq, 'DDPMScheduler', DummyScheduler)
    monkeypatch.setattr(vq, 'DDIMScheduler', DummyScheduler)
    monkeypatch.setattr(vq, 'PipelineCond', DummyPipeline)
    model = vq.DiVAE(
        image_size=16,
        n_channels=3,
        enc_type="MLP_tiny_enc",
        quant_type="fsq",
        dec_type="unet_patched",
        latent_dim=8,
        cls_free_guidance_dropout=1.0,
        masked_cfg=True,
        masked_cfg_low=0,
        masked_cfg_high=4,
    )
    model.train(True)
    x_clean = torch.randn(2, 3, 16, 16)
    x_noised = torch.randn(2, 3, 16, 16)
    # forward (autoencode style)
    out = model.forward(x_clean, timesteps=5, verbose=False)
    assert out.shape == (2, 3, 16, 16)
    # train_forward with masking path
    dec, code_loss = model.train_forward(x_clean, x_noised, timesteps=torch.tensor([5,5]))
    assert dec.shape == (2, 3, 16, 16)
    assert code_loss.shape == ()


def test_vq_config_delegation_init():
    # Ensure config-based init path executes recursion correctly
    config = {"image_size": 8, "n_channels": 3, "enc_type": "MLP_tiny_enc", "quant_type": "fsq"}
    model = vq.VQ(config=config)
    assert model.image_size == 8 and model.n_channels == 3


def test_vqvae_config_delegation_init():
    config = {"image_size": 8, "n_channels": 3, "enc_type": "MLP_tiny_enc", "dec_type": "MLP_tiny_dec", "quant_type": "fsq"}
    model = vq.VQVAE(config=config)
    assert model.image_size == 8 and model.n_channels == 3


def test_vq_vit_enc_type_and_dec_type_paths(monkeypatch):
    # Patch vit_models to return a dummy ViT-like encoder/decoder
    class DummyViTEnc(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.dim_tokens = 16
        def forward(self, x):
            b, c, h, w = x.shape
            return torch.zeros(b, self.dim_tokens, h // 2, w // 2)
    class DummyViTDec(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.dim_tokens = 16
        def forward(self, x):
            b, d, h, w = x.shape
            return torch.zeros(b, 3, h * 2, w * 2)
    monkeypatch.setattr(vq, 'vit_models', types.SimpleNamespace(vit_tiny_enc=DummyViTEnc, vit_tiny_dec=DummyViTDec))
    m = vq.VQ(image_size=8, n_channels=3, enc_type="vit_tiny_enc", quant_type="fsq", latent_dim=8)
    assert m.enc_dim == 16
    m_vae = vq.VQVAE(image_size=8, n_channels=3, enc_type="MLP_tiny_enc", dec_type="vit_tiny_dec", quant_type="fsq")
    assert m_vae.dec_dim == 16


def test_divae_dec_type_branches_uvit_and_invalid(monkeypatch):
    sys.modules.setdefault('diffusers', types.ModuleType('diffusers'))
    class DummyUViTDec(nn.Module):
        def __init__(self, sample_size=None, in_channels=None, out_channels=None, cond_dim=None, cond_type=None, mid_drop_rate=None):
            super().__init__()
            self.out_channels = out_channels
            self.image_size = sample_size
        def forward(self, input_noised, timesteps, quant, cond_mask=None, orig_res=None):
            b = input_noised.shape[0]
            return torch.zeros(b, self.out_channels, self.image_size, self.image_size)
        def __call__(self, *a, **k):
            return self.forward(*a, **k)
    class DummySched:
        def __init__(self, **kwargs):
            pass
    class DummyPipe:
        def __init__(self, **kwargs):
            pass
        def __call__(self, *a, **k):
            return torch.zeros(1, 3, 8, 8)
    monkeypatch.setattr(vq, 'uvit', types.SimpleNamespace(uvit_small=DummyUViTDec))
    monkeypatch.setattr(vq, 'DDPMScheduler', DummySched)
    monkeypatch.setattr(vq, 'DDIMScheduler', DummySched)
    monkeypatch.setattr(vq, 'PipelineCond', DummyPipe)
    m = vq.DiVAE(image_size=8, n_channels=3, enc_type="MLP_tiny_enc", dec_type="uvit_small", quant_type="fsq", latent_dim=4)
    assert m.decoder.out_channels == 3
    # invalid dec_type
    with pytest.raises(NotImplementedError):
        _ = vq.DiVAE(image_size=8, n_channels=3, enc_type="MLP_tiny_enc", dec_type="invalid_dec", quant_type="fsq")


def test_divae_ddpm_scheduler_selection(monkeypatch):
    sys.modules.setdefault('diffusers', types.ModuleType('diffusers'))
    class DummyDec(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.out_channels = 3
            self.image_size = 8
    ddpm_called = []
    ddim_called = []
    class DummyDDPM:
        def __init__(self, **kwargs):
            ddpm_called.append(True)
    class DummyDDIM:
        def __init__(self, **kwargs):
            ddim_called.append(True)
    class DummyPipe:
        def __init__(self, **kwargs):
            pass
    monkeypatch.setattr(vq, 'unet', types.SimpleNamespace(unet_patched=DummyDec))
    monkeypatch.setattr(vq, 'DDPMScheduler', DummyDDPM)
    monkeypatch.setattr(vq, 'DDIMScheduler', DummyDDIM)
    monkeypatch.setattr(vq, 'PipelineCond', DummyPipe)
    _ = vq.DiVAE(image_size=8, n_channels=3, enc_type="MLP_tiny_enc", dec_type="unet_patched", quant_type="fsq", scheduler="ddpm")
    assert len(ddpm_called) == 1 and len(ddim_called) == 0


def test_divae_decode_quant_and_get_pipeline(monkeypatch):
    sys.modules.setdefault('diffusers', types.ModuleType('diffusers'))
    class DummyDec(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.out_channels = 3
            self.image_size = 8
    class DummySched:
        def __init__(self, **kwargs):
            pass
    pipe_calls = []
    class DummyPipe:
        def __init__(self, model=None, scheduler=None, n_channels=None):
            pipe_calls.append({'model': model, 'scheduler': scheduler})
        def __call__(self, quant, **kwargs):
            return torch.zeros(1, 3, 8, 8)
    monkeypatch.setattr(vq, 'unet', types.SimpleNamespace(unet_patched=DummyDec))
    monkeypatch.setattr(vq, 'DDPMScheduler', DummySched)
    monkeypatch.setattr(vq, 'DDIMScheduler', DummySched)
    monkeypatch.setattr(vq, 'PipelineCond', DummyPipe)
    model = vq.DiVAE(image_size=8, n_channels=3, enc_type="MLP_tiny_enc", dec_type="unet_patched", quant_type="fsq", latent_dim=4)
    # decode_quant with custom scheduler
    custom_sched = DummySched()
    quant = torch.zeros(1, 4, 2, 2)
    out = model.decode_quant(quant, scheduler=custom_sched, timesteps=10, image_size=(8,8), verbose=False)
    assert out.shape == (1, 3, 8, 8)
    # _get_pipeline internally called
    assert len(pipe_calls) >= 2  # init + decode_quant call


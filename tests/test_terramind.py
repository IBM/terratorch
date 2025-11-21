# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for TerraMind model"""

import pytest
import torch
import torch.nn as nn
from functools import partial

from terratorch.models.backbones.terramind.model.terramind import (
    TerraMind,
    build_tokenizer,
    build_modality_embeddings,
    build_output_modality_embeddings,
)
from terratorch.models.backbones.terramind.model.tm_utils import LayerNorm


# Mock embedding modules for testing
class MockEncoderEmbedding(nn.Module):
    def __init__(self, num_tokens=16, dim=64):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.mod_emb = nn.Parameter(torch.randn(1, 1, dim))
        
    def init(self, dim_tokens, init_std):
        self.dim = dim_tokens
        
    def forward(self, input_dict):
        B = input_dict['tensor'].shape[0]
        return {
            'x': torch.randn(B, self.num_tokens, self.dim),
            'emb': torch.randn(B, self.num_tokens, self.dim),
            'input_mask': torch.zeros(B, self.num_tokens, dtype=torch.bool),
            'tensor': input_dict['tensor'],  # Pass through tensor for B extraction
        }
    
    def no_weight_decay(self):
        return {'mod_emb'}


class MockDecoderEmbedding(nn.Module):
    def __init__(self, num_tokens=16, dim=64, vocab_size=100):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.vocab_size = vocab_size
        self.mod_emb = nn.Parameter(torch.randn(1, 1, dim))
        self.proj = nn.Linear(dim, vocab_size)
        
    def init(self, dim_tokens, init_std):
        self.dim = dim_tokens
        
    def forward_embed(self, input_dict):
        B = input_dict['tensor'].shape[0]
        return {
            'x': torch.randn(B, self.num_tokens, self.dim),
            'ids': torch.randint(0, self.vocab_size, (B, self.num_tokens)),
            'emb': torch.randn(B, self.num_tokens, self.dim),
            'target_mask': torch.zeros(B, self.num_tokens, dtype=torch.bool),
            'decoder_attention_mask': torch.ones(B, self.num_tokens, dtype=torch.long),
        }
    
    def forward_logits(self, y):
        return self.proj(y)
    
    def no_weight_decay(self):
        return {'mod_emb'}


@pytest.fixture
def modality_info():
    """Create mock modality info"""
    return {
        'mod1': {'id': 0, 'type': 'img'},
        'mod2': {'id': 1, 'type': 'img'},
        'mod3': {'id': 2, 'type': 'seq'},
    }


@pytest.fixture
def encoder_embeddings():
    """Create mock encoder embeddings"""
    return {
        'mod1': MockEncoderEmbedding(num_tokens=16, dim=64),
        'mod2': MockEncoderEmbedding(num_tokens=12, dim=64),
    }


@pytest.fixture
def decoder_embeddings():
    """Create mock decoder embeddings"""
    return {
        'mod1': MockDecoderEmbedding(num_tokens=16, dim=64, vocab_size=100),
        'mod3': MockDecoderEmbedding(num_tokens=8, dim=64, vocab_size=50),
    }


@pytest.fixture
def small_terramind(modality_info, encoder_embeddings, decoder_embeddings):
    """Create a small TerraMind model for testing"""
    model = TerraMind(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        modality_info=modality_info,
        dim=64,
        encoder_depth=2,
        decoder_depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        proj_bias=True,
        mlp_bias=True,
        drop_path_rate_encoder=0.0,
        drop_path_rate_decoder=0.0,
        num_register_tokens=0,
    )
    return model


def test_terramind_initialization(small_terramind, modality_info):
    """Test TerraMind initialization"""
    model = small_terramind
    
    assert model.dim == 64
    assert len(model.encoder) == 2
    assert len(model.decoder) == 2
    assert model.modality_info == modality_info
    assert model.encoder_modalities == {'mod1', 'mod2'}
    assert model.decoder_modalities == {'mod1', 'mod3'}
    assert model.init_std == 0.02
    assert model.num_register_tokens == 0


def test_terramind_with_register_tokens(modality_info, encoder_embeddings, decoder_embeddings):
    """Test TerraMind with register tokens"""
    model = TerraMind(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        modality_info=modality_info,
        dim=64,
        encoder_depth=2,
        decoder_depth=2,
        num_heads=4,
        num_register_tokens=4,
    )
    
    assert model.num_register_tokens == 4
    assert model.register_tokens is not None
    assert model.register_tokens.shape == (1, 4, 64)


def test_terramind_causal_mask(modality_info, encoder_embeddings, decoder_embeddings):
    """Test TerraMind with causal masking"""
    model = TerraMind(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        modality_info=modality_info,
        dim=64,
        encoder_depth=2,
        decoder_depth=2,
        num_heads=4,
        decoder_causal_mask=True,
        decoder_sep_mask=False,
    )
    
    assert model.decoder_causal_mask is True
    assert model.decoder_sep_mask is False


def test_terramind_gated_mlp(modality_info, encoder_embeddings, decoder_embeddings):
    """Test TerraMind with gated MLP"""
    model = TerraMind(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        modality_info=modality_info,
        dim=64,
        encoder_depth=2,
        decoder_depth=2,
        num_heads=4,
        gated_mlp=True,
    )
    
    # Check that encoder blocks use gated MLP
    assert hasattr(model.encoder[0], 'mlp')


def test_terramind_qk_norm(modality_info, encoder_embeddings, decoder_embeddings):
    """Test TerraMind with QK normalization"""
    model = TerraMind(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        modality_info=modality_info,
        dim=64,
        encoder_depth=2,
        decoder_depth=2,
        num_heads=4,
        qk_norm=True,
    )
    
    # Model should initialize without errors
    assert len(model.encoder) == 2


def test_share_modality_embeddings(modality_info, encoder_embeddings, decoder_embeddings):
    """Test sharing of modality embeddings"""
    model = TerraMind(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        modality_info=modality_info,
        dim=64,
        encoder_depth=2,
        decoder_depth=2,
        num_heads=4,
        share_modality_embeddings=True,
    )
    
    # mod1 should be shared
    assert model.encoder_embeddings['mod1'].mod_emb is model.decoder_embeddings['mod1'].mod_emb


def test_no_share_modality_embeddings(modality_info, encoder_embeddings, decoder_embeddings):
    """Test not sharing modality embeddings"""
    model = TerraMind(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        modality_info=modality_info,
        dim=64,
        encoder_depth=2,
        decoder_depth=2,
        num_heads=4,
        share_modality_embeddings=False,
    )
    
    # mod1 should NOT be shared (different instances)
    assert model.encoder_embeddings['mod1'].mod_emb is not model.decoder_embeddings['mod1'].mod_emb


def test_get_num_layers(small_terramind):
    """Test layer count methods"""
    model = small_terramind
    
    assert model.get_num_layers_encoder() == 2
    assert model.get_num_layers_decoder() == 2
    assert model.get_num_layers() == 4


def test_no_weight_decay(small_terramind):
    """Test no_weight_decay method"""
    model = small_terramind
    
    no_wd_set = model.no_weight_decay()
    
    assert isinstance(no_wd_set, set)
    assert 'encoder_embeddings.mod1.mod_emb' in no_wd_set
    assert 'decoder_embeddings.mod1.mod_emb' in no_wd_set


def test_cat_encoder_tensors(small_terramind):
    """Test concatenation of encoder tensors"""
    model = small_terramind
    B = 2
    
    mod_dict = {
        'mod1': {
            'x': torch.randn(B, 16, 64),
            'emb': torch.randn(B, 16, 64),
            'input_mask': torch.zeros(B, 16, dtype=torch.bool),
        },
        'mod2': {
            'x': torch.randn(B, 12, 64),
            'emb': torch.randn(B, 12, 64),
            'input_mask': torch.zeros(B, 12, dtype=torch.bool),
        },
    }
    
    encoder_tokens, emb, encoder_mask, mod_mask = model.cat_encoder_tensors(mod_dict)
    
    assert encoder_tokens.shape == (B, 28, 64)  # 16 + 12
    assert emb.shape == (B, 28, 64)
    assert encoder_mask.shape == (B, 28)
    assert mod_mask.shape == (B, 28)
    assert mod_mask[0, 0] == 0  # mod1 id
    assert mod_mask[0, 16] == 1  # mod2 id


def test_cat_decoder_tensors_img_modality(small_terramind):
    """Test concatenation of decoder tensors for image modality"""
    model = small_terramind
    B = 2
    
    mod_dict = {
        'mod1': {
            'x': torch.randn(B, 16, 64),
            'ids': torch.randint(0, 100, (B, 16)),
            'emb': torch.randn(B, 16, 64),
            'target_mask': torch.zeros(B, 16, dtype=torch.bool),
            'decoder_attention_mask': torch.ones(B, 16, dtype=torch.long),
        },
    }
    
    decoder_tokens, emb, decoder_mask, target_ids, attn_mask, mod_mask = model.cat_decoder_tensors(mod_dict)
    
    assert decoder_tokens.shape == (B, 16, 64)
    assert emb.shape == (B, 16, 64)
    assert decoder_mask.shape == (B, 16)
    assert target_ids.shape == (B, 16)
    assert attn_mask.shape == (B, 16)
    assert mod_mask.shape == (B, 16)


def test_cat_decoder_tensors_seq_modality(small_terramind):
    """Test concatenation of decoder tensors for sequence modality"""
    model = small_terramind
    B = 2
    
    mod_dict = {
        'mod3': {
            'x': torch.randn(B, 8, 64),
            'ids': torch.randint(0, 50, (B, 8)),
            'emb': torch.randn(B, 8, 64),
            'target_mask': torch.zeros(B, 8, dtype=torch.bool),
            'decoder_attention_mask': torch.ones(B, 8, dtype=torch.long),
        },
    }
    
    decoder_tokens, emb, decoder_mask, target_ids, attn_mask, mod_mask = model.cat_decoder_tensors(mod_dict)
    
    # For seq modality, tokens are shifted left (8 -> 7)
    assert decoder_tokens.shape == (B, 7, 64)
    assert target_ids.shape == (B, 7)


def test_forward_mask_encoder(small_terramind):
    """Test forward_mask_encoder"""
    model = small_terramind
    B = 2
    
    # Create encoder_mod_dict directly (output of encoder embeddings)
    encoder_mod_dict = {
        'mod1': {
            'x': torch.randn(B, 16, 64),
            'emb': torch.randn(B, 16, 64),
            'input_mask': torch.zeros(B, 16, dtype=torch.bool),
            'tensor': torch.randn(B, 3, 224, 224),  # Keep tensor for B extraction
        },
        'mod2': {
            'x': torch.randn(B, 12, 64),
            'emb': torch.randn(B, 12, 64),
            'input_mask': torch.zeros(B, 12, dtype=torch.bool),
            'tensor': torch.randn(B, 3, 224, 224),
        },
    }
    
    num_encoder_tokens = 20
    encoder_tokens, encoder_emb, encoder_mask, mod_mask = model.forward_mask_encoder(
        encoder_mod_dict, num_encoder_tokens
    )
    
    assert encoder_tokens.shape == (B, num_encoder_tokens, 64)
    assert encoder_emb.shape == (B, num_encoder_tokens, 64)
    assert encoder_mask.shape == (B, 1, num_encoder_tokens)
    assert mod_mask.shape == (B, num_encoder_tokens)


def test_forward_mask_encoder_with_register_tokens(modality_info, encoder_embeddings, decoder_embeddings):
    """Test forward_mask_encoder with register tokens"""
    model = TerraMind(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        modality_info=modality_info,
        dim=64,
        encoder_depth=2,
        decoder_depth=2,
        num_heads=4,
        num_register_tokens=4,
    )
    
    B = 2
    encoder_mod_dict = {
        'mod1': {
            'x': torch.randn(B, 16, 64),
            'emb': torch.randn(B, 16, 64),
            'input_mask': torch.zeros(B, 16, dtype=torch.bool),
            'tensor': torch.randn(B, 3, 224, 224),
        },
    }
    
    num_encoder_tokens = 10
    encoder_tokens, encoder_emb, encoder_mask, mod_mask = model.forward_mask_encoder(
        encoder_mod_dict, num_encoder_tokens
    )
    
    # Should include register tokens
    assert encoder_tokens.shape == (B, num_encoder_tokens + 4, 64)
    assert mod_mask[0, 0] == -1  # Register tokens marked as -1


def test_forward_mask_decoder(small_terramind):
    """Test forward_mask_decoder"""
    model = small_terramind
    B = 2
    
    decoder_mod_dict = {
        'mod1': {
            'x': torch.randn(B, 16, 64),
            'ids': torch.randint(0, 100, (B, 16)),
            'emb': torch.randn(B, 16, 64),
            'target_mask': torch.zeros(B, 16, dtype=torch.bool),
            'decoder_attention_mask': torch.ones(B, 16, dtype=torch.long),
        },
    }
    
    num_decoder_tokens = 12
    decoder_tokens, decoder_emb, decoder_mask, target_ids, attn_mask, mod_mask = model.forward_mask_decoder(
        decoder_mod_dict, num_decoder_tokens
    )
    
    assert decoder_tokens.shape == (B, num_decoder_tokens, 64)
    assert decoder_emb.shape == (B, num_decoder_tokens, 64)
    assert decoder_mask.shape == (B, 1, num_decoder_tokens)
    assert target_ids.shape == (B, num_decoder_tokens)
    assert attn_mask.shape == (B, num_decoder_tokens, num_decoder_tokens)
    assert mod_mask.shape == (B, num_decoder_tokens)


def test_adapt_decoder_attention_mask_causal(modality_info, encoder_embeddings, decoder_embeddings):
    """Test adapt_decoder_attention_mask with causal mask"""
    model = TerraMind(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        modality_info=modality_info,
        dim=64,
        encoder_depth=2,
        decoder_depth=2,
        num_heads=4,
        decoder_causal_mask=True,
        decoder_sep_mask=False,
    )
    
    B = 2
    N = 8
    decoder_attention_mask = torch.ones(B, N, dtype=torch.long)
    mod_mask = torch.zeros(B, N, dtype=torch.int)
    
    adapted_mask = model.adapt_decoder_attention_mask(decoder_attention_mask, mod_mask)
    
    assert adapted_mask.shape == (B, N, N)
    # Causal mask: upper triangular should be True (masked)
    assert adapted_mask[0, 0, 1].item() is True
    assert adapted_mask[0, 1, 0].item() is False


def test_adapt_decoder_attention_mask_sep(modality_info, encoder_embeddings, decoder_embeddings):
    """Test adapt_decoder_attention_mask with separate modality mask"""
    model = TerraMind(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        modality_info=modality_info,
        dim=64,
        encoder_depth=2,
        decoder_depth=2,
        num_heads=4,
        decoder_causal_mask=False,
        decoder_sep_mask=True,
    )
    
    B = 2
    N = 8
    decoder_attention_mask = torch.ones(B, N, dtype=torch.long) * N  # All attend to all
    mod_mask = torch.cat([torch.zeros(B, 4, dtype=torch.int), torch.ones(B, 4, dtype=torch.int)], dim=1)
    
    adapted_mask = model.adapt_decoder_attention_mask(decoder_attention_mask, mod_mask)
    
    assert adapted_mask.shape == (B, N, N)
    # Different modalities should not attend to each other
    assert adapted_mask[0, 0, 4].item() is True  # mod1 token can't attend to mod2 token
    assert adapted_mask[0, 0, 1].item() is False  # mod1 token can attend to another mod1 token


def test_forward_encoder(small_terramind):
    """Test forward_encoder"""
    model = small_terramind
    B = 2
    N = 16
    
    x = torch.randn(B, N, 64)
    encoder_mask = torch.zeros(B, 1, N, dtype=torch.bool)
    
    output = model.forward_encoder(x, encoder_mask)
    
    assert output.shape == (B, N, 64)


def test_forward_decoder(small_terramind):
    """Test forward_decoder"""
    model = small_terramind
    B = 2
    M = 12
    N = 16
    
    y = torch.randn(B, M, 64)
    context = torch.randn(B, N, 64)
    encoder_mask = torch.zeros(B, 1, N, dtype=torch.bool)
    decoder_attention_mask = torch.zeros(B, M, M, dtype=torch.bool)
    
    output = model.forward_decoder(y, context, encoder_mask, decoder_attention_mask)
    
    assert output.shape == (B, M, 64)


def test_forward_logits(small_terramind):
    """Test forward_logits"""
    model = small_terramind
    B = 2
    M = 16
    
    y = torch.randn(B, M, 64)
    decoder_mod_dict = {
        'mod1': {},
    }
    decoder_mod_mask = torch.zeros(B, M, dtype=torch.int)
    
    logits_dict = model.forward_logits(y, decoder_mod_dict, decoder_mod_mask, return_all_logits=True)
    
    assert 'mod1' in logits_dict
    assert logits_dict['mod1'].shape == (B, M, 100)  # vocab_size=100


def test_forward_logits_partial(small_terramind):
    """Test forward_logits with partial tokens"""
    model = small_terramind
    B = 2
    M = 16
    
    y = torch.randn(B, M, 64)
    decoder_mod_dict = {
        'mod1': {},
    }
    # Only first 8 tokens belong to mod1
    decoder_mod_mask = torch.cat([torch.zeros(B, 8, dtype=torch.int), torch.ones(B, 8, dtype=torch.int)], dim=1)
    
    logits_dict = model.forward_logits(y, decoder_mod_dict, decoder_mod_mask, return_all_logits=False)
    
    assert 'mod1' in logits_dict
    assert logits_dict['mod1'].shape == (B * 8, 100)  # Only 8 tokens per batch


def test_forward_mod_loss(small_terramind):
    """Test forward_mod_loss"""
    model = small_terramind
    B = 2
    M = 16
    
    y = torch.randn(B, M, 64)
    target_ids = torch.randint(0, 100, (B, M))
    decoder_mod_dict = {
        'mod1': {},
    }
    decoder_mod_mask = torch.zeros(B, M, dtype=torch.int)
    
    loss, mod_loss = model.forward_mod_loss(y, target_ids, decoder_mod_dict, decoder_mod_mask)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1
    assert 'mod1' in mod_loss
    assert mod_loss['mod1'].numel() == 1


def test_forward_mod_loss_empty_modality(small_terramind):
    """Test forward_mod_loss with empty modality (no tokens)"""
    model = small_terramind
    B = 2
    M = 16
    
    y = torch.randn(B, M, 64)
    target_ids = torch.randint(0, 100, (B, M))
    decoder_mod_dict = {
        'mod1': {},
    }
    # All tokens belong to a different modality (id=1)
    decoder_mod_mask = torch.ones(B, M, dtype=torch.int)
    
    loss, mod_loss = model.forward_mod_loss(y, target_ids, decoder_mod_dict, decoder_mod_mask)
    
    assert 'mod1' in mod_loss
    assert mod_loss['mod1'] == 0.0  # Should be 0 for empty modality


def test_forward_token_loss(small_terramind):
    """Test forward_token_loss"""
    model = small_terramind
    B = 2
    M = 16
    
    y = torch.randn(B, M, 64)
    target_ids = torch.randint(0, 100, (B, M))
    decoder_mod_dict = {
        'mod1': {},
    }
    decoder_mod_mask = torch.zeros(B, M, dtype=torch.int)
    
    loss, mod_loss = model.forward_token_loss(y, target_ids, decoder_mod_dict, decoder_mod_mask)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1
    assert 'mod1' in mod_loss


def test_forward_token_loss_empty_modality(small_terramind):
    """Test forward_token_loss with empty modality"""
    model = small_terramind
    B = 2
    M = 16
    
    y = torch.randn(B, M, 64)
    target_ids = torch.randint(0, 100, (B, M))
    decoder_mod_dict = {
        'mod1': {},
    }
    # All tokens belong to a different modality
    decoder_mod_mask = torch.ones(B, M, dtype=torch.int)
    
    loss, mod_loss = model.forward_token_loss(y, target_ids, decoder_mod_dict, decoder_mod_mask)
    
    assert 'mod1' in mod_loss
    assert mod_loss['mod1'] == 0.0


def test_forward_loss_mod_type(small_terramind):
    """Test forward_loss with mod loss type"""
    model = small_terramind
    B = 2
    M = 16
    
    y = torch.randn(B, M, 64)
    target_ids = torch.randint(0, 100, (B, M))
    decoder_mod_dict = {'mod1': {}}
    decoder_mod_mask = torch.zeros(B, M, dtype=torch.int)
    
    loss, mod_loss = model.forward_loss(y, target_ids, decoder_mod_dict, decoder_mod_mask, loss_type='mod')
    
    assert isinstance(loss, torch.Tensor)
    assert 'mod1' in mod_loss


def test_forward_loss_token_type(small_terramind):
    """Test forward_loss with token loss type"""
    model = small_terramind
    B = 2
    M = 16
    
    y = torch.randn(B, M, 64)
    target_ids = torch.randint(0, 100, (B, M))
    decoder_mod_dict = {'mod1': {}}
    decoder_mod_mask = torch.zeros(B, M, dtype=torch.int)
    
    loss, mod_loss = model.forward_loss(y, target_ids, decoder_mod_dict, decoder_mod_mask, loss_type='token')
    
    assert isinstance(loss, torch.Tensor)
    assert 'mod1' in mod_loss


def test_forward_loss_invalid_type(small_terramind):
    """Test forward_loss with invalid loss type"""
    model = small_terramind
    B = 2
    M = 16
    
    y = torch.randn(B, M, 64)
    target_ids = torch.randint(0, 100, (B, M))
    decoder_mod_dict = {'mod1': {}}
    decoder_mod_mask = torch.zeros(B, M, dtype=torch.int)
    
    with pytest.raises(ValueError, match="Invalid loss type"):
        model.forward_loss(y, target_ids, decoder_mod_dict, decoder_mod_mask, loss_type='invalid')


def test_full_forward_return_loss(small_terramind):
    """Test full forward pass returning loss"""
    model = small_terramind
    B = 2
    
    mod_dict = {
        'mod1': {
            'tensor': torch.randn(B, 3, 224, 224),
        },
    }
    
    loss, mod_loss = model.forward(
        mod_dict=mod_dict,
        num_encoder_tokens=10,
        num_decoder_tokens=12,
        loss_type='mod',
        return_logits=False,
    )
    
    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1
    assert isinstance(mod_loss, dict)
    assert 'mod1' in mod_loss


def test_full_forward_return_logits(small_terramind):
    """Test full forward pass returning logits"""
    model = small_terramind
    B = 2
    
    mod_dict = {
        'mod1': {
            'tensor': torch.randn(B, 3, 224, 224),
        },
    }
    
    logits_dict = model.forward(
        mod_dict=mod_dict,
        num_encoder_tokens=10,
        num_decoder_tokens=12,
        return_logits=True,
    )
    
    assert isinstance(logits_dict, dict)
    assert 'mod1' in logits_dict


def test_freeze_encoder(small_terramind):
    """Test freezing encoder"""
    model = small_terramind
    
    model.freeze_encoder(freeze_embeddings=True)
    
    # Check that encoder parameters are frozen
    for param in model.encoder.parameters():
        assert param.requires_grad is False
    
    for param in model.encoder_norm.parameters():
        assert param.requires_grad is False
    
    for param in model.encoder_embeddings.parameters():
        assert param.requires_grad is False
    
    # Decoder should still be trainable
    for param in model.decoder.parameters():
        assert param.requires_grad is True


def test_freeze_encoder_keep_embeddings(small_terramind):
    """Test freezing encoder but keeping embeddings trainable"""
    model = small_terramind
    
    model.freeze_encoder(freeze_embeddings=False)
    
    # Encoder should be frozen
    for param in model.encoder.parameters():
        assert param.requires_grad is False
    
    # Embeddings should still be trainable
    for param in model.encoder_embeddings.parameters():
        assert param.requires_grad is True


def test_unfreeze_encoder(small_terramind):
    """Test unfreezing encoder"""
    model = small_terramind
    
    model.freeze_encoder(freeze_embeddings=True)
    model.unfreeze_encoder(unfreeze_embeddings=True)
    
    # All should be trainable again
    for param in model.encoder.parameters():
        assert param.requires_grad is True
    
    for param in model.encoder_embeddings.parameters():
        assert param.requires_grad is True


def test_freeze_decoder(small_terramind):
    """Test freezing decoder"""
    model = small_terramind
    
    model.freeze_decoder(freeze_embeddings=True)
    
    # Check that decoder parameters are frozen
    for param in model.decoder.parameters():
        assert param.requires_grad is False
    
    for param in model.decoder_norm.parameters():
        assert param.requires_grad is False
    
    for param in model.decoder_embeddings.parameters():
        assert param.requires_grad is False
    
    # Encoder should still be trainable
    for param in model.encoder.parameters():
        assert param.requires_grad is True


def test_freeze_decoder_keep_embeddings(small_terramind):
    """Test freezing decoder but keeping embeddings trainable"""
    model = small_terramind
    
    model.freeze_decoder(freeze_embeddings=False)
    
    # Decoder should be frozen
    for param in model.decoder.parameters():
        assert param.requires_grad is False
    
    # Embeddings should still be trainable
    for param in model.decoder_embeddings.parameters():
        assert param.requires_grad is True


def test_unfreeze_decoder(small_terramind):
    """Test unfreezing decoder"""
    model = small_terramind
    
    model.freeze_decoder(freeze_embeddings=True)
    model.unfreeze_decoder(unfreeze_embeddings=True)
    
    # All should be trainable again
    for param in model.decoder.parameters():
        assert param.requires_grad is True
    
    for param in model.decoder_embeddings.parameters():
        assert param.requires_grad is True


def test_freeze_shared_params(small_terramind):
    """Test freezing shared parameters"""
    model = small_terramind
    
    model.freeze_shared_params()
    
    # Encoder and decoder should be frozen, but not embeddings
    for param in model.encoder.parameters():
        assert param.requires_grad is False
    
    for param in model.decoder.parameters():
        assert param.requires_grad is False
    
    for param in model.encoder_embeddings.parameters():
        assert param.requires_grad is True


def test_unfreeze_shared_params(small_terramind):
    """Test unfreezing shared parameters"""
    model = small_terramind
    
    model.freeze_shared_params()
    model.unfreeze_shared_params()
    
    # Encoder and decoder should be unfrozen
    for param in model.encoder.parameters():
        assert param.requires_grad is True
    
    for param in model.decoder.parameters():
        assert param.requires_grad is True


def test_unfreeze_all(small_terramind):
    """Test unfreezing all parameters"""
    model = small_terramind
    
    model.freeze_encoder(freeze_embeddings=True)
    model.freeze_decoder(freeze_embeddings=True)
    model.unfreeze_all()
    
    # Everything should be trainable
    for param in model.parameters():
        assert param.requires_grad is True


def test_freeze_encoder_except_specific_embeddings(small_terramind):
    """Test freezing encoder except specific embeddings"""
    model = small_terramind
    
    model.freeze_encoder_except_specific_embeddings(frozen_embedding_domain='mod1')
    
    # Encoder should be frozen
    for param in model.encoder.parameters():
        assert param.requires_grad is False
    
    # mod1 embeddings should be frozen
    for param in model.encoder_embeddings['mod1'].parameters():
        assert param.requires_grad is False
    
    # mod2 embeddings should be trainable
    for param in model.encoder_embeddings['mod2'].parameters():
        assert param.requires_grad is True


def test_freeze_decoder_except_specific_embeddings(small_terramind):
    """Test freezing decoder except specific embeddings"""
    model = small_terramind
    
    model.freeze_decoder_except_specific_embeddings(frozen_embedding_domain='mod1')
    
    # Decoder should be frozen
    for param in model.decoder.parameters():
        assert param.requires_grad is False
    
    # mod1 embeddings should be frozen
    for param in model.decoder_embeddings['mod1'].parameters():
        assert param.requires_grad is False
    
    # mod3 embeddings should be trainable
    for param in model.decoder_embeddings['mod3'].parameters():
        assert param.requires_grad is True


def test_freeze_params_except_specific_embeddings(small_terramind):
    """Test freezing all params except specific embeddings"""
    model = small_terramind
    
    model.freeze_params_except_specific_embeddings(frozen_embedding_domain='mod1-mod3')
    
    # Encoder and decoder should be frozen
    for param in model.encoder.parameters():
        assert param.requires_grad is False
    
    for param in model.decoder.parameters():
        assert param.requires_grad is False
    
    # mod1 and mod3 embeddings should be frozen
    for param in model.encoder_embeddings['mod1'].parameters():
        assert param.requires_grad is False
    
    for param in model.decoder_embeddings['mod3'].parameters():
        assert param.requires_grad is False
    
    # mod2 embeddings should be trainable
    for param in model.encoder_embeddings['mod2'].parameters():
        assert param.requires_grad is True


def test_shared_drop_path_encoder_decoder(modality_info, encoder_embeddings, decoder_embeddings):
    """Test shared drop path between encoder and decoder"""
    model = TerraMind(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        modality_info=modality_info,
        dim=64,
        encoder_depth=3,
        decoder_depth=3,
        num_heads=4,
        drop_path_rate_encoder=0.1,
        drop_path_rate_decoder=0.2,
        shared_drop_path=True,
    )
    
    # Model should initialize without errors
    assert len(model.encoder) == 3
    assert len(model.decoder) == 3


def test_independent_drop_path(modality_info, encoder_embeddings, decoder_embeddings):
    """Test independent drop path for encoder and decoder"""
    model = TerraMind(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        modality_info=modality_info,
        dim=64,
        encoder_depth=3,
        decoder_depth=3,
        num_heads=4,
        drop_path_rate_encoder=0.1,
        drop_path_rate_decoder=0.2,
        shared_drop_path=False,
    )
    
    # Model should initialize without errors
    assert len(model.encoder) == 3
    assert len(model.decoder) == 3


def test_build_tokenizer():
    """Test build_tokenizer function"""
    # Mock tokenizer
    class MockTokenizer(nn.Module):
        def __init__(self, pretrained=True):
            super().__init__()
            self.pretrained = pretrained
    
    tokenizer_dict = {
        'mod1': MockTokenizer,
        'mod2': MockTokenizer,
    }
    
    input_modalities = ['mod1']
    output_modalities = ['mod2']
    
    tokenizers = build_tokenizer(tokenizer_dict, input_modalities, output_modalities, pretrained=True)
    
    assert isinstance(tokenizers, nn.ModuleDict)
    assert 'mod1' in tokenizers
    assert 'mod2' in tokenizers
    assert tokenizers['mod1'].pretrained is True


def test_build_tokenizer_no_output_modalities():
    """Test build_tokenizer without output modalities"""
    class MockTokenizer(nn.Module):
        def __init__(self, pretrained=True):
            super().__init__()
            self.pretrained = pretrained
    
    tokenizer_dict = {'mod1': MockTokenizer}
    input_modalities = ['mod1']
    
    tokenizers = build_tokenizer(tokenizer_dict, input_modalities, output_modalities=None, pretrained=False)
    
    assert 'mod1' in tokenizers
    assert tokenizers['mod1'].pretrained is False

import gc
import pdb

import pytest
import torch
from torch import nn

from terratorch.models import EncoderDecoderFactory
from terratorch.models.utils import TemporalWrapper
from terratorch.registry import BACKBONE_REGISTRY


# Define a dummy encoder for testing
class DummyEncoder(nn.Module):
    def __init__(self, out_channels=64, in_channels=3):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class DummyDictEncoder(nn.Module):
    def forward(self, x):
        B = x.shape[0]
        return {
            "mod1": torch.randn(B, 6, 16, 16, device=x.device),
            "mod2": torch.randn(B, 4, 4, 8, device=x.device),
            "mod3": torch.randn(B, 2, 128, device=x.device),
        }


@pytest.fixture
def dummy_encoder():
    return DummyEncoder()


@pytest.fixture
def dummy_dict_encoder():
    return DummyDictEncoder()


def test_temporal_wrapper_swin_forward_shapes(dummy_encoder):
    # Built-in Swin
    NUM_CHANNELS = 6
    encoder = BACKBONE_REGISTRY.build("prithvi_swin_B", out_indices=[0, 1, 2, 3])

    wrapper = TemporalWrapper(encoder)
    batch_size = 2

    # Test case 2: Valid input shape
    x = torch.randn(batch_size, NUM_CHANNELS, 4, 224, 224)  # [B, C, T, H, W]
    output = wrapper(x)
    assert [o.shape for o in output] == [
        torch.Size([2, 56, 56, 128]),
        torch.Size([2, 28, 28, 256]),
        torch.Size([2, 14, 14, 512]),
        torch.Size([2, 7, 7, 1024]),
    ]

    gc.collect()


def test_encoder_returning_dict_modalities(dummy_dict_encoder):
    B, C, T, H, W = 2, 3, 4, 16, 16
    x = torch.randn(B, C, T, H, W)
    wrapper = TemporalWrapper(dummy_dict_encoder, pooling="mean")

    out_list = wrapper(x)
    assert isinstance(out_list, list) and len(out_list) == 1
    out = out_list[0]
    assert set(out.keys()) == {"mod1", "mod2", "mod3"}

    assert out["mod1"].shape == (B, 6, 16, 16)
    assert out["mod2"].shape == (B, 4, 4, 8)
    assert out["mod3"].shape == (B, 2, 128)
    gc.collect()


def test_temporal_wrapper_initialization(dummy_encoder):
    # Test valid initialization with default parameters
    wrapper = TemporalWrapper(dummy_encoder)
    assert wrapper.pooling == "mean"

    # Test valid initialization with custom parameters
    wrapper = TemporalWrapper(dummy_encoder, pooling="max")
    assert wrapper.pooling == "max"

    # Test invalid pooling type
    with pytest.raises(ValueError, match="Unsupported pooling 'invalid'"):
        TemporalWrapper(dummy_encoder, pooling="invalid")

    gc.collect()


def test_temporal_wrapper_forward_shapes(dummy_encoder):
    wrapper = TemporalWrapper(dummy_encoder)
    batch_size = 2

    # Test case 1: Invalid number of dimensions
    x = torch.randn(batch_size, 3, 32, 32)  # 4D tensor
    with pytest.raises(ValueError, match=r"Expected input shape \[B, C, T, H, W\]"):
        wrapper(x)

    # Test case 2: Valid input shape
    x = torch.randn(batch_size, 3, 4, 32, 32)  # [B, C, T, H, W]
    output = wrapper(x)
    assert isinstance(output, list)
    assert len(output) == 1  # Single feature map
    assert output[0].shape == (batch_size, dummy_encoder.out_channels, 32, 32)

    # Test case 3: Different number of timepoints
    x = torch.randn(batch_size, 3, 6, 32, 32)  # 6 timepoints
    output = wrapper(x)
    assert isinstance(output, list)
    assert len(output) == 1
    assert output[0].shape == (batch_size, dummy_encoder.out_channels, 32, 32)

    # Test case 4: Invalid number of input channels
    x = torch.randn(batch_size, 4, 3, 32, 32)  # 4 channels
    with pytest.raises(RuntimeError):
        wrapper(x)  # Should fail because Conv2d expects 3 channels

    gc.collect()


def test_temporal_wrapper_pooling_modes(dummy_encoder):
    batch_size = 2
    timesteps = 4

    ### conv-like features
    n_channels = 3
    x = torch.randn(batch_size, n_channels, timesteps, 224, 224)
    encoder = BACKBONE_REGISTRY.build("timm_resnet50")

    # Test mean pooling
    wrapper = TemporalWrapper(encoder, pooling="mean")
    output = wrapper(x)
    assert isinstance(output, list)
    assert len(output) == 5
    assert output[0].shape == (batch_size, encoder.out_channels[0], 112, 112)
    del wrapper, output

    gc.collect()

    # Test max pooling
    wrapper = TemporalWrapper(encoder, pooling="max")
    output = wrapper(x)
    assert isinstance(output, list)
    assert len(output) == 5
    assert output[0].shape == (batch_size, encoder.out_channels[0], 112, 112)
    del wrapper, output

    gc.collect()

    # Test concatenation
    wrapper = TemporalWrapper(encoder, pooling="concat")
    output = wrapper(x)
    assert isinstance(output, list)
    assert len(output) == 5
    # For concatenation, channels should be multiplied by number of timesteps
    assert output[0].shape == (batch_size, encoder.out_channels[0] * timesteps, 112, 112)
    del wrapper, output

    gc.collect()

    # Test diff
    wrapper = TemporalWrapper(encoder, pooling="diff")
    output = wrapper(x[:, :, [0, 1], ...])
    assert isinstance(output, list)
    assert len(output) == 5
    assert output[0].shape == (batch_size, encoder.out_channels[0], 112, 112)
    del wrapper, output

    gc.collect()

    ### transformer-like features
    encoder = BACKBONE_REGISTRY.build("clay_v1_base")
    n_channels = 6
    x = torch.randn(batch_size, n_channels, timesteps, 256, 256)
    n_tokens = 1025
    # Test mean pooling
    wrapper = TemporalWrapper(encoder, pooling="mean")
    output = wrapper(x)
    assert isinstance(output, list)
    assert len(output) == 12
    assert output[0].shape == (batch_size, n_tokens, encoder.out_channels[0])
    del wrapper, output

    gc.collect()

    # Test max pooling
    wrapper = TemporalWrapper(encoder, pooling="max")
    output = wrapper(x)
    assert isinstance(output, list)
    assert len(output) == 12
    assert output[0].shape == (batch_size, n_tokens, encoder.out_channels[0])
    del wrapper, output

    gc.collect()

    # Test concatenation
    wrapper = TemporalWrapper(encoder, pooling="concat")
    output = wrapper(x)
    assert isinstance(output, list)
    assert len(output) == 12
    # For concatenation, channels should be multiplied by number of timesteps
    assert output[0].shape == (batch_size, n_tokens, encoder.out_channels[0] * timesteps)
    del output, wrapper

    gc.collect()

    # Test diff
    wrapper = TemporalWrapper(encoder, pooling="diff")
    output = wrapper(x[:, :, [0, 1], ...])
    assert isinstance(output, list)
    assert len(output) == 12
    assert output[0].shape == (batch_size, n_tokens, encoder.out_channels[0])
    del output, wrapper

    # gc.collect()

    # ### Swin-like features
    # encoder = BACKBONE_REGISTRY.build(
    #     "satlas_swin_b_sentinel2_si_ms", model_bands=[0, 1, 2, 3, 4, 5], out_indices=[1, 3, 5, 7]
    # )

    n_channels = 6
    x = torch.randn(batch_size, n_channels, timesteps, 256, 256)
    # pdb.set_trace()
    # Test mean pooling
    wrapper = TemporalWrapper(encoder, pooling="mean", features_permute_op=(0, 3, 1, 2))
    output = wrapper(x)
    assert isinstance(output, list)
    assert len(output) == 4
    assert output[0].shape == (batch_size, 64, 64, encoder.out_channels[0])
    del output, wrapper

    gc.collect()

    # Test max pooling
    wrapper = TemporalWrapper(encoder, pooling="max", features_permute_op=(0, 3, 1, 2))
    output = wrapper(x)
    assert isinstance(output, list)
    assert len(output) == 4
    assert output[0].shape == (batch_size, 64, 64, encoder.out_channels[0])
    del output, wrapper

    gc.collect()

    # Test concatenation
    wrapper = TemporalWrapper(encoder, pooling="concat", features_permute_op=(0, 3, 1, 2))
    output = wrapper(x)
    assert isinstance(output, list)
    assert len(output) == 4
    # For concatenation, channels should be multiplied by number of timesteps
    assert output[0].shape == (batch_size, 64, 64, encoder.out_channels[0] * timesteps)
    del output, wrapper

    gc.collect()

    # Test diff
    wrapper = TemporalWrapper(encoder, pooling="diff", features_permute_op=(0, 3, 1, 2))
    output = wrapper(x[:, :, [0, 1], ...])
    assert isinstance(output, list)
    assert len(output) == 4
    assert output[0].shape == (batch_size, 64, 64, 128)
    del output, wrapper

    gc.collect()


@pytest.mark.parametrize("pooling", ["mean", "concat"])
def test_temporal_encoder_decoder_factory(pooling):
    model = EncoderDecoderFactory().build_model(
        task="classification",
        backbone_use_temporal=True,
        backbone_temporal_pooling=pooling,
        backbone_temporal_n_timestamps=3,
        backbone="terramind_v1_tiny",
        backbone_modalities=["S2L2A"],
        decoder="IdentityDecoder",
        backbone_pretrained=False,
        num_classes=2,
        necks=[{"name": "AggregateTokens"}],
    )

    x = torch.randn(2, 12, 3, 224, 224)  # [B, C, T, H, W]
    output = model(x).output

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 2)

    del model, output

    gc.collect()

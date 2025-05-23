import pytest
import torch
from torch import nn
from terratorch.models.encoder_decoder_factory import TemporalWrapper
from terratorch.registry import BACKBONE_REGISTRY

# Define a dummy encoder for testing
class DummyEncoder(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(3, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x)

@pytest.fixture
def dummy_encoder():
    return DummyEncoder()

def test_temporal_wrapper_swin_forward_shapes(dummy_encoder):

    # Built-in Swin
    NUM_CHANNELS = 6
    encoder = BACKBONE_REGISTRY.build("prithvi_swin_B",
                                      out_indices=[0,1,2,3])

    wrapper = TemporalWrapper(encoder)
    batch_size = 2

    # Test case 2: Valid input shape
    x = torch.randn(batch_size, NUM_CHANNELS, 4, 224, 224)  # [B, C, T, H, W]
    output = wrapper(x)
    assert [o.shape for o in output] == [torch.Size([2, 56, 56, 128]), torch.Size([2, 28, 28, 256]), torch.Size([2, 14, 14, 512]), torch.Size([2, 7, 7, 1024])]

    # Satlas Swin
    NUM_CHANNELS = 6
    encoder = BACKBONE_REGISTRY.build("satlas_swin_b_sentinel2_si_ms",
                                      model_bands=[0,1,2,3,4,5], out_indices=[1,3,5,7])

    wrapper = TemporalWrapper(encoder)
    batch_size = 2

    # Test case 2: Valid input shape
    x = torch.randn(batch_size, NUM_CHANNELS, 4, 224, 224)  # [B, C, T, H, W]
    output = wrapper(x)
    assert [o.shape for o in output] == [torch.Size([2, 56, 56, 128]), torch.Size([2, 28, 28, 256]), torch.Size([2, 14, 14, 512]), torch.Size([2, 7, 7, 1024])]

def test_temporal_wrapper_initialization(dummy_encoder):
    # Test valid initialization with default parameters
    wrapper = TemporalWrapper(dummy_encoder)
    assert wrapper.pooling_type == "mean"
    assert not wrapper.concat
    assert wrapper.out_channels == dummy_encoder.out_channels

    # Test valid initialization with custom parameters
    wrapper = TemporalWrapper(dummy_encoder, pooling="max", concat=True)
    assert wrapper.pooling_type == "max"
    assert wrapper.concat
    assert wrapper.out_channels == dummy_encoder.out_channels * dummy_encoder.out_channels

    # Test invalid pooling type
    with pytest.raises(ValueError, match="Pooling must be 'mean' or 'max'"):
        TemporalWrapper(dummy_encoder, pooling="invalid")

    # Test encoder without out_channels attribute
    class InvalidEncoder(nn.Module):
        def forward(self, x):
            return x
    
    with pytest.raises(AttributeError, match="Encoder must have an `out_channels` attribute"):
        TemporalWrapper(InvalidEncoder())

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

def test_temporal_wrapper_pooling_modes(dummy_encoder):
    batch_size = 2
    timesteps = 4
    x = torch.randn(batch_size, 3, timesteps, 32, 32)

    # Test mean pooling
    mean_wrapper = TemporalWrapper(dummy_encoder, pooling="mean")
    mean_output = mean_wrapper(x)
    assert isinstance(mean_output, list)
    assert len(mean_output) == 1
    assert mean_output[0].shape == (batch_size, dummy_encoder.out_channels, 32, 32)

    # Test max pooling
    max_wrapper = TemporalWrapper(dummy_encoder, pooling="max")
    max_output = max_wrapper(x)
    assert isinstance(max_output, list)
    assert len(max_output) == 1
    assert max_output[0].shape == (batch_size, dummy_encoder.out_channels, 32, 32)

    # Test concatenation
    concat_wrapper = TemporalWrapper(dummy_encoder, concat=True)
    concat_output = concat_wrapper(x)
    assert isinstance(concat_output, list)
    assert len(concat_output) == 1
    # For concatenation, channels should be multiplied by number of timesteps
    assert concat_output[0].shape == (batch_size, dummy_encoder.out_channels * timesteps, 32, 32)

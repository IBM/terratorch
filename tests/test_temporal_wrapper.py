import gc
import pytest
import torch
from torch import nn

from terratorch.models import EncoderDecoderFactory
from terratorch.models.utils import TemporalWrapper
from terratorch.registry import BACKBONE_REGISTRY

class DummyEncoder(nn.Module):
    """
    Minimal CNN-like encoder returning a single 4D feature map (B, C, H, W).
    Used to validate shape handling and temporal pooling behavior.
    """
    def __init__(self, out_channels=64, in_channels=3):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

class DummyDictEncoder(nn.Module):
    """
    Minimal multi-modal encoder returning a dict of random feature maps.
    Used to test the dict-handling branch of TemporalWrapper.
    """
    def forward(self, x):
        B = x.shape[0]
        return {
            "mod1": torch.randn(B, 6, 16, 16, device=x.device),
            "mod2": torch.randn(B, 4, 4, 8, device=x.device),
            "mod3": torch.randn(B, 2, 128, device=x.device),
        }

class IdentityEncoder(nn.Module):
    def __init__(self, out_channels=3, in_channels=3):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x):
        # no spatial change, no learned weights
        return x

@pytest.fixture
def dummy_encoder():
    """Provide a simple CNN-like encoder instance."""
    return DummyEncoder()


@pytest.fixture
def dummy_dict_encoder():
    """Provide a simple multi-modal encoder instance."""
    return DummyDictEncoder()


def test_subset_lengths_aggregation_diff(dummy_encoder):
    # diff pooling with two subsets [1, 2] on T=3: mean per subset then difference
    B, C, T, H, W = 2, 3, 3, 16, 16
    x = torch.randn(B, C, T, H, W)
    wrapper = TemporalWrapper(dummy_encoder, pooling="diff", subset_lengths=[1, 2])
    out = wrapper(x)[0]

    # Should be a single timestep output (after subset aggregation then diff)
    assert out.shape == (B, dummy_encoder.out_channels, H, W)

def test_invalid_subset_lengths_for_diff(dummy_encoder):
    # check that diff pooling supports exactly two subsets
    B, C, T, H, W = 1, 3, 3, 8, 8
    x = torch.randn(B, C, T, H, W)

    with pytest.raises(ValueError, match="exactly two subsets"):
        TemporalWrapper(dummy_encoder, pooling="diff", subset_lengths=[1, 1, 1])(x)

def test_subset_lengths_sum_mismatch(dummy_encoder):
    # subset_lengths must sum to T
    B, C, T, H, W = 1, 3, 4, 8, 8
    x = torch.randn(B, C, T, H, W)

    wrapper = TemporalWrapper(dummy_encoder, pooling="mean", subset_lengths=[1, 1])
    with pytest.raises(ValueError, match="must equal timesteps"):
        wrapper(x)

def test_diff_subset_pooling_logic_identity():
    # test numerical output with Identity Encoder
    B, C, T, H, W = 1, 1, 3, 1, 1
    x = torch.tensor([[[[[1.0]], [[3.0]], [[5.0]]]]])  # shape: (1, 1, 3, 1, 1)

    decoder = IdentityEncoder()
    wrapper = TemporalWrapper(decoder, pooling="diff", subset_lengths=[1, 2])

    out = wrapper(x)[0]
    expected = torch.tensor([[[[1.0 - 4.0]]]])

    assert torch.allclose(out, expected, atol=1e-6)

def test_temporal_wrapper_initialization(dummy_encoder):
    """
    Verify correct initialization behavior:
      - Default pooling type is 'mean'
      - Custom pooling types are accepted
      - Invalid pooling raises ValueError
    """
    wrapper = TemporalWrapper(dummy_encoder)
    assert wrapper.pooling == "mean"

    wrapper = TemporalWrapper(dummy_encoder, pooling="max")
    assert wrapper.pooling == "max"

    with pytest.raises(ValueError, match="Unsupported pooling 'invalid'"):
        TemporalWrapper(dummy_encoder, pooling="invalid")


def test_permute_and_reverse_operations(dummy_encoder):
    """Check that applying and reversing a permutation recovers the original tensor."""
    x = torch.randn(2, 3, 4, 32, 32)
    wrapper = TemporalWrapper(dummy_encoder, pooling="mean", features_permute_op=[0, 2, 1, 3])

    flat = x.permute(0, 2, 1, 3, 4).reshape(2 * 4, 3, 32, 32)
    permuted = wrapper.permute_op(flat, wrapper.features_permute_op)
    recovered = wrapper.permute_op(permuted, wrapper.reverse_permute_op)
    assert torch.allclose(flat, recovered, atol=1e-6)


def test_vit_postprocess_removes_helper_dim(dummy_encoder):
    """Ensure vit_postprocess correctly removes helper dim only when needed."""
    wrapper = TemporalWrapper(dummy_encoder)

    # Case 1: already fine (no trailing helper dim)
    t1 = torch.randn(2, 8, 128)
    assert torch.equal(wrapper.vit_postprocess(t1), t1)

    # Case 2: with trailing helper dim
    t2 = torch.randn(2, 128, 16, 1)
    post = wrapper.vit_postprocess(t2)
    assert post.dim() in {3, 4} and post.shape[-1] != 1


def test_deprecated_concat_argument_warning(dummy_encoder):
    """Using deprecated 'concat' argument should raise a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="deprecated"):
        TemporalWrapper(dummy_encoder, concat=True)


def test_temporal_wrapper_forward_shapes(dummy_encoder):
    """
    Validate input dimensionality, temporal reshaping, and pooling output shapes
    for a CNN-like encoder.

    Covers:
      - Invalid 4D input (ValueError)
      - Valid 5D input with mean pooling
      - Variable timesteps (T)
      - Channel mismatch (RuntimeError)
    """
    wrapper = TemporalWrapper(dummy_encoder)
    batch_size = 2

    # Case 1: invalid input (4D)
    x = torch.randn(batch_size, 3, 32, 32)
    with pytest.raises(ValueError, match=r"Expected input shape \[B, C, T, H, W\]"):
        wrapper(x)

    # Case 2: valid 5D input, default pooling (mean)
    x = torch.randn(batch_size, 3, 4, 32, 32)
    output = wrapper(x)
    assert isinstance(output, list)
    assert output[0].shape == (batch_size, dummy_encoder.out_channels, 32, 32)

    # Case 3: different timesteps
    x = torch.randn(batch_size, 3, 6, 32, 32)
    output = wrapper(x)
    assert output[0].shape == (batch_size, dummy_encoder.out_channels, 32, 32)

    # Case 4: invalid channel count for Conv2d
    x = torch.randn(batch_size, 4, 3, 32, 32)
    with pytest.raises(RuntimeError):
        wrapper(x)


def test_encoder_returning_dict_modalities(dummy_dict_encoder):
    """
    Verify that the wrapper correctly handles encoders returning dict outputs.

    Checks:
      - Per-modality shape, pooling, and postprocess logic
      - All modalities preserved
    """
    B, C, T, H, W = 2, 3, 4, 16, 16
    x = torch.randn(B, C, T, H, W)
    wrapper = TemporalWrapper(dummy_dict_encoder, pooling="mean")

    out_list = wrapper(x)
    assert isinstance(out_list, list) and len(out_list) == 1

    out = out_list[0]
    assert set(out.keys()) == {"mod1", "mod2", "mod3"}

    # Validate per-modality shapes
    assert out["mod1"].shape == (B, 6, 16, 16)
    assert out["mod2"].shape == (B, 4, 4, 8)
    assert out["mod3"].shape == (B, 2, 128)


def test_temporal_wrapper_pooling_modes():
    """
    Test all supported pooling strategies for CNN-, ViT-, and Swin-style encoders.

    Covers:
      - Pooling variants ('mean', 'max', 'concat', 'diff')
      - Output list consistency and shape validation
      - features_permute_op for Swin backbones
    """
    batch_size, timesteps = 2, 4 # Randomly chosen


    # CNN-like backbone (ResNet18)
    #
    encoder = BACKBONE_REGISTRY.build("timm_resnet18")

    # Sample (B,C,T,H,W) input
    x = torch.randn(batch_size, 3, timesteps, 224, 224)

    for pooling in ["mean", "max", "concat", "diff"]:
        wrapper = TemporalWrapper(encoder,
                                  pooling=pooling,
                                  n_timestamps=timesteps)

        x_in = x if pooling != "diff" else x[:, :, [0, 1], ...] #Diff operates 2 timesteps
        output = wrapper(x_in)

        assert isinstance(output, list)
        assert len(output) == 5 #Expect a list of 5 outputs (intermediate layers)

        expected_c = encoder.out_channels[0] * timesteps if pooling == "concat" else encoder.out_channels[0] #Concat keeps timesteps and appends along channel dim, hence we multiply expected channel count by number of input timesteps
        assert output[0].shape == (batch_size, expected_c, 112, 112)
        gc.collect()


    # ViT-like backbone (clay_v1_base)
    #
    encoder = BACKBONE_REGISTRY.build("clay_v1_base")
    n_channels, n_tokens = 6, 1025

    # Sample (B,C,T,H,W) input
    x = torch.randn(batch_size, n_channels, timesteps, 256, 256)

    for pooling in ["mean", "max", "concat", "diff"]:
        wrapper = TemporalWrapper(encoder,
                                  pooling=pooling,
                                  n_timestamps=timesteps)

        x_in = x if pooling != "diff" else x[:, :, [0, 1], ...]
        output = wrapper(x_in)

        assert isinstance(output, list)
        assert len(output) == 12 #Expect a list of 12 outputs (intermediate layers)

        expected_c = encoder.out_channels[0] * timesteps if pooling == "concat" else encoder.out_channels[0] #Concat keeps timesteps and appends along channel dim, hence we multiply expected channel count by number of input timesteps
        assert output[0].shape == (batch_size, n_tokens, expected_c)
        gc.collect()


    # Swin-like backbone (Satlas Swin-B Sentinel-2)
    #
    encoder = BACKBONE_REGISTRY.build(
        "satlas_swin_b_sentinel2_si_ms",
        model_bands=[0, 1, 2, 3, 4, 5],
        out_indices=[1, 3, 5, 7],
    )
    n_channels = 6

    # Sample (B,C,T,H,W) input
    x = torch.randn(batch_size, n_channels, timesteps, 256, 256)

    for pooling in ["mean", "max", "concat", "diff"]:
        wrapper = TemporalWrapper(encoder,
                                  pooling=pooling,
                                  n_timestamps=timesteps,
                                  features_permute_op=(0, 3, 1, 2)) # Use feature permute op as Swin backbones output B,H,W,C but TempWrapper expects B,C,H,W

        x_in = x if pooling != "diff" else x[:, :, [0, 1], ...]
        output = wrapper(x_in)

        assert isinstance(output, list)
        assert len(output) == 4 #Expect a list of 4 outputs (intermediate layers)

        if pooling == "concat":
            expected_c = encoder.out_channels[0] * timesteps #Concat keeps timesteps and appends along channel dim, hence we multiply expected channel count by number of input timesteps
        else:
            expected_c = encoder.out_channels[0]

        assert output[0].shape == (batch_size, 64, 64, expected_c)
        gc.collect()

    # Swin-like backbone (Prithvi Swin-B)
    #
    encoder = BACKBONE_REGISTRY.build("prithvi_swin_B", out_indices=[0, 1, 2, 3])
    n_channels = 6
    x = torch.randn(batch_size, n_channels, timesteps, 224, 224)

    for pooling in ["mean", "max", "concat", "diff"]:

        wrapper = TemporalWrapper(encoder,
                                  pooling=pooling,
                                  n_timestamps=timesteps,
                                  features_permute_op=(0, 3, 1, 2)) # Use feature permute op as Swin backbones output B,H,W,C but TemporalWrapper expects B,C,H,W

        x_in = x if pooling != "diff" else x[:, :, [0, 1], ...]
        output = wrapper(x_in)

        assert isinstance(output, list)
        assert len(output) == 4 #Expect a list of 4 outputs (intermediate layers)

        if pooling == "concat":
            expected_c = encoder.out_channels[0] * timesteps #Concat keeps timesteps and appends along channel dim, hence we multiply expected channel count by number of input timesteps
        else:
            expected_c = encoder.out_channels[0]

        assert output[0].shape == (batch_size, 56, 56, expected_c)
        gc.collect()


# Integration Test
@pytest.mark.parametrize("pooling", ["mean", "concat"])
def test_temporal_encoder_decoder_factory(pooling):
    """
    Integration test for EncoderDecoderFactory with temporal backbones.

    Verifies that temporal pooling options ('mean', 'concat') correctly integrate
    into an end-to-end classification pipeline using a lightweight Terramind encoder.
    """
    batch_size, n_channels, timesteps, height, width = 2, 12, 3, 224, 224

    model = EncoderDecoderFactory().build_model(
        task="classification",
        backbone_use_temporal=True,
        backbone_temporal_pooling=pooling,
        backbone_temporal_n_timestamps=timesteps,
        backbone="terramind_v1_tiny",
        backbone_modalities=["S2L2A"],
        decoder="IdentityDecoder",
        backbone_pretrained=False,
        num_classes=2,
        necks=[{"name": "AggregateTokens"}],
    )

    # Simulate multi-band Sentinel-2 input [B, C, T, H, W]
    x = torch.randn(batch_size, n_channels, timesteps, height, width)
    output = model(x).output

    assert isinstance(output, torch.Tensor), "Model output should be a Tensor"
    assert output.shape == (batch_size, 2), f"Unexpected output shape {output.shape}"

    del model, output
    gc.collect()

def test_temporal_wrapper_subset_diff_full():
    model = EncoderDecoderFactory().build_model(
        task="classification",
        backbone_use_temporal=True,
        backbone_temporal_pooling="diff",
        backbone_temporal_subset_lengths=[1, 2],
        backbone="terramind_v1_tiny",
        backbone_modalities=["S2L2A"],
        decoder="IdentityDecoder",
        num_classes=2,
        necks=[{"name": "AggregateTokens"}],
    )

    x = torch.randn(2, 12, 3, 224, 224)
    out = model(x).output

    assert out.shape == (2, 2)

    gc.collect()
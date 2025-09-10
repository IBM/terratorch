import pytest
import torch
from torch import nn

from terratorch.models.backbones.terramind.tokenizer.models.uvit import DropPath, Mlp

IN_FEATURES = 10


def mock_drop_path(x, drop_prob, training):
    # Mock the drop_path function for testing purposes
    return x * (1 - drop_prob) if training else x


@pytest.fixture
def create_drop_path(drop_prob=0.1, training=True):
    if not training:
        drop_prob = 0

    def _create_drop_path(drop_prob=drop_prob, training=training):
        return DropPath(drop_prob)

    yield _create_drop_path

    # Clean up after the test
    mock_drop_path.call_args_list = []


def create_mlp(in_features=IN_FEATURES, temb_dim=None, hidden_features=None, out_features=None, drop=0.0):
    return Mlp(in_features, temb_dim, hidden_features, out_features, drop=0)


def test_forward_when_training(create_drop_path):
    drop_path = create_drop_path()

    input_tensor = torch.randn(2, 3)
    output_tensor = drop_path(input_tensor)

    assert output_tensor.shape == input_tensor.shape
    assert (output_tensor != input_tensor).sum() > 0  # Check if some elements were dropped


def test_forward():
    mlp = create_mlp()

    input_tensor = torch.randn(2, 3, IN_FEATURES)
    output_tensor = mlp(input_tensor)

    assert output_tensor.shape[:2] == input_tensor.shape[:2]  # Check batch dimensions
    assert output_tensor.shape[1:] == mlp.out_features  # Check output features

    # Add more assertions based on your specific requirements


def test_forward_with_temb():
    mlp = create_mlp(temb_dim=6)  # Assuming temb_dim is provided

    input_tensor = torch.randn(2, 3, IN_FEATURES)
    temb = torch.randn(2, mlp.hidden_features // 2)  # Mock time embedding tensor
    output_tensor = mlp(input_tensor, temb)

    assert output_tensor.shape[:2] == input_tensor.shape[:2]  # Check batch dimensions
    assert output_tensor.shape[1:] == mlp.out_features  # Check output features

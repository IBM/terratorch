import pytest
import torch
from terratorch.models.backbones.dinov3.dinov3_wrapper import DinoV3Wrapper
import pdb

# Fixture for dummy input
@pytest.fixture
def dummy_input():
    return torch.randn(1, 3, 224, 224)  # Batch size 1, RGB, 224x224

# Fixture for model
@pytest.fixture
def model():
    return DinoV3Wrapper(model="dinov3_vits16", img_size=224)

# Test model initialization
def test_model_initialization(model):
    assert model.dinov3 is not None
    assert hasattr(model.dinov3, "get_intermediate_layers")

# Test forward pass
def test_forward_features(model, dummy_input):
    output = model.forward(dummy_input)
    assert isinstance(output, list)
    assert isinstance(output[0], torch.Tensor)
  # batch size match

# Test output shape consistency
def test_output_shape(model, dummy_input):

    output = model.forward(dummy_input)
    # Depending on the model, output shape may vary. Adjust as needed.
    assert len(output) == 12
    assert output[0].shape[0] == 1
    assert output[0].shape[1] == 197
    assert output[0].shape[2] == 384

    model.return_cls_token = False

    output = model.forward(dummy_input)
    assert output[0].shape[1] == 196

# Test model with different image size
def test_model_different_img_size():
    # pdb.set_trace()
    model = DinoV3Wrapper(model="dinov3_vits16", img_size=256)
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model.forward(dummy_input)
    assert output[0].shape[0] == 1

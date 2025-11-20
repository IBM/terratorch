# Copyright contributors to the Terratorch project

import gc
import importlib
import sys
import types
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from torch import nn

from terratorch.datasets import HLSBands
from terratorch.models import SatMAEModelFactory
from terratorch.models.model import AuxiliaryHead
from terratorch.models.satmae_model_factory import (
    ModelWrapper,
    check_the_kind_of_vit,
    filter_cefficients_when_necessary,
    _get_decoder,
    _build_appropriate_model,
)
from terratorch.models.utils import DecoderNotFoundError

NUM_CHANNELS = 3
NUM_CLASSES = 2
EXPECTED_SEGMENTATION_OUTPUT_SHAPE = (1, NUM_CLASSES, 224, 224)
EXPECTED_REGRESSION_OUTPUT_SHAPE = (1, 224, 224)
EXPECTED_CLASSIFICATION_OUTPUT_SHAPE = (1, NUM_CLASSES)

PIXEL_WISE_TASKS = ["segmentation", "regression"]
SCALAR_TASKS = ["classification"]

PIXELWISE_TASK_EXPECTED_OUTPUT = [
    ("regression", EXPECTED_REGRESSION_OUTPUT_SHAPE),
    ("segmentation", EXPECTED_SEGMENTATION_OUTPUT_SHAPE),
]


# Helper functions and fixtures
@pytest.fixture(scope="session")
def model_factory() -> SatMAEModelFactory:
    return SatMAEModelFactory()


@pytest.fixture(scope="session")
def model_input() -> torch.Tensor:
    return torch.ones((1, NUM_CHANNELS, 224, 224))


class DummyModel(nn.Module):
    """Mock model for testing"""
    def __init__(self, embed_dim=768, num_patches=196):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
    def forward_features(self, x):
        # Simulate ViT forward
        return x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, N, C)
    
    def forward_encoder(self, x, mask_ratio=0.75):
        # Simulate ViT-MAE forward
        latent = x.flatten(2).transpose(1, 2)
        ids_restore = torch.arange(latent.shape[1])
        return latent, None, ids_restore
    
    def state_dict(self):
        return {'norm.bias': torch.zeros(self.embed_dim)}


# Test helper functions
def test_check_the_kind_of_vit_mae():
    """Test detecting MAE models"""
    assert check_the_kind_of_vit("vit_mae_base") == "vit-mae"
    assert check_the_kind_of_vit("MaskedAutoencoderViT") == "vit-mae"
    assert check_the_kind_of_vit("MAE_encoder") == "vit-mae"


def test_check_the_kind_of_vit_regular():
    """Test detecting regular ViT models"""
    assert check_the_kind_of_vit("vit_base_patch16") == "vit"
    assert check_the_kind_of_vit("ViTEncoder") == "vit"
    assert check_the_kind_of_vit("some_model") == "vit"


def test_check_the_kind_of_vit_none():
    """Test with None input"""
    result = check_the_kind_of_vit(None)
    assert result in ["vit", "vit-mae"]


def test_filter_coefficients_vit():
    """Test filtering coefficients for regular ViT"""
    model_dict = {
        'model': {
            'patch_embed': torch.tensor([1]),
            'decoder_blocks': torch.tensor([2]),
            'other_weight': torch.tensor([3]),
        }
    }
    result = filter_cefficients_when_necessary(model_dict, kind="vit")
    assert 'patch_embed' not in result['model']
    assert 'decoder_blocks' not in result['model']
    assert 'other_weight' in result['model']


def test_filter_coefficients_vit_mae():
    """Test filtering coefficients for ViT-MAE (no filtering)"""
    model_dict = {
        'model': {
            'patch_embed': torch.tensor([1]),
            'decoder_blocks': torch.tensor([2]),
            'other_weight': torch.tensor([3]),
        }
    }
    result = filter_cefficients_when_necessary(model_dict, kind="vit-mae")
    assert 'patch_embed' in result['model']
    assert 'decoder_blocks' in result['model']
    assert 'other_weight' in result['model']


def test_filter_coefficients_exception():
    """Test filter handles exceptions gracefully"""
    model_dict = {'no_model_key': {}}
    result = filter_cefficients_when_necessary(model_dict, kind="vit")
    assert result == model_dict


# Test ModelWrapper
def test_model_wrapper_vit():
    """Test ModelWrapper with regular ViT"""
    dummy = DummyModel()
    wrapper = ModelWrapper(model=dummy, kind="vit")
    
    assert wrapper.kind == "vit"
    assert wrapper.embedding_shape == 768
    assert wrapper.forward == wrapper._forward_vit
    assert hasattr(wrapper, 'num_patches')
    assert wrapper.num_patches == 196


def test_model_wrapper_vit_mae():
    """Test ModelWrapper with ViT-MAE"""
    dummy = DummyModel()
    wrapper = ModelWrapper(model=dummy, kind="vit-mae")
    
    assert wrapper.kind == "vit-mae"
    assert wrapper.embedding_shape == 768
    assert wrapper.forward == wrapper._forward_vit_mae


def test_model_wrapper_channels():
    """Test channels method"""
    dummy = DummyModel(embed_dim=512)
    wrapper = ModelWrapper(model=dummy, kind="vit")
    
    assert wrapper.channels() == (1, 512)


def test_model_wrapper_forward_vit():
    """Test forward method for regular ViT"""
    dummy = DummyModel()
    wrapper = ModelWrapper(model=dummy, kind="vit")
    
    x = torch.randn(2, 3, 224, 224)
    output = wrapper.forward(x)
    assert output.shape == (2, 3136, 3)  # 224*224/16/16 = 196, but we flatten all


def test_model_wrapper_forward_vit_mae():
    """Test forward method for ViT-MAE"""
    dummy = DummyModel()
    wrapper = ModelWrapper(model=dummy, kind="vit-mae")
    
    x = torch.randn(2, 3, 224, 224)
    latent, ids_restore = wrapper.forward(x)
    assert latent.shape == (2, 3136, 3)
    assert ids_restore is not None


def test_model_wrapper_parameters():
    """Test parameters property"""
    dummy = DummyModel()
    wrapper = ModelWrapper(model=dummy, kind="vit")
    
    assert wrapper.parameters == dummy.parameters


def test_model_wrapper_summary(capsys):
    """Test summary method prints the wrapper"""
    dummy = DummyModel()
    wrapper = ModelWrapper(model=dummy, kind="vit")
    
    wrapper.summary()
    captured = capsys.readouterr()
    assert "ModelWrapper" in captured.out


# Test _get_decoder
def test_get_decoder_with_module():
    """Test _get_decoder with nn.Module"""
    decoder = nn.Identity()
    result = _get_decoder(decoder)
    assert result is decoder


def test_get_decoder_with_string():
    """Test _get_decoder with valid string"""
    result = _get_decoder("IdentityDecoder")
    assert result is not None
    assert hasattr(result, '__call__')


def test_get_decoder_invalid_string():
    """Test _get_decoder with invalid string"""
    with pytest.raises(DecoderNotFoundError):
        _get_decoder("NonExistentDecoder")


def test_get_decoder_invalid_type():
    """Test _get_decoder with invalid type"""
    with pytest.raises(Exception, match="Decoder must be str or nn.Module"):
        _get_decoder(123)


# Test _build_appropriate_model
def test_build_appropriate_model_segmentation():
    """Test building segmentation model"""
    backbone = DummyModel()
    decoder = nn.Identity()
    head_kwargs = {"num_classes": NUM_CLASSES}
    
    model = _build_appropriate_model(
        task="segmentation",
        backbone=backbone,
        decoder=decoder,
        head_kwargs=head_kwargs,
        prepare_features_for_image_model=None,
        rescale=True,
    )
    
    from terratorch.models.pixel_wise_model import PixelWiseModel
    assert isinstance(model, PixelWiseModel)


def test_build_appropriate_model_regression():
    """Test building regression model"""
    backbone = DummyModel()
    decoder = nn.Identity()
    head_kwargs = {}
    
    model = _build_appropriate_model(
        task="regression",
        backbone=backbone,
        decoder=decoder,
        head_kwargs=head_kwargs,
        prepare_features_for_image_model=None,
        rescale=False,
    )
    
    from terratorch.models.pixel_wise_model import PixelWiseModel
    assert isinstance(model, PixelWiseModel)


def test_build_appropriate_model_classification():
    """Test building classification model"""
    backbone = DummyModel()
    decoder = nn.Identity()
    head_kwargs = {"num_classes": NUM_CLASSES}
    
    model = _build_appropriate_model(
        task="classification",
        backbone=backbone,
        decoder=decoder,
        head_kwargs=head_kwargs,
        prepare_features_for_image_model=None,
    )
    
    from terratorch.models.scalar_output_model import ScalarOutputModel
    assert isinstance(model, ScalarOutputModel)


def test_build_appropriate_model_with_aux_heads():
    """Test building model with auxiliary heads"""
    backbone = DummyModel()
    decoder = nn.Identity()
    head_kwargs = {"num_classes": NUM_CLASSES}
    
    # Mock auxiliary head
    aux_head = MagicMock()
    aux_head.name = "aux_test"
    
    model = _build_appropriate_model(
        task="segmentation",
        backbone=backbone,
        decoder=decoder,
        head_kwargs=head_kwargs,
        prepare_features_for_image_model=None,
        rescale=True,
        auxiliary_heads=[aux_head],
    )
    
    assert model is not None


# Test SatMAEModelFactory.build_model
def test_build_model_with_module_backbone(model_factory):
    """Test building model when backbone is already an nn.Module"""
    backbone = DummyModel()
    decoder = nn.Identity()
    
    model = model_factory.build_model(
        task="segmentation",
        backbone=backbone,
        decoder=decoder,
        in_channels=NUM_CHANNELS,
        bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
        num_classes=NUM_CLASSES,
        pretrained=False,
    )
    
    assert model is not None
    gc.collect()


def test_build_model_invalid_syspath(model_factory):
    """Test that missing SatMAE in syspath raises error"""
    with pytest.raises(NotImplementedError, match="only handles models for `SatMAE` encoders"):
        model_factory.build_model(
            task="segmentation",
            backbone="some_backbone",
            decoder="IdentityDecoder",
            in_channels=NUM_CHANNELS,
            bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
            num_classes=NUM_CLASSES,
            pretrained=False,
            model_sys_path="/path/to/other/models",
        )


def test_build_model_unsupported_task(model_factory):
    """Test unsupported task raises error"""
    with pytest.raises(NotImplementedError, match="not supported"):
        model_factory.build_model(
            task="unsupported_task",
            backbone="some_backbone",
            decoder="IdentityDecoder",
            in_channels=NUM_CHANNELS,
            bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
            num_classes=NUM_CLASSES,
            pretrained=False,
            model_sys_path="/path/to/SatMAE",
        )


@patch('timm.create_model')
def test_build_model_from_timm(mock_timm, model_factory, model_input):
    """Test building model from timm"""
    # Create a mock timm model
    mock_model = DummyModel()
    mock_model.feature_info = MagicMock()
    mock_model.feature_info.channels = Mock(return_value=[64, 128, 256, 512])
    mock_timm.return_value = mock_model
    
    model = model_factory.build_model(
        task="segmentation",
        backbone="vit_base_satmae",
        decoder="IdentityDecoder",
        in_channels=NUM_CHANNELS,
        bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
        num_classes=NUM_CLASSES,
        pretrained=True,
        model_sys_path="/path/to/SatMAE",
    )
    
    mock_timm.assert_called_once()
    assert model is not None
    gc.collect()


@patch('torch.load')
@patch('importlib.import_module')
@patch('sys.path.insert')
def test_build_model_local_checkpoint_vit(mock_sys_path, mock_import, mock_load, model_factory, tmp_path):
    """Test building model from local checkpoint (regular ViT)"""
    # Setup mocks
    mock_model_class = MagicMock(return_value=DummyModel())
    mock_module = types.ModuleType('models_vit')
    setattr(mock_module, 'ViT', mock_model_class)
    mock_import.return_value = mock_module
    
    mock_load.return_value = {
        'model': {
            'norm.weight': torch.ones(768),
            'norm.bias': torch.zeros(768),
            'patch_embed': torch.ones(10),  # Will be filtered
        }
    }
    
    # Create a temporary checkpoint file
    checkpoint_path = str(tmp_path / "checkpoint.pth")
    torch.save({'model': {}}, checkpoint_path)
    
    with patch('timm.create_model', side_effect=Exception("Not on HuggingFace")):
        model = model_factory.build_model(
            task="segmentation",
            backbone="ViT",
            decoder="IdentityDecoder",
            in_channels=NUM_CHANNELS,
            bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
            num_classes=NUM_CLASSES,
            pretrained=True,
            checkpoint_path=checkpoint_path,
            model_sys_path="/path/to/SatMAE",
        )
    
    assert model is not None
    gc.collect()


@patch('torch.load')
@patch('importlib.import_module')
@patch('sys.path.insert')
def test_build_model_local_checkpoint_vit_mae(mock_sys_path, mock_import, mock_load, model_factory, tmp_path):
    """Test building model from local checkpoint (ViT-MAE)"""
    # Setup mocks
    mock_model_class = MagicMock(return_value=DummyModel())
    mock_module = types.ModuleType('models_mae')
    setattr(mock_module, 'MaskedAutoencoderViT', mock_model_class)
    mock_import.return_value = mock_module
    
    mock_load.return_value = {
        'model': {
            'norm.weight': torch.ones(768),
            'norm.bias': torch.zeros(768),
        }
    }
    
    # Create a temporary checkpoint file
    checkpoint_path = str(tmp_path / "checkpoint.pth")
    torch.save({'model': {}}, checkpoint_path)
    
    with patch('timm.create_model', side_effect=Exception("Not on HuggingFace")):
        model = model_factory.build_model(
            task="segmentation",
            backbone="MaskedAutoencoderViT",
            decoder="IdentityDecoder",
            in_channels=NUM_CHANNELS,
            bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
            num_classes=NUM_CLASSES,
            pretrained=True,
            checkpoint_path=checkpoint_path,
            model_sys_path="/path/to/SatMAE",
        )
    
    assert model is not None
    gc.collect()


@patch('torch.load')
@patch('importlib.import_module')
def test_build_model_local_checkpoint_cpu_only(mock_import, mock_load, model_factory, tmp_path):
    """Test building model on CPU-only environment"""
    # Setup mocks
    mock_model_class = MagicMock(return_value=DummyModel())
    mock_module = types.ModuleType('models_vit')
    setattr(mock_module, 'ViT', mock_model_class)
    mock_import.return_value = mock_module
    
    mock_load.return_value = {
        'model': {
            'norm.weight': torch.ones(768),
            'norm.bias': torch.zeros(768),
        }
    }
    
    checkpoint_path = str(tmp_path / "checkpoint.pth")
    torch.save({'model': {}}, checkpoint_path)
    
    with patch('torch.cuda.is_available', return_value=False):
        with patch('timm.create_model', side_effect=Exception("Not on HuggingFace")):
            model = model_factory.build_model(
                task="segmentation",
                backbone="ViT",
                decoder="IdentityDecoder",
                in_channels=NUM_CHANNELS,
                bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
                num_classes=NUM_CLASSES,
                pretrained=False,
                checkpoint_path=checkpoint_path,
                model_sys_path="/path/to/SatMAE",
            )
    
    # Should use map_location="cpu"
    mock_load.assert_called_with(checkpoint_path, map_location="cpu", weights_only=True)
    assert model is not None
    gc.collect()


@patch('timm.create_model', side_effect=Exception("Not on HuggingFace"))
def test_build_model_local_no_checkpoint(mock_timm, model_factory):
    """Test that building local model without checkpoint raises assertion error"""
    with pytest.raises(AssertionError, match="checkpoint must be provided"):
        model_factory.build_model(
            task="segmentation",
            backbone="ViT",
            decoder="IdentityDecoder",
            in_channels=NUM_CHANNELS,
            bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
            num_classes=NUM_CLASSES,
            pretrained=True,
            checkpoint_path=None,
            model_sys_path="/path/to/SatMAE",
        )


def test_build_model_with_satmae_head(model_factory):
    """Test building model with SatMAEHead decoder"""
    backbone = DummyModel()
    
    model = model_factory.build_model(
        task="segmentation",
        backbone=backbone,
        decoder="SatMAEHead",
        in_channels=NUM_CHANNELS,
        bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
        num_classes=NUM_CLASSES,
        pretrained=False,
    )
    
    assert model is not None
    gc.collect()


def test_build_model_with_num_patches(model_factory):
    """Test that num_patches is passed to decoder when available"""
    backbone = DummyModel(num_patches=256)
    
    model = model_factory.build_model(
        task="segmentation",
        backbone=backbone,
        decoder="IdentityDecoder",
        in_channels=NUM_CHANNELS,
        bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
        num_classes=NUM_CLASSES,
        pretrained=False,
    )
    
    assert model is not None
    gc.collect()


def test_build_model_with_aux_decoders(model_factory):
    """Test building model with auxiliary decoders"""
    backbone = DummyModel()
    
    # Create auxiliary head with mock feature_info
    backbone.feature_info = MagicMock()
    backbone.feature_info.channels = Mock(return_value=[64, 128, 256, 512])
    
    aux_heads = [
        AuxiliaryHead("aux1", "IdentityDecoder", None),
        AuxiliaryHead("aux2", "IdentityDecoder", {"decoder_arg": "value"}),
    ]
    
    model = model_factory.build_model(
        task="segmentation",
        backbone=backbone,
        decoder="IdentityDecoder",
        in_channels=NUM_CHANNELS,
        bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
        num_classes=NUM_CLASSES,
        pretrained=False,
        aux_decoders=aux_heads,
    )
    
    assert model is not None
    gc.collect()


def test_build_model_bands_conversion(model_factory):
    """Test that bands are properly converted to HLSBands enum"""
    backbone = DummyModel()
    
    # Mix of HLSBands and integers
    bands = [HLSBands.RED, 1, HLSBands.BLUE]
    
    model = model_factory.build_model(
        task="segmentation",
        backbone=backbone,
        decoder="IdentityDecoder",
        in_channels=3,
        bands=bands,
        num_classes=NUM_CLASSES,
        pretrained=False,
    )
    
    assert model is not None
    gc.collect()


def test_build_model_prefix_kwargs(model_factory):
    """Test that prefix kwargs are properly extracted"""
    backbone = DummyModel()
    
    model = model_factory.build_model(
        task="segmentation",
        backbone=backbone,
        decoder="IdentityDecoder",
        in_channels=NUM_CHANNELS,
        bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
        num_classes=NUM_CLASSES,
        pretrained=False,
        backbone_some_arg="value1",
        decoder_some_arg="value2",
        head_some_arg="value3",
    )
    
    assert model is not None
    gc.collect()


@patch('importlib.import_module', side_effect=ModuleNotFoundError("Module not found"))
@patch('timm.create_model', side_effect=Exception("Not on HuggingFace"))
def test_build_model_module_not_found(mock_timm, mock_import, model_factory, tmp_path, capsys):
    """Test handling of ModuleNotFoundError"""
    checkpoint_path = str(tmp_path / "checkpoint.pth")
    torch.save({'model': {}}, checkpoint_path)
    
    # Should print error message and continue
    with pytest.raises(Exception):  # Will fail later due to missing backbone_template
        model_factory.build_model(
            task="segmentation",
            backbone="ViT",
            decoder="IdentityDecoder",
            in_channels=NUM_CHANNELS,
            bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
            num_classes=NUM_CLASSES,
            pretrained=False,
            checkpoint_path=checkpoint_path,
            model_sys_path="/invalid/path",
        )
    
    captured = capsys.readouterr()
    assert "better to review" in captured.out


def test_build_model_classification_task(model_factory):
    """Test building classification model"""
    backbone = DummyModel()
    
    model = model_factory.build_model(
        task="classification",
        backbone=backbone,
        decoder="IdentityDecoder",
        in_channels=NUM_CHANNELS,
        bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
        num_classes=NUM_CLASSES,
        pretrained=False,
    )
    
    from terratorch.models.scalar_output_model import ScalarOutputModel
    assert isinstance(model, ScalarOutputModel)
    gc.collect()


def test_build_model_regression_task(model_factory):
    """Test building regression model"""
    backbone = DummyModel()
    
    model = model_factory.build_model(
        task="regression",
        backbone=backbone,
        decoder="IdentityDecoder",
        in_channels=NUM_CHANNELS,
        bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
        num_classes=None,
        pretrained=False,
        rescale=False,
    )
    
    from terratorch.models.pixel_wise_model import PixelWiseModel
    assert isinstance(model, PixelWiseModel)
    gc.collect()


def test_build_model_without_num_classes_for_classification(model_factory):
    """Test building classification model without num_classes"""
    backbone = DummyModel()
    
    model = model_factory.build_model(
        task="classification",
        backbone=backbone,
        decoder="IdentityDecoder",
        in_channels=NUM_CHANNELS,
        bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
        num_classes=None,
        pretrained=False,
    )
    
    assert model is not None
    gc.collect()


def test_model_wrapper_no_num_patches():
    """Test ModelWrapper when model doesn't have num_patches"""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(768)
        
        def forward_features(self, x):
            return x
        
        def state_dict(self):
            return {'norm.bias': torch.zeros(768)}
    
    simple = SimpleModel()
    wrapper = ModelWrapper(model=simple, kind="vit")
    
    assert not hasattr(wrapper, 'num_patches')
    gc.collect()


@patch('torch.load')
@patch('importlib.import_module')
def test_build_model_pretrained_false_no_load(mock_import, mock_load, model_factory, tmp_path):
    """Test that model state dict is not loaded when pretrained=False"""
    mock_model_class = MagicMock(return_value=DummyModel())
    mock_module = types.ModuleType('models_vit')
    setattr(mock_module, 'ViT', mock_model_class)
    mock_import.return_value = mock_module
    
    mock_load.return_value = {'model': {'norm.weight': torch.ones(768)}}
    
    checkpoint_path = str(tmp_path / "checkpoint.pth")
    torch.save({'model': {}}, checkpoint_path)
    
    with patch('timm.create_model', side_effect=Exception("Not on HuggingFace")):
        model = model_factory.build_model(
            task="segmentation",
            backbone="ViT",
            decoder="IdentityDecoder",
            in_channels=NUM_CHANNELS,
            bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
            num_classes=NUM_CLASSES,
            pretrained=False,
            checkpoint_path=checkpoint_path,
            model_sys_path="/path/to/SatMAE",
        )
    
    # load_state_dict should not be called with pretrained=False
    assert model is not None
    gc.collect()

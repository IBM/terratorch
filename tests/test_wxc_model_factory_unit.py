import types
import sys
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, mock_open

from terratorch.models.wxc_model_factory import WxCModuleWrapper, WxCModelFactory


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(10, 10)
        self.head = nn.Linear(10, 10)
    
    def forward(self, x):
        return x * 2


def test_wxc_module_wrapper_init():
    dummy = DummyModule()
    wrapper = WxCModuleWrapper(dummy)
    assert wrapper.module is dummy


def test_wxc_module_wrapper_freeze_encoder():
    dummy = DummyModule()
    wrapper = WxCModuleWrapper(dummy)
    wrapper.freeze_encoder()
    for param in dummy.backbone.parameters():
        assert not param.requires_grad


def test_wxc_module_wrapper_freeze_decoder():
    dummy = DummyModule()
    wrapper = WxCModuleWrapper(dummy)
    wrapper.freeze_decoder()
    for param in dummy.head.parameters():
        assert not param.requires_grad


def test_wxc_module_wrapper_forward():
    dummy = DummyModule()
    wrapper = WxCModuleWrapper(dummy)
    x = torch.randn(2, 10)
    output = wrapper(x)
    assert output.output.shape == x.shape
    assert torch.allclose(output.output, x * 2)


def test_wxc_module_wrapper_load_state_dict():
    dummy = DummyModule()
    wrapper = WxCModuleWrapper(dummy)
    state_dict = {'backbone.weight': torch.randn(10, 10), 'backbone.bias': torch.randn(10)}
    # Just ensure it calls through without error
    wrapper.load_state_dict(state_dict, strict=False)


def test_wxc_model_factory_prithviwxc_module_not_found(monkeypatch):
    factory = WxCModelFactory()
    # Mock importlib.import_module to raise ModuleNotFoundError for PrithviWxC
    import importlib
    original_import = importlib.import_module
    def mock_import(name, *args, **kwargs):
        if 'PrithviWxC' in name:
            raise ModuleNotFoundError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(importlib, 'import_module', mock_import)
    
    with pytest.raises(ModuleNotFoundError):
        factory.build_model(backbone='prithviwxc', aux_decoders='unetpincer')


def test_wxc_model_factory_prithviwxc_unetpincer_path(monkeypatch):
    # Mock PrithviWxC module
    mock_prithvi = types.ModuleType('PrithviWxC')
    mock_model_module = types.ModuleType('model')
    
    class MockPrithviWxC(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)
        def forward(self, x):
            return x
    
    mock_model_module.PrithviWxC = MockPrithviWxC
    mock_prithvi.model = mock_model_module
    sys.modules['PrithviWxC'] = mock_prithvi
    sys.modules['PrithviWxC.model'] = mock_model_module
    
    # Mock UNetPincer
    class MockUNetPincer(nn.Module):
        def __init__(self, backbone, skip_connection=None):
            super().__init__()
            self.backbone = backbone
    
    monkeypatch.setattr('terratorch.models.wxc_model_factory.UNetPincer', MockUNetPincer)
    
    factory = WxCModelFactory()
    model = factory.build_model(
        backbone='prithviwxc',
        aux_decoders='unetpincer',
        skip_connection=True,
        in_chans=3
    )
    assert isinstance(model, MockUNetPincer)
    
    # Cleanup
    del sys.modules['PrithviWxC']
    del sys.modules['PrithviWxC.model']


def test_wxc_model_factory_prithviwxc_with_backbone_weights(monkeypatch, tmp_path):
    # Mock PrithviWxC module
    mock_prithvi = types.ModuleType('PrithviWxC')
    mock_model_module = types.ModuleType('model')
    
    class MockPrithviWxC(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)
        def forward(self, x):
            return x
    
    mock_model_module.PrithviWxC = MockPrithviWxC
    mock_prithvi.model = mock_model_module
    sys.modules['PrithviWxC'] = mock_prithvi
    sys.modules['PrithviWxC.model'] = mock_model_module
    
    # Mock UNetPincer
    class MockUNetPincer(nn.Module):
        def __init__(self, backbone, skip_connection=None):
            super().__init__()
            self.backbone = backbone
    
    monkeypatch.setattr('terratorch.models.wxc_model_factory.UNetPincer', MockUNetPincer)
    
    # Create fake checkpoint
    checkpoint_path = tmp_path / "backbone.pth"
    state_dict = {'conv.weight': torch.randn(3, 3, 1, 1), 'conv.bias': torch.randn(3)}
    torch.save(state_dict, checkpoint_path)
    
    factory = WxCModelFactory()
    model = factory.build_model(
        backbone='prithviwxc',
        aux_decoders='unetpincer',
        backbone_weights=str(checkpoint_path),
        skip_connection=False,
        in_chans=3
    )
    assert isinstance(model, MockUNetPincer)
    
    # Cleanup
    del sys.modules['PrithviWxC']
    del sys.modules['PrithviWxC.model']


def test_wxc_model_factory_prithviwxc_downscaler_path(monkeypatch, tmp_path):
    # Mock PrithviWxC module
    mock_prithvi = types.ModuleType('PrithviWxC')
    mock_model_module = types.ModuleType('model')
    
    class MockPrithviWxC(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
        def forward(self, x):
            return x
    
    mock_model_module.PrithviWxC = MockPrithviWxC
    mock_prithvi.model = mock_model_module
    sys.modules['PrithviWxC'] = mock_prithvi
    sys.modules['PrithviWxC.model'] = mock_model_module
    
    # Mock granitewxc modules
    mock_granitewxc = types.ModuleType('granitewxc')
    mock_utils = types.ModuleType('utils')
    mock_config_module = types.ModuleType('config')
    mock_downscaling = types.ModuleType('downscaling_model')
    
    class MockExperimentConfig:
        def __init__(self):
            self.data = types.SimpleNamespace(
                data_path_surface='',
                data_path_vertical='',
                climatology_path_surface='',
                climatology_path_vertical=''
            )
            self.model = types.SimpleNamespace(
                input_scalers_surface_path='',
                input_scalers_vertical_path='',
                output_scalers_surface_path='',
                output_scalers_vertical_path=''
            )
        @classmethod
        def from_dict(cls, d):
            return cls()
    
    def mock_get_config(path):
        return MockExperimentConfig()
    
    def mock_get_backbone(config):
        return MockPrithviWxC()
    
    mock_config_module.ExperimentConfig = MockExperimentConfig
    mock_config_module.get_config = mock_get_config
    mock_downscaling.get_backbone = mock_get_backbone
    mock_utils.config = mock_config_module
    mock_utils.downscaling_model = mock_downscaling
    mock_granitewxc.utils = mock_utils
    
    sys.modules['granitewxc'] = mock_granitewxc
    sys.modules['granitewxc.utils'] = mock_utils
    sys.modules['granitewxc.utils.config'] = mock_config_module
    sys.modules['granitewxc.utils.downscaling_model'] = mock_downscaling
    
    # Mock yaml
    mock_yaml = types.ModuleType('yaml')
    mock_yaml.safe_load = lambda f: {}
    sys.modules['yaml'] = mock_yaml
    
    # Mock get_downscaling_pincer
    class MockDownscalingPincer(nn.Module):
        def __init__(self, config, backbone):
            super().__init__()
            self.backbone = backbone
    
    monkeypatch.setattr('terratorch.models.wxc_model_factory.get_downscaling_pincer', 
                        lambda c, b: MockDownscalingPincer(c, b))
    
    # Create config file
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: {}\ndata: {}")
    
    # Create auxiliary data path
    aux_path = tmp_path / "aux_data"
    aux_path.mkdir()
    (aux_path / "merra-2").mkdir()
    (aux_path / "climatology").mkdir()
    
    factory = WxCModelFactory()
    model = factory.build_model(
        backbone='prithviwxc',
        aux_decoders='downscaler',
        config_path=str(config_path),
        wxc_auxiliary_data_path=str(aux_path),
        in_chans=3
    )
    assert isinstance(model, MockDownscalingPincer)
    
    # Cleanup
    for mod in ['PrithviWxC', 'PrithviWxC.model', 'granitewxc', 'granitewxc.utils',
                'granitewxc.utils.config', 'granitewxc.utils.downscaling_model', 'yaml']:
        if mod in sys.modules:
            del sys.modules[mod]


def test_wxc_model_factory_prithviwxc_downscaler_with_checkpoint(monkeypatch, tmp_path):
    # Similar setup to previous test
    mock_prithvi = types.ModuleType('PrithviWxC')
    mock_model_module = types.ModuleType('model')
    
    class MockPrithviWxC(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)
        def forward(self, x):
            return x
    
    mock_model_module.PrithviWxC = MockPrithviWxC
    mock_prithvi.model = mock_model_module
    sys.modules['PrithviWxC'] = mock_prithvi
    sys.modules['PrithviWxC.model'] = mock_model_module
    
    # Mock granitewxc modules
    mock_granitewxc = types.ModuleType('granitewxc')
    mock_utils = types.ModuleType('utils')
    mock_config_module = types.ModuleType('config')
    mock_downscaling = types.ModuleType('downscaling_model')
    
    class MockExperimentConfig:
        def __init__(self):
            self.data = types.SimpleNamespace(
                data_path_surface='',
                data_path_vertical='',
                climatology_path_surface='',
                climatology_path_vertical=''
            )
            self.model = types.SimpleNamespace(
                input_scalers_surface_path='',
                input_scalers_vertical_path='',
                output_scalers_surface_path='',
                output_scalers_vertical_path=''
            )
        @classmethod
        def from_dict(cls, d):
            return cls()
    
    mock_config_module.ExperimentConfig = MockExperimentConfig
    mock_config_module.get_config = lambda p: MockExperimentConfig()
    mock_downscaling.get_backbone = lambda c: MockPrithviWxC()
    mock_utils.config = mock_config_module
    mock_utils.downscaling_model = mock_downscaling
    mock_granitewxc.utils = mock_utils
    
    sys.modules['granitewxc'] = mock_granitewxc
    sys.modules['granitewxc.utils'] = mock_utils
    sys.modules['granitewxc.utils.config'] = mock_config_module
    sys.modules['granitewxc.utils.downscaling_model'] = mock_downscaling
    
    mock_yaml = types.ModuleType('yaml')
    mock_yaml.safe_load = lambda f: {}
    sys.modules['yaml'] = mock_yaml
    
    class MockDownscalingPincer(nn.Module):
        def __init__(self, config, backbone):
            super().__init__()
            self.backbone = backbone
            self.conv = nn.Conv2d(3, 3, 1)
    
    monkeypatch.setattr('terratorch.models.wxc_model_factory.get_downscaling_pincer',
                        lambda c, b: MockDownscalingPincer(c, b))
    
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: {}\ndata: {}")
    
    aux_path = tmp_path / "aux_data"
    aux_path.mkdir()
    (aux_path / "merra-2").mkdir()
    (aux_path / "climatology").mkdir()
    
    # Create checkpoint
    checkpoint_path = tmp_path / "checkpoint.pth"
    state_dict = {'conv.weight': torch.randn(3, 3, 1, 1), 'conv.bias': torch.randn(3)}
    torch.save(state_dict, checkpoint_path)
    
    factory = WxCModelFactory()
    model = factory.build_model(
        backbone='prithviwxc',
        aux_decoders='downscaler',
        config_path=str(config_path),
        wxc_auxiliary_data_path=str(aux_path),
        checkpoint_path=str(checkpoint_path),
        in_chans=3
    )
    assert isinstance(model, MockDownscalingPincer)
    
    # Cleanup
    for mod in ['PrithviWxC', 'PrithviWxC.model', 'granitewxc', 'granitewxc.utils',
                'granitewxc.utils.config', 'granitewxc.utils.downscaling_model', 'yaml']:
        if mod in sys.modules:
            del sys.modules[mod]


def test_wxc_model_factory_prithvi_eccc_downscaling_not_installed(monkeypatch):
    factory = WxCModelFactory()
    # Mock __import__ to raise ImportError for granitewxc.models.model imports
    orig_import = __import__
    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'granitewxc.models.model' or name == 'granitewxc.utils.config' or name == 'granitewxc.utils.downscaling_model':
            raise ImportError(f"No module named '{name}'")
        return orig_import(name, globals, locals, fromlist, level)
    
    monkeypatch.setattr('builtins.__import__', mock_import)
    
    # When ImportError is caught for prithvi-eccc, it prints and falls through
    # to the deprecated else block which also catches ImportError and returns None implicitly
    result = factory.build_model(
        backbone='prithvi-eccc-downscaling',
        aux_decoders='any',
        model_args=types.SimpleNamespace(model=types.SimpleNamespace(unet=False))
    )
    assert result is None


def test_wxc_model_factory_gravitywave_not_installed():
    factory = WxCModelFactory()
    result = factory.build_model(backbone='gravitywave', aux_decoders='any')
    assert result is None


def test_wxc_model_factory_downscaling_fallback_not_installed(monkeypatch):
    factory = WxCModelFactory()
    # Mock __import__ to raise ImportError for granitewxc.utils modules
    orig_import = __import__
    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if 'granitewxc.utils' in name:
            raise ImportError(f"No module named '{name}'")
        return orig_import(name, globals, locals, fromlist, level)
    
    monkeypatch.setattr('builtins.__import__', mock_import)
    
    # The else block catches ImportError and returns None implicitly
    result = factory.build_model(
        backbone='unknown_backbone',
        aux_decoders='any',
        model_config={}
    )
    assert result is None
def test_wxc_model_factory_prithviwxc_default_return_wrapper(monkeypatch):
    # Test the final return WxCModuleWrapper(backbone) path
    mock_prithvi = types.ModuleType('PrithviWxC')
    mock_model_module = types.ModuleType('model')
    
    class MockPrithviWxC(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
        def forward(self, x):
            return x
    
    mock_model_module.PrithviWxC = MockPrithviWxC
    mock_prithvi.model = mock_model_module
    sys.modules['PrithviWxC'] = mock_prithvi
    sys.modules['PrithviWxC.model'] = mock_model_module
    
    factory = WxCModelFactory()
    model = factory.build_model(
        backbone='prithviwxc',
        aux_decoders='unknown_decoder',  # Not unetpincer or downscaler
        in_chans=3
    )
    assert isinstance(model, WxCModuleWrapper)
    
    # Cleanup
    del sys.modules['PrithviWxC']
    del sys.modules['PrithviWxC.model']

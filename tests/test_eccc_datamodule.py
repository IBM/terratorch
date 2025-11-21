"""Comprehensive tests for ECCCDataModule.

Tests cover initialization, setup, dataloader methods with various configurations
to ensure maximum code coverage.
"""

import gc
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from torch.utils.data import DataLoader

# Mock granitewxc before importing ECCCDataModule
mock_granitewxc = MagicMock()
sys.modules['granitewxc'] = mock_granitewxc
sys.modules['granitewxc.utils'] = MagicMock()
sys.modules['granitewxc.utils.config'] = MagicMock()
sys.modules['granitewxc.datasets'] = MagicMock()
sys.modules['granitewxc.datasets.eccc'] = MagicMock()

from terratorch.datamodules.eccc import ECCCDataModule


@pytest.fixture
def mock_config():
    """Create a mock ExperimentConfig object."""
    config = MagicMock()
    
    # Basic config attributes
    config.batch_size = 4
    config.dl_num_workers = 2
    config.dl_prefetch_size = 2
    
    # Data config
    config.data = MagicMock()
    config.data.static_data_index = "/path/to/static_data.json"
    config.data.input_surface_vars = ["temperature", "pressure"]
    config.data.vertical_pres_vars = ["geopotential"]
    config.data.vertical_level1_vars = ["wind_u"]
    config.data.vertical_level2_vars = ["wind_v"]
    config.data.other = ["humidity"]
    config.data.input_static_surface_vars = ["elevation"]
    config.data.output_vars = ["precipitation"]
    config.data.downsample_factor = 2
    config.data.n_random_windows = 1
    config.data.crop_factor = 0.5
    config.data.data_training_index = "/path/to/train.json"
    config.data.data_test_index = "/path/to/test.json"
    config.data.data_val_index = "/path/to/val.json"
    
    return config


@pytest.fixture
def mock_config_no_workers():
    """Create a mock config with no workers."""
    config = MagicMock()
    
    config.batch_size = 2
    config.dl_num_workers = 0
    config.dl_prefetch_size = 2
    
    config.data = MagicMock()
    config.data.static_data_index = "/path/to/static_data.json"
    config.data.input_surface_vars = ["temperature"]
    config.data.vertical_pres_vars = []
    config.data.vertical_level1_vars = []
    config.data.vertical_level2_vars = []
    config.data.other = []
    config.data.input_static_surface_vars = []
    config.data.output_vars = ["output"]
    config.data.downsample_factor = 1
    config.data.n_random_windows = 1
    config.data.crop_factor = 1.0
    config.data.data_training_index = "/path/to/train.json"
    config.data.data_test_index = "/path/to/test.json"
    config.data.data_val_index = "/path/to/val.json"
    
    return config


@pytest.fixture
def mock_dataset():
    """Create a mock dataset."""
    dataset = MagicMock()
    dataset.__len__ = Mock(return_value=10)
    dataset.__getitem__ = Mock(return_value={
        "input": torch.randn(3, 32, 32),
        "target": torch.randn(1, 32, 32),
    })
    return dataset


class TestECCCDataModuleInitialization:
    """Test suite for ECCCDataModule initialization."""

    def test_basic_initialization(self, mock_config):
        """Test basic initialization with config."""
        datamodule = ECCCDataModule(config=mock_config)
        
        assert datamodule.config is mock_config
        assert datamodule.dl_kwargs is not None
        assert datamodule.ds_kwargs is not None

    def test_initialization_dl_kwargs(self, mock_config):
        """Test that dl_kwargs are correctly set."""
        datamodule = ECCCDataModule(config=mock_config)
        
        assert datamodule.dl_kwargs["batch_size"] == 4
        assert datamodule.dl_kwargs["num_workers"] == 2
        assert datamodule.dl_kwargs["prefetch_factor"] == 2
        assert datamodule.dl_kwargs["pin_memory"] is True
        assert datamodule.dl_kwargs["drop_last"] is True
        assert datamodule.dl_kwargs["persistent_workers"] is True

    def test_initialization_dl_kwargs_no_workers(self, mock_config_no_workers):
        """Test dl_kwargs with no workers."""
        datamodule = ECCCDataModule(config=mock_config_no_workers)
        
        assert datamodule.dl_kwargs["num_workers"] == 0
        assert datamodule.dl_kwargs["persistent_workers"] is False

    def test_initialization_ds_kwargs(self, mock_config):
        """Test that ds_kwargs are correctly set."""
        datamodule = ECCCDataModule(config=mock_config)
        
        assert datamodule.ds_kwargs["json_static_var_path"] == "/path/to/static_data.json"
        assert datamodule.ds_kwargs["surface_vars"] == ["temperature", "pressure"]
        assert datamodule.ds_kwargs["vertical_pres_vars"] == ["geopotential"]
        assert datamodule.ds_kwargs["vertical_level1_vars"] == ["wind_u"]
        assert datamodule.ds_kwargs["vertical_level2_vars"] == ["wind_v"]
        assert datamodule.ds_kwargs["other_vars"] == ["humidity"]
        assert datamodule.ds_kwargs["static_vars"] == ["elevation"]
        assert datamodule.ds_kwargs["output_vars"] == ["precipitation"]
        assert datamodule.ds_kwargs["downsample_factor"] == 2
        assert datamodule.ds_kwargs["n_random_windows"] == 1
        assert datamodule.ds_kwargs["crop_factor"] == 0.5

    def test_initialization_with_different_batch_sizes(self):
        """Test initialization with various batch sizes."""
        batch_sizes = [1, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            config = MagicMock()
            config.batch_size = batch_size
            config.dl_num_workers = 2
            config.dl_prefetch_size = 2
            config.data = MagicMock()
            config.data.static_data_index = "/path/to/static.json"
            config.data.input_surface_vars = []
            config.data.vertical_pres_vars = []
            config.data.vertical_level1_vars = []
            config.data.vertical_level2_vars = []
            config.data.other = []
            config.data.input_static_surface_vars = []
            config.data.output_vars = []
            config.data.downsample_factor = 1
            config.data.n_random_windows = 1
            config.data.crop_factor = 1.0
            
            datamodule = ECCCDataModule(config=config)
            assert datamodule.dl_kwargs["batch_size"] == batch_size

    def test_initialization_with_different_workers(self):
        """Test initialization with various worker counts."""
        worker_counts = [0, 1, 2, 4, 8]
        
        for num_workers in worker_counts:
            config = MagicMock()
            config.batch_size = 4
            config.dl_num_workers = num_workers
            config.dl_prefetch_size = 2
            config.data = MagicMock()
            config.data.static_data_index = "/path/to/static.json"
            config.data.input_surface_vars = []
            config.data.vertical_pres_vars = []
            config.data.vertical_level1_vars = []
            config.data.vertical_level2_vars = []
            config.data.other = []
            config.data.input_static_surface_vars = []
            config.data.output_vars = []
            config.data.downsample_factor = 1
            config.data.n_random_windows = 1
            config.data.crop_factor = 1.0
            
            datamodule = ECCCDataModule(config=config)
            assert datamodule.dl_kwargs["num_workers"] == num_workers
            assert datamodule.dl_kwargs["persistent_workers"] == (num_workers > 0)


class TestECCCDataModulePrepareData:
    """Test suite for prepare_data method."""

    def test_prepare_data_executes(self, mock_config):
        """Test that prepare_data executes without error."""
        datamodule = ECCCDataModule(config=mock_config)
        
        # Should not raise any error
        datamodule.prepare_data()

    def test_prepare_data_is_noop(self, mock_config):
        """Test that prepare_data doesn't do anything."""
        datamodule = ECCCDataModule(config=mock_config)
        
        # Call multiple times - should be safe
        datamodule.prepare_data()
        datamodule.prepare_data()
        datamodule.prepare_data()


class TestECCCDataModuleSetup:
    """Test suite for setup method."""

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_setup_fit_stage(self, mock_dataset_class, mock_config):
        """Test setup with 'fit' stage."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage="fit")
        
        assert hasattr(datamodule, "train_dataset")
        mock_dataset_class.assert_called_once()

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_setup_test_stage(self, mock_dataset_class, mock_config):
        """Test setup with 'test' stage."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage="test")
        
        assert hasattr(datamodule, "test_dataset")
        mock_dataset_class.assert_called_once()
        # Check that test=True was passed
        call_kwargs = mock_dataset_class.call_args[1]
        assert call_kwargs.get("test") is True

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_setup_val_stage(self, mock_dataset_class, mock_config):
        """Test setup with 'val' stage."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage="val")
        
        assert hasattr(datamodule, "valid_dataset")
        mock_dataset_class.assert_called_once()

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_setup_predict_stage(self, mock_dataset_class, mock_config):
        """Test setup with 'predict' stage."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage="predict")
        
        assert hasattr(datamodule, "predict_dataset")
        mock_dataset_class.assert_called_once()
        # Check that test=True was passed
        call_kwargs = mock_dataset_class.call_args[1]
        assert call_kwargs.get("test") is True

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_setup_none_stage(self, mock_dataset_class, mock_config):
        """Test setup with None stage."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage=None)
        
        # Should not create any datasets when stage is None
        mock_dataset_class.assert_not_called()

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_setup_fit_uses_correct_path(self, mock_dataset_class, mock_config):
        """Test that fit stage uses training data path."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage="fit")
        
        call_kwargs = mock_dataset_class.call_args[1]
        assert call_kwargs["json_file_path"] == "/path/to/train.json"

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_setup_test_uses_correct_path(self, mock_dataset_class, mock_config):
        """Test that test stage uses test data path."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage="test")
        
        call_kwargs = mock_dataset_class.call_args[1]
        assert call_kwargs["json_file_path"] == "/path/to/test.json"

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_setup_val_uses_correct_path(self, mock_dataset_class, mock_config):
        """Test that val stage uses validation data path."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage="val")
        
        call_kwargs = mock_dataset_class.call_args[1]
        assert call_kwargs["json_file_path"] == "/path/to/val.json"

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_setup_passes_ds_kwargs(self, mock_dataset_class, mock_config):
        """Test that setup passes ds_kwargs to dataset."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage="fit")
        
        call_kwargs = mock_dataset_class.call_args[1]
        assert call_kwargs["surface_vars"] == ["temperature", "pressure"]
        assert call_kwargs["vertical_pres_vars"] == ["geopotential"]
        assert call_kwargs["downsample_factor"] == 2


class TestECCCDataModuleDataLoaders:
    """Test suite for dataloader methods."""

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_train_dataloader(self, mock_dataset_class, mock_config):
        """Test train_dataloader method."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage="fit")
        
        train_loader = datamodule.train_dataloader()
        
        assert isinstance(train_loader, DataLoader)
        assert train_loader.batch_size == 4
        assert train_loader.num_workers == 2

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_val_dataloader(self, mock_dataset_class, mock_config):
        """Test val_dataloader method."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage="val")
        
        val_loader = datamodule.val_dataloader()
        
        assert isinstance(val_loader, DataLoader)
        assert val_loader.batch_size == 4

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_test_dataloader(self, mock_dataset_class, mock_config):
        """Test test_dataloader method."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage="test")
        
        test_loader = datamodule.test_dataloader()
        
        assert isinstance(test_loader, DataLoader)
        assert test_loader.batch_size == 4

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_predict_dataloader(self, mock_dataset_class, mock_config):
        """Test predict_dataloader method."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage="predict")
        
        predict_loader = datamodule.predict_dataloader()
        
        assert isinstance(predict_loader, DataLoader)
        assert predict_loader.batch_size == 4

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_dataloader_attributes(self, mock_dataset_class, mock_config):
        """Test that dataloaders have correct attributes."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage="fit")
        
        train_loader = datamodule.train_dataloader()
        
        assert train_loader.pin_memory is True
        assert train_loader.drop_last is True


class TestECCCDataModuleEdgeCases:
    """Test edge cases and special scenarios."""

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_multiple_setup_calls(self, mock_dataset_class, mock_config):
        """Test calling setup multiple times."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        
        # Setup for fit
        datamodule.setup(stage="fit")
        assert hasattr(datamodule, "train_dataset")
        
        # Setup for val
        datamodule.setup(stage="val")
        assert hasattr(datamodule, "valid_dataset")
        
        # Setup for test
        datamodule.setup(stage="test")
        assert hasattr(datamodule, "test_dataset")

    def test_config_attribute_access(self, mock_config):
        """Test that config attributes are accessible."""
        datamodule = ECCCDataModule(config=mock_config)
        
        assert datamodule.config.batch_size == 4
        assert datamodule.config.dl_num_workers == 2
        assert datamodule.config.data.downsample_factor == 2

    def test_different_downsample_factors(self):
        """Test with different downsample factors."""
        downsample_factors = [1, 2, 4, 8]
        
        for factor in downsample_factors:
            config = MagicMock()
            config.batch_size = 4
            config.dl_num_workers = 0
            config.dl_prefetch_size = 2
            config.data = MagicMock()
            config.data.static_data_index = "/path/to/static.json"
            config.data.input_surface_vars = []
            config.data.vertical_pres_vars = []
            config.data.vertical_level1_vars = []
            config.data.vertical_level2_vars = []
            config.data.other = []
            config.data.input_static_surface_vars = []
            config.data.output_vars = []
            config.data.downsample_factor = factor
            config.data.n_random_windows = 1
            config.data.crop_factor = 1.0
            
            datamodule = ECCCDataModule(config=config)
            assert datamodule.ds_kwargs["downsample_factor"] == factor

    def test_empty_variable_lists(self):
        """Test with empty variable lists."""
        config = MagicMock()
        config.batch_size = 4
        config.dl_num_workers = 0
        config.dl_prefetch_size = 2
        config.data = MagicMock()
        config.data.static_data_index = "/path/to/static.json"
        config.data.input_surface_vars = []
        config.data.vertical_pres_vars = []
        config.data.vertical_level1_vars = []
        config.data.vertical_level2_vars = []
        config.data.other = []
        config.data.input_static_surface_vars = []
        config.data.output_vars = []
        config.data.downsample_factor = 1
        config.data.n_random_windows = 1
        config.data.crop_factor = 1.0
        
        datamodule = ECCCDataModule(config=config)
        
        assert datamodule.ds_kwargs["surface_vars"] == []
        assert datamodule.ds_kwargs["vertical_pres_vars"] == []
        assert datamodule.ds_kwargs["output_vars"] == []


class TestECCCDataModuleIntegration:
    """Integration tests for ECCCDataModule."""

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_full_workflow_fit(self, mock_dataset_class, mock_config):
        """Test complete workflow for fit stage."""
        mock_dataset_class.return_value = MagicMock()
        
        # Create datamodule
        datamodule = ECCCDataModule(config=mock_config)
        
        # Prepare data
        datamodule.prepare_data()
        
        # Setup
        datamodule.setup(stage="fit")
        
        # Get dataloader
        train_loader = datamodule.train_dataloader()
        
        assert isinstance(train_loader, DataLoader)
        assert train_loader.batch_size == 4

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_full_workflow_all_stages(self, mock_dataset_class, mock_config):
        """Test complete workflow for all stages."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        
        # Setup all stages
        datamodule.setup(stage="fit")
        datamodule.setup(stage="val")
        datamodule.setup(stage="test")
        datamodule.setup(stage="predict")
        
        # Get all dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        predict_loader = datamodule.predict_dataloader()
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        assert isinstance(predict_loader, DataLoader)

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_datamodule_reusability(self, mock_dataset_class, mock_config):
        """Test that datamodule can be reused across epochs."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage="fit")
        
        # Get dataloader multiple times
        loader1 = datamodule.train_dataloader()
        loader2 = datamodule.train_dataloader()
        
        # Both should be valid DataLoaders
        assert isinstance(loader1, DataLoader)
        assert isinstance(loader2, DataLoader)

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_memory_cleanup(self, mock_dataset_class, mock_config):
        """Test that datamodule can be properly cleaned up."""
        mock_dataset_class.return_value = MagicMock()
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage="fit")
        _ = datamodule.train_dataloader()
        
        # Delete and cleanup
        del datamodule
        gc.collect()

    @patch('terratorch.datamodules.eccc.EcccHrdpsGdpsDataset')
    def test_config_immutability(self, mock_dataset_class, mock_config):
        """Test that original config is not modified."""
        original_batch_size = mock_config.batch_size
        
        datamodule = ECCCDataModule(config=mock_config)
        datamodule.setup(stage="fit")
        
        # Config should remain unchanged
        assert mock_config.batch_size == original_batch_size


class TestECCCDataModuleLightningCompatibility:
    """Test Lightning-specific functionality."""

    def test_inheritance_from_lightning(self, mock_config):
        """Test that ECCCDataModule inherits from LightningDataModule."""
        import lightning as pl
        
        datamodule = ECCCDataModule(config=mock_config)
        
        assert isinstance(datamodule, pl.LightningDataModule)

    def test_has_required_methods(self, mock_config):
        """Test that datamodule has all required methods."""
        datamodule = ECCCDataModule(config=mock_config)
        
        assert hasattr(datamodule, "prepare_data")
        assert hasattr(datamodule, "setup")
        assert hasattr(datamodule, "train_dataloader")
        assert hasattr(datamodule, "val_dataloader")
        assert hasattr(datamodule, "test_dataloader")
        assert hasattr(datamodule, "predict_dataloader")
        
        assert callable(datamodule.prepare_data)
        assert callable(datamodule.setup)
        assert callable(datamodule.train_dataloader)
        assert callable(datamodule.val_dataloader)
        assert callable(datamodule.test_dataloader)
        assert callable(datamodule.predict_dataloader)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

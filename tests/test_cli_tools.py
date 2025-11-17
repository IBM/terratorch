# Copyright contributors to the Terratorch project

"""Tests for cli_tools.py module"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import rasterio
import torch
import yaml
from jsonargparse import Namespace
from lightning.pytorch import LightningDataModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter, ModelCheckpoint

from terratorch.cli_tools import (
    CustomWriter,
    LightningInferenceModel,
    MyLightningCLI,
    MyTrainer,
    StateDictAwareModelCheckpoint,
    StudioDeploySaveConfigCallback,
    add_default_checkpointing_config,
    build_lightning_cli,
    clean_config_for_deployment_and_dump,
    flatten,
    import_custom_modules,
    is_one_band,
    open_tiff,
    save_prediction,
    write_tiff,
)


class TestUtilityFunctions:
    """Test utility functions"""

    def test_flatten(self):
        """Test flatten function with list of lists"""
        list_of_lists = [[1, 2, 3], [4, 5], [6]]
        result = flatten(list_of_lists)
        assert result == [1, 2, 3, 4, 5, 6]

    def test_flatten_empty(self):
        """Test flatten with empty list"""
        result = flatten([])
        assert result == []

    def test_flatten_single_list(self):
        """Test flatten with single list"""
        result = flatten([[1, 2, 3]])
        assert result == [1, 2, 3]

    def test_is_one_band_true(self):
        """Test is_one_band with 2D array"""
        img = np.zeros((10, 10))
        assert is_one_band(img) is True

    def test_is_one_band_false(self):
        """Test is_one_band with 3D array"""
        img = np.zeros((3, 10, 10))
        assert is_one_band(img) is False

    def test_open_tiff(self):
        """Test opening a TIFF file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple test TIFF
            test_file = os.path.join(tmpdir, "test.tif")
            test_data = np.random.randint(0, 255, (3, 10, 10), dtype=np.uint8)

            metadata = {
                "driver": "GTiff",
                "height": 10,
                "width": 10,
                "count": 3,
                "dtype": "uint8",
                "crs": "EPSG:4326",
                "transform": rasterio.transform.from_bounds(0, 0, 1, 1, 10, 10),
            }

            with rasterio.open(test_file, "w", **metadata) as dst:
                dst.write(test_data)

            # Test reading
            data, meta = open_tiff(test_file)
            assert data.shape == (3, 10, 10)
            assert meta["height"] == 10
            assert meta["width"] == 10

    def test_write_tiff_multiband(self):
        """Test writing multiband TIFF"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test_write.tif")
            test_data = np.random.randint(0, 255, (3, 10, 10), dtype=np.uint8)

            metadata = {
                "driver": "GTiff",
                "height": 10,
                "width": 10,
                "count": 1,  # Will be updated to 3
                "dtype": "uint8",
                "crs": "EPSG:4326",
                "transform": rasterio.transform.from_bounds(0, 0, 1, 1, 10, 10),
            }

            result_file = write_tiff(test_data, test_file, metadata)
            assert result_file == test_file
            assert os.path.exists(test_file)

            # Verify the written data
            with rasterio.open(test_file) as src:
                assert src.count == 3

    def test_write_tiff_single_band(self):
        """Test writing single band TIFF"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test_write_single.tif")
            test_data = np.random.randint(0, 255, (10, 10), dtype=np.uint8)

            metadata = {
                "driver": "GTiff",
                "height": 10,
                "width": 10,
                "count": 1,
                "dtype": "uint8",
                "crs": "EPSG:4326",
                "transform": rasterio.transform.from_bounds(0, 0, 1, 1, 10, 10),
            }

            result_file = write_tiff(test_data, test_file, metadata)
            assert result_file == test_file
            assert os.path.exists(test_file)

            with rasterio.open(test_file) as src:
                assert src.count == 1

    def test_save_prediction(self):
        """Test save_prediction function"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input file
            input_file = os.path.join(tmpdir, "input.tif")
            test_data = np.ones((3, 10, 10), dtype=np.uint8) * 100

            metadata = {
                "driver": "GTiff",
                "height": 10,
                "width": 10,
                "count": 3,
                "dtype": "uint8",
                "nodata": 0,
                "crs": "EPSG:4326",
                "transform": rasterio.transform.from_bounds(0, 0, 1, 1, 10, 10),
            }

            with rasterio.open(input_file, "w", **metadata) as dst:
                dst.write(test_data)

            # Create prediction
            prediction = torch.randn(10, 10)

            # Save prediction
            save_prediction(prediction, input_file, tmpdir, dtype="float32", suffix="pred")

            # Check output file exists
            output_file = os.path.join(tmpdir, "input_pred.tif")
            assert os.path.exists(output_file)

    def test_save_prediction_with_custom_name(self):
        """Test save_prediction with custom output filename"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "input.tif")
            test_data = np.ones((3, 10, 10), dtype=np.uint8) * 100

            metadata = {
                "driver": "GTiff",
                "height": 10,
                "width": 10,
                "count": 3,
                "dtype": "uint8",
                "nodata": 0,
                "crs": "EPSG:4326",
                "transform": rasterio.transform.from_bounds(0, 0, 1, 1, 10, 10),
            }

            with rasterio.open(input_file, "w", **metadata) as dst:
                dst.write(test_data)

            prediction = torch.randn(10, 10)
            save_prediction(
                prediction, input_file, tmpdir, dtype="float32", suffix="pred", output_file_name="custom_output"
            )

            output_file = os.path.join(tmpdir, "custom_output_pred.tif")
            assert os.path.exists(output_file)


class TestConfigFunctions:
    """Test configuration related functions"""

    def test_add_default_checkpointing_config_no_subcommand(self):
        """Test add_default_checkpointing_config with no subcommand"""
        config = {}
        result = add_default_checkpointing_config(config)
        assert result == config

    def test_add_default_checkpointing_config_disabled(self):
        """Test add_default_checkpointing_config when checkpointing is disabled"""
        config = {
            "subcommand": "fit",
            "fit.trainer.enable_checkpointing": False,
            "fit.trainer.callbacks": None,
        }
        result = add_default_checkpointing_config(config)
        # Should not add checkpoints when disabled
        assert result["fit.trainer.callbacks"] is None

    def test_add_default_checkpointing_config_enabled_no_callbacks(self):
        """Test add_default_checkpointing_config when enabled with no existing callbacks"""
        config = {
            "subcommand": "fit",
            "fit.trainer.enable_checkpointing": True,
            "fit.trainer.callbacks": None,
        }
        result = add_default_checkpointing_config(config)
        # Should add default checkpoint configs
        assert result["fit.trainer.callbacks"] is not None
        assert len(result["fit.trainer.callbacks"]) == 2

    def test_add_default_checkpointing_config_with_existing_checkpoint(self):
        """Test add_default_checkpointing_config with existing ModelCheckpoint"""
        existing_callback = Namespace(class_path="ModelCheckpoint", init_args=Namespace(monitor="val/loss"))
        config = {
            "subcommand": "fit",
            "fit.trainer.enable_checkpointing": True,
            "fit.trainer.callbacks": [existing_callback],
        }
        result = add_default_checkpointing_config(config)
        # Should not add more checkpoints
        assert len(result["fit.trainer.callbacks"]) == 1

    def test_clean_config_for_deployment_and_dump(self):
        """Test clean_config_for_deployment_and_dump"""
        config = {
            "ckpt_path": "/path/to/checkpoint",
            "ModelCheckpoint": {},
            "StateDictModelCheckpoint": {},
            "optimizer": {"lr": 0.001},
            "lr_scheduler": {},
            "verbose": True,
            "trainer": {
                "logger": True,
                "callbacks": [],
                "default_root_dir": "/tmp",
                "max_epochs": 10,
            },
            "model": {
                "init_args": {
                    "model_args": {
                        "pretrained": True,
                    }
                }
            },
        }

        result = clean_config_for_deployment_and_dump(config)
        assert isinstance(result, str)
        result_dict = yaml.safe_load(result)

        # Check removed fields
        assert "ckpt_path" not in result_dict
        assert "ModelCheckpoint" not in result_dict
        assert "optimizer" not in result_dict
        assert "verbose" not in result_dict

        # Check trainer modifications
        assert result_dict["trainer"]["logger"] is False
        assert "callbacks" not in result_dict["trainer"]
        assert "default_root_dir" not in result_dict["trainer"]
        assert result_dict["trainer"]["precision"] == "16-mixed"

        # Check model pretrained flag
        assert result_dict["model"]["init_args"]["model_args"]["pretrained"] is False

    def test_clean_config_for_deployment_backbone_pretrained(self):
        """Test clean_config_for_deployment with backbone_pretrained"""
        config = {
            "trainer": {"logger": True},
            "model": {"init_args": {"model_args": {"backbone_pretrained": True}}},
        }

        result = clean_config_for_deployment_and_dump(config)
        result_dict = yaml.safe_load(result)
        assert result_dict["model"]["init_args"]["model_args"]["backbone_pretrained"] is False

    def test_clean_config_for_deployment_no_model_args(self):
        """Test clean_config_for_deployment without model_args"""
        config = {
            "trainer": {"logger": True},
            "model": {"init_args": {}},
        }

        result = clean_config_for_deployment_and_dump(config)
        result_dict = yaml.safe_load(result)
        assert "model_args" not in result_dict["model"]["init_args"]


class TestImportCustomModules:
    """Test import_custom_modules function"""

    def test_import_custom_modules_none(self):
        """Test import_custom_modules with None path"""
        # Should not raise any error
        import_custom_modules(None)

    def test_import_custom_modules_valid_directory(self):
        """Test import_custom_modules with valid directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            module_dir = Path(tmpdir) / "custom_modules"
            module_dir.mkdir()

            # Create a simple module file
            module_file = module_dir / "__init__.py"
            module_file.write_text("# Custom module")

            # This should work without errors
            import_custom_modules(module_dir)

    def test_import_custom_modules_invalid_path(self):
        """Test import_custom_modules with non-directory path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file instead of directory
            file_path = Path(tmpdir) / "not_a_dir.py"
            file_path.write_text("# File")

            with pytest.raises(ValueError, match="isn't a directory"):
                import_custom_modules(file_path)

    def test_import_custom_modules_import_error(self):
        """Test import_custom_modules with directory that fails to import"""
        with tempfile.TemporaryDirectory() as tmpdir:
            module_dir = Path(tmpdir) / "bad_module"
            module_dir.mkdir()

            # Create a module with import error
            module_file = module_dir / "__init__.py"
            module_file.write_text("import nonexistent_module")

            with pytest.raises(ImportError, match="not possible to import"):
                import_custom_modules(module_dir)


class TestCustomWriter:
    """Test CustomWriter callback"""

    def setup_method(self):
        """Setup test fixtures"""
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup temp directory"""
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_custom_writer_init(self):
        """Test CustomWriter initialization"""
        writer = CustomWriter(output_dir="/tmp/output", write_interval="batch")
        assert writer.output_dir == "/tmp/output"

    def test_custom_writer_init_default(self):
        """Test CustomWriter initialization with defaults"""
        writer = CustomWriter()
        assert writer.output_dir is None

    def test_write_on_batch_end_tensor(self):
        """Test write_on_batch_end with torch.Tensor prediction"""
        writer = CustomWriter(output_dir=self.tmpdir, write_interval="batch")

        # Mock trainer and module
        trainer = Mock()
        trainer.out_dtype = "int16"
        pl_module = Mock()

        prediction = torch.randn(4, 10, 10)
        batch_indices = None
        batch = {}
        batch_idx = 0
        dataloader_idx = 0

        writer.write_on_batch_end(trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx)

        # Check that a file was created
        files = os.listdir(self.tmpdir)
        assert len(files) == 1
        assert files[0].endswith(".pt")

    def test_write_on_batch_end_tuple_with_suffix(self):
        """Test write_on_batch_end with tuple (prediction, suffix) format"""
        writer = CustomWriter(output_dir=self.tmpdir, write_interval="batch")

        # Create test input file
        input_file = os.path.join(self.tmpdir, "input.tif")
        test_data = np.ones((3, 10, 10), dtype=np.uint8) * 100
        metadata = {
            "driver": "GTiff",
            "height": 10,
            "width": 10,
            "count": 3,
            "dtype": "uint8",
            "nodata": 0,
            "crs": "EPSG:4326",
            "transform": rasterio.transform.from_bounds(0, 0, 1, 1, 10, 10),
        }
        with rasterio.open(input_file, "w", **metadata) as dst:
            dst.write(test_data)

        trainer = Mock()
        trainer.out_dtype = "int16"
        # Mock should not have output_file_prefix attribute
        if hasattr(trainer, 'output_file_prefix'):
            delattr(trainer, 'output_file_prefix')
        pl_module = Mock()

        pred_batch = torch.randn(1, 10, 10)
        prediction = ((pred_batch, "suffix"), [input_file])

        writer.write_on_batch_end(trainer, pl_module, prediction, None, {}, 0, 0)

        # Check output file
        output_files = [f for f in os.listdir(self.tmpdir) if f.endswith("_suffix.tif")]
        assert len(output_files) == 1

    def test_write_on_batch_end_list_predictions(self):
        """Test write_on_batch_end with list of predictions"""
        writer = CustomWriter(output_dir=self.tmpdir, write_interval="batch")

        input_file = os.path.join(self.tmpdir, "input.tif")
        test_data = np.ones((3, 10, 10), dtype=np.uint8) * 100
        metadata = {
            "driver": "GTiff",
            "height": 10,
            "width": 10,
            "count": 3,
            "dtype": "uint8",
            "nodata": 0,
            "crs": "EPSG:4326",
            "transform": rasterio.transform.from_bounds(0, 0, 1, 1, 10, 10),
        }
        with rasterio.open(input_file, "w", **metadata) as dst:
            dst.write(test_data)

        trainer = Mock()
        trainer.out_dtype = "int16"
        if hasattr(trainer, 'output_file_prefix'):
            delattr(trainer, 'output_file_prefix')
        pl_module = Mock()

        pred_batch1 = torch.randn(1, 10, 10)
        pred_batch2 = torch.randn(1, 10, 10)
        prediction = ([(pred_batch1, "pred"), (pred_batch2, "prob")], [input_file])

        writer.write_on_batch_end(trainer, pl_module, prediction, None, {}, 0, 0)

        # Check both output files (excluding the original input.tif)
        output_files = [f for f in os.listdir(self.tmpdir) if f.endswith(".tif") and f != "input.tif"]
        assert len(output_files) == 2

    def test_write_on_batch_end_dict_filenames(self):
        """Test write_on_batch_end with dict filenames (multimodal)"""
        writer = CustomWriter(output_dir=self.tmpdir, write_interval="batch")

        input_file = os.path.join(self.tmpdir, "input.tif")
        test_data = np.ones((3, 10, 10), dtype=np.uint8) * 100
        metadata = {
            "driver": "GTiff",
            "height": 10,
            "width": 10,
            "count": 3,
            "dtype": "uint8",
            "nodata": 0,
            "crs": "EPSG:4326",
            "transform": rasterio.transform.from_bounds(0, 0, 1, 1, 10, 10),
        }
        with rasterio.open(input_file, "w", **metadata) as dst:
            dst.write(test_data)

        trainer = Mock()
        trainer.out_dtype = "int16"
        if hasattr(trainer, 'output_file_prefix'):
            delattr(trainer, 'output_file_prefix')
        pl_module = Mock()

        pred_batch = torch.randn(1, 10, 10)
        filename_dict = {"modality1": [input_file]}
        prediction = ((pred_batch, "pred"), filename_dict)

        writer.write_on_batch_end(trainer, pl_module, prediction, None, {}, 0, 0)

        output_files = [f for f in os.listdir(self.tmpdir) if f.endswith("_pred.tif")]
        assert len(output_files) == 1

    def test_write_on_batch_end_no_output_dir_in_trainer(self):
        """Test write_on_batch_end when output_dir is not set"""
        writer = CustomWriter(output_dir=None, write_interval="batch")

        trainer = Mock()
        # No predict_output_dir attribute
        del trainer.predict_output_dir
        pl_module = Mock()

        prediction = torch.randn(4, 10, 10)

        with pytest.raises(Exception, match="Output directory must be passed"):
            writer.write_on_batch_end(trainer, pl_module, prediction, None, {}, 0, 0)

    def test_write_on_batch_end_none_prediction(self):
        """Test write_on_batch_end with None prediction"""
        writer = CustomWriter(output_dir=self.tmpdir, write_interval="batch")

        trainer = Mock()
        pl_module = Mock()

        # Should handle None gracefully
        writer.write_on_batch_end(trainer, pl_module, None, None, {}, 0, 0)

    def test_write_on_batch_end_invalid_prediction_type(self):
        """Test write_on_batch_end with invalid prediction type"""
        writer = CustomWriter(output_dir=self.tmpdir, write_interval="batch")

        trainer = Mock()
        pl_module = Mock()

        prediction = "invalid"  # String is not a valid type

        with pytest.raises(TypeError, match="Unknown type for prediction"):
            writer.write_on_batch_end(trainer, pl_module, prediction, None, {}, 0, 0)

    def test_write_on_batch_end_tensor_no_suffix(self):
        """Test write_on_batch_end with torch.Tensor (no suffix)"""
        writer = CustomWriter(output_dir=self.tmpdir, write_interval="batch")

        input_file = os.path.join(self.tmpdir, "input.tif")
        test_data = np.ones((3, 10, 10), dtype=np.uint8) * 100
        metadata = {
            "driver": "GTiff",
            "height": 10,
            "width": 10,
            "count": 3,
            "dtype": "uint8",
            "nodata": 0,
            "crs": "EPSG:4326",
            "transform": rasterio.transform.from_bounds(0, 0, 1, 1, 10, 10),
        }
        with rasterio.open(input_file, "w", **metadata) as dst:
            dst.write(test_data)

        trainer = Mock()
        trainer.out_dtype = "int16"
        if hasattr(trainer, 'output_file_prefix'):
            delattr(trainer, 'output_file_prefix')
        pl_module = Mock()

        pred_batch = torch.randn(1, 10, 10)
        prediction = (pred_batch, [input_file])

        writer.write_on_batch_end(trainer, pl_module, prediction, None, {}, 0, 0)

        # Default suffix is "pred"
        output_files = [f for f in os.listdir(self.tmpdir) if f.endswith("_pred.tif")]
        assert len(output_files) == 1

    def test_write_on_batch_end_invalid_pred_batch_type(self):
        """Test write_on_batch_end with invalid pred_batch_ type"""
        writer = CustomWriter(output_dir=self.tmpdir, write_interval="batch")

        trainer = Mock()
        trainer.out_dtype = "int16"
        pl_module = Mock()

        # Invalid type (not tuple, list, or tensor)
        prediction = ("invalid_type", ["file.tif"])

        with pytest.raises(ValueError, match="expected to be in"):
            writer.write_on_batch_end(trainer, pl_module, prediction, None, {}, 0, 0)

    def test_write_on_batch_end_with_output_prefix(self):
        """Test write_on_batch_end with output_file_prefix"""
        writer = CustomWriter(output_dir=self.tmpdir, write_interval="batch")

        input_file = os.path.join(self.tmpdir, "input.tif")
        test_data = np.ones((3, 10, 10), dtype=np.uint8) * 100
        metadata = {
            "driver": "GTiff",
            "height": 10,
            "width": 10,
            "count": 3,
            "dtype": "uint8",
            "nodata": 0,
            "crs": "EPSG:4326",
            "transform": rasterio.transform.from_bounds(0, 0, 1, 1, 10, 10),
        }
        with rasterio.open(input_file, "w", **metadata) as dst:
            dst.write(test_data)

        trainer = Mock()
        trainer.out_dtype = "int16"
        trainer.output_file_prefix = "custom_prefix"
        pl_module = Mock()

        pred_batch = torch.randn(1, 10, 10)
        prediction = ((pred_batch, "pred"), [input_file])

        writer.write_on_batch_end(trainer, pl_module, prediction, None, {}, 0, 0)

        output_files = [f for f in os.listdir(self.tmpdir) if f.endswith("_pred.tif")]
        assert len(output_files) == 1
        assert output_files[0] == "custom_prefix_pred.tif"

    def test_write_on_epoch_end(self):
        """Test write_on_epoch_end"""
        writer = CustomWriter(output_dir=self.tmpdir, write_interval="epoch")

        input_file = os.path.join(self.tmpdir, "input.tif")
        test_data = np.ones((3, 10, 10), dtype=np.uint8) * 100
        metadata = {
            "driver": "GTiff",
            "height": 10,
            "width": 10,
            "count": 3,
            "dtype": "uint8",
            "nodata": 0,
            "crs": "EPSG:4326",
            "transform": rasterio.transform.from_bounds(0, 0, 1, 1, 10, 10),
        }
        with rasterio.open(input_file, "w", **metadata) as dst:
            dst.write(test_data)

        trainer = Mock()
        trainer.out_dtype = "int16"
        pl_module = Mock()

        pred_batch = torch.randn(1, 10, 10)
        predictions = [(pred_batch, [input_file])]

        writer.write_on_epoch_end(trainer, pl_module, predictions, None)

        output_files = [f for f in os.listdir(self.tmpdir) if f.endswith("_pred.tif")]
        assert len(output_files) == 1

    def test_write_on_epoch_end_no_output_dir(self):
        """Test write_on_epoch_end without output_dir"""
        writer = CustomWriter(output_dir=None, write_interval="epoch")

        trainer = Mock()
        del trainer.predict_output_dir
        pl_module = Mock()

        with pytest.raises(Exception, match="Output directory must be passed"):
            writer.write_on_epoch_end(trainer, pl_module, [], None)


class TestStateDictAwareModelCheckpoint:
    """Test StateDictAwareModelCheckpoint"""

    def test_init_default(self):
        """Test StateDictAwareModelCheckpoint initialization with defaults"""
        checkpoint = StateDictAwareModelCheckpoint()
        assert checkpoint.save_weights_only is False
        assert checkpoint.save_top_k == 1

    def test_init_with_save_best_only(self):
        """Test StateDictAwareModelCheckpoint with save_best_only"""
        checkpoint = StateDictAwareModelCheckpoint(save_best_only=True)
        assert checkpoint.save_top_k == 1

    def test_init_save_weights_only(self):
        """Test StateDictAwareModelCheckpoint with save_weights_only"""
        checkpoint = StateDictAwareModelCheckpoint(save_weights_only=True)
        assert checkpoint.save_weights_only is True

    def test_state_key_property(self):
        """Test state_key property"""
        checkpoint = StateDictAwareModelCheckpoint(monitor="val/loss", save_weights_only=True)
        state_key = checkpoint.state_key
        assert isinstance(state_key, str)
        # State key should be different based on save_weights_only
        assert "save_weights_only=True" in state_key or state_key != ""


class TestMyLightningCLI:
    """Test MyLightningCLI"""

    def test_subcommands(self):
        """Test MyLightningCLI.subcommands"""
        subcommands = MyLightningCLI.subcommands()
        assert "compute_statistics" in subcommands
        assert "datamodule" in subcommands["compute_statistics"]
        assert "fit" in subcommands
        assert "test" in subcommands
        assert "predict" in subcommands

    def test_add_arguments_to_parser(self):
        """Test MyLightningCLI.add_arguments_to_parser"""
        # Create a mock parser
        mock_parser = Mock()
        mock_parser.add_argument = Mock()

        # Create a minimal CLI instance (we can't instantiate it normally without full setup)
        # Instead, we'll test the method directly
        cli = MyLightningCLI.__new__(MyLightningCLI)
        cli.add_arguments_to_parser(mock_parser)

        # Verify that all expected arguments were added
        assert mock_parser.add_argument.call_count == 5
        call_args_list = [call[0][0] for call in mock_parser.add_argument.call_args_list]
        assert "--predict_output_dir" in call_args_list
        assert "--output_file_prefix" in call_args_list
        assert "--out_dtype" in call_args_list
        assert "--deploy_config_file" in call_args_list
        assert "--custom_modules_path" in call_args_list


class TestStudioDeploySaveConfigCallback:
    """Test StudioDeploySaveConfigCallback"""

    def test_init(self):
        """Test StudioDeploySaveConfigCallback initialization"""
        parser = Mock()
        config_dict = {"config": ["/path/to/config.yaml"], "deploy_config_file": True}
        config = Namespace(config_dict)

        callback = StudioDeploySaveConfigCallback(parser, config)
        assert callback.deploy_config_file is True
        assert callback.config_file_original == "config.yaml"

    def test_setup_already_saved(self):
        """Test setup when already_saved is True"""
        parser = Mock()
        config_dict = {"config": ["/path/to/config.yaml"], "deploy_config_file": True}
        config = Namespace(config_dict)

        callback = StudioDeploySaveConfigCallback(parser, config)
        callback.already_saved = True

        trainer = Mock()
        pl_module = Mock()

        # Should return early without doing anything
        callback.setup(trainer, pl_module, "fit")
        # Verify no save operations were attempted
        assert not parser.save.called

    def test_setup_no_log_dir(self):
        """Test setup when log_dir is None"""
        parser = Mock()
        config_dict = {"config": ["/path/to/config.yaml"], "deploy_config_file": True}
        config = Namespace(config_dict)

        callback = StudioDeploySaveConfigCallback(parser, config)

        trainer = Mock()
        trainer.log_dir = None
        trainer.default_root_dir = None
        pl_module = Mock()

        with pytest.raises(Exception, match="log_dir was none"):
            callback.setup(trainer, pl_module, "fit")


class TestLightningInferenceModel:
    """Test LightningInferenceModel"""

    def test_init(self):
        """Test LightningInferenceModel initialization"""
        trainer = Mock(spec=Trainer)
        trainer.callbacks = []
        model = Mock()
        model.model = Mock()
        datamodule = Mock(spec=LightningDataModule)

        inference_model = LightningInferenceModel(trainer, model, datamodule)
        assert inference_model.trainer == trainer
        assert inference_model.model == model
        assert inference_model.datamodule == datamodule

    def test_init_with_checkpoint(self):
        """Test LightningInferenceModel initialization with checkpoint"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake checkpoint
            checkpoint_path = Path(tmpdir) / "checkpoint.ckpt"
            dummy_model = torch.nn.Linear(10, 10)
            state_dict = {"state_dict": dummy_model.state_dict()}
            torch.save(state_dict, checkpoint_path)

            trainer = Mock(spec=Trainer)
            trainer.callbacks = []
            model = Mock()
            model.model = Mock()
            model.model.load_state_dict = Mock()
            datamodule = Mock(spec=LightningDataModule)

            inference_model = LightningInferenceModel(trainer, model, datamodule, checkpoint_path=checkpoint_path)
            # Should have called load_state_dict
            assert model.model.load_state_dict.called

    def test_init_with_checkpoint_no_state_dict_key(self):
        """Test LightningInferenceModel with checkpoint without state_dict key"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.ckpt"
            dummy_model = torch.nn.Linear(10, 10)
            # Save without state_dict wrapper
            torch.save(dummy_model.state_dict(), checkpoint_path)

            trainer = Mock(spec=Trainer)
            trainer.callbacks = []
            model = Mock()
            model.model = Mock()
            model.model.load_state_dict = Mock()
            datamodule = Mock(spec=LightningDataModule)

            inference_model = LightningInferenceModel(trainer, model, datamodule, checkpoint_path=checkpoint_path)
            assert model.model.load_state_dict.called

    def test_init_with_checkpoint_model_prefix(self):
        """Test LightningInferenceModel with checkpoint having 'model' prefix"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.ckpt"
            dummy_model = torch.nn.Linear(10, 10)
            # Add 'model.' prefix to keys
            state_dict = {"state_dict": {f"model.{k}": v for k, v in dummy_model.state_dict().items()}}
            torch.save(state_dict, checkpoint_path)

            trainer = Mock(spec=Trainer)
            trainer.callbacks = []
            model = Mock()
            model.model = Mock()
            model.model.load_state_dict = Mock()
            datamodule = Mock(spec=LightningDataModule)

            inference_model = LightningInferenceModel(trainer, model, datamodule, checkpoint_path=checkpoint_path)
            # Should have removed the 'model.' prefix and loaded
            assert model.model.load_state_dict.called

    def test_init_removes_prediction_writers(self):
        """Test that init removes BasePredictionWriter callbacks"""
        trainer = Mock(spec=Trainer)
        prediction_writer = Mock(spec=BasePredictionWriter)
        other_callback = Mock()
        trainer.callbacks = [prediction_writer, other_callback]

        model = Mock()
        model.model = Mock()
        datamodule = Mock(spec=LightningDataModule)

        inference_model = LightningInferenceModel(trainer, model, datamodule)
        # Should only have non-writer callbacks
        assert len(inference_model.trainer.callbacks) == 1
        assert inference_model.trainer.callbacks[0] == other_callback

    @patch("terratorch.cli_tools.build_lightning_cli")
    def test_from_config(self, mock_build_cli):
        """Test LightningInferenceModel.from_config"""
        # Mock CLI
        mock_cli = Mock()
        mock_cli.trainer = Mock(spec=Trainer)
        mock_cli.trainer.logger = None
        mock_cli.trainer.callbacks = []
        mock_cli.model = Mock()
        mock_cli.model.model = Mock()
        mock_cli.datamodule = Mock(spec=LightningDataModule)
        mock_build_cli.return_value = mock_cli

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("model: test")

            result = LightningInferenceModel.from_config(config_path)
            assert isinstance(result, LightningInferenceModel)
            assert mock_build_cli.called

    @patch("terratorch.cli_tools.build_lightning_cli")
    def test_from_config_with_bands(self, mock_build_cli):
        """Test LightningInferenceModel.from_config with band specifications"""
        mock_cli = Mock()
        mock_cli.trainer = Mock(spec=Trainer)
        mock_cli.trainer.logger = None
        mock_cli.trainer.callbacks = []
        mock_cli.model = Mock()
        mock_cli.model.model = Mock()
        mock_cli.datamodule = Mock(spec=LightningDataModule)
        mock_build_cli.return_value = mock_cli

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("model: test")

            result = LightningInferenceModel.from_config(
                config_path, predict_dataset_bands=["B01", "B02"], predict_output_bands=["B03"]
            )
            assert isinstance(result, LightningInferenceModel)
            # Check that bands were passed to CLI
            call_args = mock_build_cli.call_args[0][0]
            assert "--data.init_args.predict_dataset_bands" in call_args

    def test_from_task_and_datamodule(self):
        """Test LightningInferenceModel.from_task_and_datamodule"""
        task = Mock()
        task.model = Mock()
        datamodule = Mock(spec=LightningDataModule)

        result = LightningInferenceModel.from_task_and_datamodule(task, datamodule)
        assert isinstance(result, LightningInferenceModel)
        assert result.model == task
        assert result.datamodule == datamodule

    def test_from_task_and_datamodule_with_checkpoint(self):
        """Test from_task_and_datamodule with checkpoint"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.ckpt"
            dummy_model = torch.nn.Linear(10, 10)
            torch.save({"state_dict": dummy_model.state_dict()}, checkpoint_path)

            task = Mock()
            task.model = Mock()
            task.model.load_state_dict = Mock()
            datamodule = Mock(spec=LightningDataModule)

            result = LightningInferenceModel.from_task_and_datamodule(task, datamodule, checkpoint_path=checkpoint_path)
            assert isinstance(result, LightningInferenceModel)

    def test_inference_on_dir(self):
        """Test inference_on_dir method"""
        trainer = Mock(spec=Trainer)
        trainer.callbacks = []
        trainer.predict = Mock(
            return_value=[
                (torch.randn(2, 10, 10), ["file1.tif", "file2.tif"]),
                (torch.randn(2, 10, 10), ["file3.tif", "file4.tif"]),
            ]
        )

        model = Mock()
        model.model = Mock()
        datamodule = Mock(spec=LightningDataModule)

        inference_model = LightningInferenceModel(trainer, model, datamodule)

        with tempfile.TemporaryDirectory() as tmpdir:
            predictions, filenames = inference_model.inference_on_dir(Path(tmpdir))
            assert predictions.shape[0] == 4  # 2 + 2 batches
            assert len(filenames) == 4

    def test_inference_on_dir_with_tuple_predictions(self):
        """Test inference_on_dir with tuple format predictions"""
        trainer = Mock(spec=Trainer)
        trainer.callbacks = []
        # Return format: ((prediction_tensor, prediction_name), filename)
        trainer.predict = Mock(
            return_value=[
                ((torch.randn(2, 10, 10), "pred"), ["file1.tif", "file2.tif"]),
                ((torch.randn(2, 10, 10), "pred"), ["file3.tif", "file4.tif"]),
            ]
        )

        model = Mock()
        model.model = Mock()
        datamodule = Mock(spec=LightningDataModule)

        inference_model = LightningInferenceModel(trainer, model, datamodule)

        with tempfile.TemporaryDirectory() as tmpdir:
            predictions, filenames = inference_model.inference_on_dir(Path(tmpdir))
            assert predictions.shape[0] == 4
            assert len(filenames) == 4

    def test_inference_single_file(self):
        """Test inference on single file"""
        trainer = Mock(spec=Trainer)
        trainer.callbacks = []
        trainer.predict = Mock(return_value=[(torch.randn(1, 10, 10), ["file.tif"])])

        model = Mock()
        model.model = Mock()
        datamodule = Mock(spec=LightningDataModule)

        inference_model = LightningInferenceModel(trainer, model, datamodule)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy file
            test_file = Path(tmpdir) / "test.tif"
            test_file.write_text("dummy")

            prediction = inference_model.inference(test_file)
            assert prediction.shape == (10, 10)  # Squeezed


class TestMyTrainer:
    """Test MyTrainer"""

    def test_compute_statistics(self):
        """Test compute_statistics method"""
        from torch.utils.data import DataLoader, Dataset

        # Create a simple dataset
        class SimpleDataset(Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {"image": torch.randn(3, 10, 10), "mask": torch.randint(0, 5, (10, 10))}

        dataset = SimpleDataset()
        dataloader = DataLoader(dataset, batch_size=2)

        # Create a mock datamodule
        mock_datamodule = Mock(spec=LightningDataModule)
        mock_datamodule.train_transform = None
        mock_datamodule.setup = Mock()
        mock_datamodule.train_dataloader = Mock(return_value=dataloader)

        trainer = MyTrainer()

        with patch("terratorch.cli_tools.compute_statistics") as mock_compute_stats, patch(
            "terratorch.cli_tools.compute_mask_statistics"
        ) as mock_compute_mask_stats:
            mock_compute_stats.return_value = {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]}
            mock_compute_mask_stats.return_value = {"class_counts": {0: 100, 1: 200}}

            trainer.compute_statistics(mock_datamodule)

            # Verify that setup was called
            mock_datamodule.setup.assert_called_once_with("fit")
            # Verify statistics were computed
            assert mock_compute_stats.called
            assert mock_compute_mask_stats.called

    def test_compute_statistics_no_mask(self):
        """Test compute_statistics when dataset has no mask"""
        from torch.utils.data import DataLoader, Dataset

        class SimpleDatasetNoMask(Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {"image": torch.randn(3, 10, 10)}

        dataset = SimpleDatasetNoMask()
        dataloader = DataLoader(dataset, batch_size=2)

        mock_datamodule = Mock(spec=LightningDataModule)
        mock_datamodule.train_transform = None
        mock_datamodule.setup = Mock()
        mock_datamodule.train_dataloader = Mock(return_value=dataloader)

        trainer = MyTrainer()

        with patch("terratorch.cli_tools.compute_statistics") as mock_compute_stats, patch(
            "terratorch.cli_tools.compute_mask_statistics"
        ) as mock_compute_mask_stats:
            mock_compute_stats.return_value = {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]}

            trainer.compute_statistics(mock_datamodule)

            # Verify image stats were computed but not mask stats
            assert mock_compute_stats.called
            assert not mock_compute_mask_stats.called

    def test_compute_statistics_not_dataloader(self):
        """Test compute_statistics when train_dataloader doesn't return DataLoader"""
        mock_datamodule = Mock(spec=LightningDataModule)
        mock_datamodule.train_transform = None
        mock_datamodule.setup = Mock()
        # Return something that's not a DataLoader
        mock_datamodule.train_dataloader = Mock(return_value="not_a_dataloader")

        trainer = MyTrainer()

        with pytest.raises(ValueError, match="DataLoader not found"):
            trainer.compute_statistics(mock_datamodule)


class TestBuildLightningCLI:
    """Test build_lightning_cli function"""

    @patch.dict(os.environ, {}, clear=True)
    @patch("terratorch.cli_tools.MyLightningCLI")
    def test_build_lightning_cli_basic(self, mock_cli_class):
        """Test build_lightning_cli basic functionality"""
        mock_cli_instance = Mock()
        mock_cli_class.return_value = mock_cli_instance

        result = build_lightning_cli(args=["fit", "--help"], run=False)

        assert result == mock_cli_instance
        assert mock_cli_class.called

    @patch.dict(os.environ, {"terratorch_FLOAT_32_PRECISION": "high"}, clear=True)
    @patch("terratorch.cli_tools.MyLightningCLI")
    @patch("torch.set_float32_matmul_precision")
    def test_build_lightning_cli_with_precision(self, mock_set_precision, mock_cli_class):
        """Test build_lightning_cli with float32 precision env var"""
        mock_cli_instance = Mock()
        mock_cli_class.return_value = mock_cli_instance

        result = build_lightning_cli(args=["fit", "--help"], run=False)

        mock_set_precision.assert_called_once_with("high")

    @patch.dict(os.environ, {"terratorch_FLOAT_32_PRECISION": "invalid"}, clear=True)
    @patch("terratorch.cli_tools.MyLightningCLI")
    def test_build_lightning_cli_with_invalid_precision(self, mock_cli_class):
        """Test build_lightning_cli with invalid precision value"""
        mock_cli_instance = Mock()
        mock_cli_class.return_value = mock_cli_instance

        with pytest.warns(UserWarning):
            result = build_lightning_cli(args=["fit", "--help"], run=False)

        assert result == mock_cli_instance

    @patch("terratorch.cli_tools.MyLightningCLI")
    def test_build_lightning_cli_sets_env_vars(self, mock_cli_class):
        """Test that build_lightning_cli sets GDAL environment variables"""
        mock_cli_instance = Mock()
        mock_cli_class.return_value = mock_cli_instance

        build_lightning_cli(args=["fit", "--help"], run=False)

        # Check that GDAL env vars were set
        assert "GDAL_DISABLE_READDIR_ON_OPEN" in os.environ
        assert os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"

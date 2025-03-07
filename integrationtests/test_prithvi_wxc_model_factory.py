# Copyright contributors to the Terratorch project

import os
import sys

import pytest
import torch
import torch.distributed as dist
import yaml
from granitewxc.utils.config import get_config
from huggingface_hub import hf_hub_download
from lightning.pytorch import Trainer

from terratorch.models.wxc_model_factory import WxCModelFactory
from terratorch.tasks.wxc_task import WxCTask
import lightning.pytorch as pl

from terratorch.datamodules.era5 import ERA5DataModule
from terratorch.tasks.wxc_task import WxCTask
from typing import Any
from terratorch.datamodules.merra2_downscale import Merra2DownscaleNonGeoDataModule
from granitewxc.utils.data import _get_transforms


def setup_function():
    print("\nSetup function is called")

def teardown_function():
    try:
        os.remove("config.yaml")
    except OSError:
        pass

class StopTrainerCallback(pl.Callback):
    def __init__(self, stop_after_n_batches):
        super().__init__()
        self.stop_after_n_batches = stop_after_n_batches
        self.current_batch = 0

    def on_predict_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Any,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.current_batch += 1
        if self.current_batch >= self.stop_after_n_batches:
            print("Stopping training early...")
            #trainer.should_stop = True
            raise StopIteration("Stopped prediction after reaching the specified batch limit.")

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Any,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.current_batch += 1
        if self.current_batch >= self.stop_after_n_batches:
            print("Stopping training early...")
            #trainer.should_stop = True
            raise StopIteration("Stopped prediction after reaching the specified batch limit.")

@pytest.mark.parametrize("backbone", ["gravitywave", None, 'prithviwxc'])
def test_can_create_wxc_models(backbone):
    if backbone == "gravitywave":
        config_data = {
            "singular_sharded_checkpoint": "./examples/notebooks/magnet-flux-uvtp122-epoch-99-loss-0.1022.pt",
        }

        with open("config.yaml", "w") as file:
            yaml.dump(config_data, file, default_flow_style=False)

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        if dist.is_initialized():
            dist.destroy_process_group()

        dist.init_process_group(
            backend='gloo',
            init_method='env://',
            rank=0,
            world_size=1
        )

        f = WxCModelFactory()
        f.build_model(backbone, None)

    elif backbone == 'prithviwxc':
        f = WxCModelFactory()
        f.build_model(backbone, aux_decoders = None, backbone_weights='/dccstor/wfm/shared/pretrained/step_400.pt')

    else:
        config = get_config('./examples/confs/granite-wxc-merra2-downscale-config.yaml')
        config.download_path = "/dccstor/wfm/shared/datasets/training/merra-2_v1/"

        config.data.data_path_surface = os.path.join(config.download_path,'merra-2')
        config.data.data_path_vertical = os.path.join(config.download_path, 'merra-2')
        config.data.climatology_path_surface = os.path.join(config.download_path,'climatology')
        config.data.climatology_path_vertical = os.path.join(config.download_path,'climatology')

        config.model.input_scalers_surface_path = os.path.join(config.download_path,'climatology/musigma_surface.nc')
        config.model.input_scalers_vertical_path = os.path.join(config.download_path,'climatology/musigma_vertical.nc')
        config.model.output_scalers_surface_path = os.path.join(config.download_path,'climatology/anomaly_variance_surface.nc')
        config.model.output_scalers_vertical_path = os.path.join(config.download_path,'climatology/anomaly_variance_vertical.nc')
        f = WxCModelFactory()
        f.build_model(backbone, aux_decoders = None, model_config=config)



def test_wxc_unet_pincer_inference():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if dist.is_initialized():
        dist.destroy_process_group()

    dist.init_process_group(
        backend='gloo',
        init_method='env://',
        rank=0,
        world_size=1
    )

    hf_hub_download(
        repo_id="Prithvi-WxC/Gravity_wave_Parameterization",
        filename=f"magnet-flux-uvtp122-epoch-99-loss-0.1022.pt",
        local_dir=".",
    )

    hf_hub_download(
    )

    hf_hub_download(
        repo_id="Prithvi-WxC/Gravity_wave_Parameterization",
        repo_type='dataset',
        filename=f"wxc_input_u_v_t_p_output_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling05.nc",
        local_dir=".",
    )

    model_args = {
        "in_channels": 1280,
        "input_size_time": 1,
        "n_lats_px": 64,
        "n_lons_px": 128,
        "patch_size_px": [2, 2],
        "mask_unit_size_px": [8, 16],
        "mask_ratio_inputs": 0.5,
        "embed_dim": 2560,
        "n_blocks_encoder": 12,
        "n_blocks_decoder": 2,
        "mlp_multiplier": 4,
        "n_heads": 16,
        "dropout": 0.0,
        "drop_path": 0.05,
        "parameter_dropout": 0.0,
        "residual": "none",
        "masking_mode": "both",
        "decoder_shifting": False,
        "positional_encoding": "absolute",
        "checkpoint_encoder": [3, 6, 9, 12, 15, 18, 21, 24],
        "checkpoint_decoder": [1, 3],
        "in_channels_static": 3,
        "input_scalers_mu": torch.tensor([0] * 1280),
        "input_scalers_sigma": torch.tensor([1] * 1280),
        "input_scalers_epsilon": 0,
        "static_input_scalers_mu": torch.tensor([0] * 3),
        "static_input_scalers_sigma": torch.tensor([1] * 3),
        "static_input_scalers_epsilon": 0,
        "output_scalers": torch.tensor([0] * 1280),
        "backbone_weights": "magnet-flux-uvtp122-epoch-99-loss-0.1022.pt",
        "backbone": "prithviwxc",
        "aux_decoders": "unetpincer",
    }
    task = WxCTask(WxCModelFactory(), model_args=model_args, mode='eval')

    trainer = Trainer(
        max_epochs=1,
        callbacks=[StopTrainerCallback(stop_after_n_batches=3)],
    )
    dm = ERA5DataModule(train_data_path='.', valid_data_path='.')
    results = trainer.predict(model=task, datamodule=dm, return_predictions=True)

    dist.destroy_process_group()

def test_wxc_downscaling_pincer_instantiate():
    kwargs = {
        "in_channels": 1280,
        "input_size_time": 1,
        "n_lats_px": 64,
        "n_lons_px": 128,
        "in_channels_static": 3,
        "input_scalers_mu": torch.tensor([0] * 1280),
        "input_scalers_sigma": torch.tensor([1] * 1280),
        "input_scalers_epsilon": 0,
        "static_input_scalers_mu": torch.tensor([0] * 3),
        "static_input_scalers_sigma": torch.tensor([1] * 3),
        "static_input_scalers_epsilon": 0,
        "output_scalers": torch.tensor([0] * 1280),
        "patch_size_px": [2, 2],
        "mask_unit_size_px": [8, 16],
        "mask_ratio_inputs": 0.5,
        "embed_dim": 2560,
        "n_blocks_encoder": 12,
        "n_blocks_decoder": 2,
        "mlp_multiplier": 4,
        "n_heads": 16,
        "dropout": 0.0,
        "drop_path": 0.05,
        "parameter_dropout": 0.0,
        "residual": "none",
        "masking_mode": "both",
        "positional_encoding": "absolute",
        "config_path": "./integrationtests/test_prithvi_wxc_model_factory_config.yaml",
    }

    WxCModelFactory().build_model(backbone="prithviwxc", aux_decoders="downscaler", **kwargs)

def test_wxc_downscaling_pincer_task():
    model_args = {
        "backbone":  "prithviwxc",
        "aux_decoders": "downscaler",
        "in_channels": 1280,
        "input_size_time": 1,
        "n_lats_px": 64,
        "n_lons_px": 128,
        "in_channels_static": 3,
        "input_scalers_mu": torch.tensor([0] * 1280),
        "input_scalers_sigma": torch.tensor([1] * 1280),
        "input_scalers_epsilon": 0,
        "static_input_scalers_mu": torch.tensor([0] * 3),
        "static_input_scalers_sigma": torch.tensor([1] * 3),
        "static_input_scalers_epsilon": 0,
        "output_scalers": torch.tensor([0] * 1280),
        "patch_size_px": [2, 2],
        "mask_unit_size_px": [8, 16],
        "mask_ratio_inputs": 0.5,
        "embed_dim": 2560,
        "n_blocks_encoder": 12,
        "n_blocks_decoder": 2,
        "mlp_multiplier": 4,
        "n_heads": 16,
        "dropout": 0.0,
        "drop_path": 0.05,
        "parameter_dropout": 0.0,
        "residual": "none",
        "masking_mode": "both",
        "positional_encoding": "absolute",
        "config_path": "./integrationtests/test_prithvi_wxc_model_factory_config.yaml",
    }

    task = WxCTask('WxCModelFactory', model_args=model_args, mode='train')

def test_wxc_downscaling_pincer_predict():
    model_args = {
        "backbone":  "prithviwxc",
        "aux_decoders": "downscaler",
        "in_channels": 1280,
        "input_size_time": 1,
        "n_lats_px": 64,
        "n_lons_px": 128,
        "in_channels_static": 3,
        "input_scalers_mu": torch.tensor([0] * 1280),
        "input_scalers_sigma": torch.tensor([1] * 1280),
        "input_scalers_epsilon": 0,
        "static_input_scalers_mu": torch.tensor([0] * 3),
        "static_input_scalers_sigma": torch.tensor([1] * 3),
        "static_input_scalers_epsilon": 0,
        "output_scalers": torch.tensor([0] * 1280),
        "patch_size_px": [2, 2],
        "mask_unit_size_px": [8, 16],
        "mask_ratio_inputs": 0.5,
        "embed_dim": 2560,
        "n_blocks_encoder": 12,
        "n_blocks_decoder": 2,
        "mlp_multiplier": 4,
        "n_heads": 16,
        "dropout": 0.0,
        "drop_path": 0.05,
        "parameter_dropout": 0.0,
        "residual": "none",
        "masking_mode": "both",
        "positional_encoding": "absolute",
        "config_path": "./integrationtests/test_prithvi_wxc_model_factory_config_ccc.yaml",
        "checkpoint_path": "pytorch_model.bin",
        "wxc_auxiliary_data_path": "/dccstor/wfm/shared/datasets/training/merra-2_v1/",
    }


    task = WxCTask('WxCModelFactory', model_args=model_args, mode='train')

    config = get_config("./integrationtests/test_prithvi_wxc_model_factory_config_ccc.yaml")
    dm = Merra2DownscaleNonGeoDataModule(
        time_range=('2020-01-01T00:00:00', '2020-01-01T23:59:59'),
        data_path_surface = config.data.data_path_surface,
        data_path_vertical = config.data.data_path_vertical,
        climatology_path_surface = config.data.climatology_path_surface,
        climatology_path_vertical = config.data.climatology_path_vertical,
        input_surface_vars = config.data.input_surface_vars,
        input_static_surface_vars = config.data.input_static_surface_vars,
        input_vertical_vars = config.data.input_vertical_vars,
        input_levels = config.data.input_levels,
        n_input_timestamps = config.data.n_input_timestamps,
        output_vars=config.data.output_vars,
        transforms=_get_transforms(config),
    )
    dm.setup('predict')

    trainer = Trainer(
        max_epochs=1,
    )
    results = trainer.predict(model=task, datamodule=dm)

def test_wxc_unet_pincer_train():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if dist.is_initialized():
        dist.destroy_process_group()

    dist.init_process_group(
        backend='gloo',
        init_method='env://',
        rank=0,
        world_size=1
    )

    hf_hub_download(
        repo_id="Prithvi-WxC/Gravity_wave_Parameterization",
        filename=f"magnet-flux-uvtp122-epoch-99-loss-0.1022.pt",
        local_dir=".",
    )

    hf_hub_download(
        repo_id="Prithvi-WxC/Gravity_wave_Parameterization",
        filename=f"config.yaml",
        local_dir=".",
    )

    hf_hub_download(
        repo_id="Prithvi-WxC/Gravity_wave_Parameterization",
        repo_type='dataset',
        filename=f"wxc_input_u_v_t_p_output_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling05.nc",
        local_dir=".",
    )

    model_args = {
        "in_channels": 1280,
        "input_size_time": 1,
        "n_lats_px": 64,
        "n_lons_px": 128,
        "patch_size_px": [2, 2],
        "mask_unit_size_px": [8, 16],
        "mask_ratio_inputs": 0.5,
        "embed_dim": 2560,
        "n_blocks_encoder": 12,
        "n_blocks_decoder": 2,
        "mlp_multiplier": 4,
        "n_heads": 16,
        "dropout": 0.0,
        "drop_path": 0.05,
        "parameter_dropout": 0.0,
        "residual": "none",
        "masking_mode": "both",
        "decoder_shifting": False,
        "positional_encoding": "absolute",
        "checkpoint_encoder": [3, 6, 9, 12, 15, 18, 21, 24],
        "checkpoint_decoder": [1, 3],
        "in_channels_static": 3,
        "input_scalers_mu": torch.tensor([0] * 1280),
        "input_scalers_sigma": torch.tensor([1] * 1280),
        "input_scalers_epsilon": 0,
        "static_input_scalers_mu": torch.tensor([0] * 3),
        "static_input_scalers_sigma": torch.tensor([1] * 3),
        "static_input_scalers_epsilon": 0,
        "output_scalers": torch.tensor([0] * 1280),
        "backbone_weights": "magnet-flux-uvtp122-epoch-99-loss-0.1022.pt",
        "backbone": "prithviwxc",
        "aux_decoders": "unetpincer",
        "skip_connection": True,
    }

    task = WxCTask(WxCModelFactory(), model_args=model_args, mode='train')

    trainer = Trainer(
        callbacks=[StopTrainerCallback(stop_after_n_batches=3)],
        max_epochs=1,
    )
    dm = ERA5DataModule(train_data_path='.', valid_data_path='.')
    results = trainer.fit(model=task, datamodule=dm)

    dist.destroy_process_group()


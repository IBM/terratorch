# Copyright contributors to the Terratorch project

import os

import pytest
import torch
import torch.distributed as dist
import yaml
from granitewxc.utils.config import get_config
from huggingface_hub import hf_hub_download
from lightning.pytorch import Trainer

from terratorch.models.wxc_model_factory import WxCModelFactory
from terratorch.tasks.wxc_task import WxCTask


def setup_function():
    print("\nSetup function is called")

def teardown_function():
    try:
        os.remove("config.yaml")
    except OSError:
        pass


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

    from prithviwxc.gravitywave.datamodule import ERA5DataModule
    from terratorch.tasks.wxc_task import WxCTask

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
    }
    task = WxCTask(WxCModelFactory(), model_args=model_args, mode='eval')

    trainer = Trainer(
        max_epochs=1,
    )
    dm = ERA5DataModule(train_data_path='.', valid_data_path='.')
    results = trainer.predict(model=task, datamodule=dm, return_predictions=True)

    dist.destroy_process_group()


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

    from prithviwxc.gravitywave.datamodule import ERA5DataModule
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
        "backbone_weights": "magnet-flux-uvtp122-epoch-99-loss-0.1022.pt"
    }
    task = WxCTask(WxCModelFactory(), model_args=model_args, mode='train')

    trainer = Trainer(
        max_epochs=1,
    )
    dm = ERA5DataModule(train_data_path='.', valid_data_path='.')
    results = trainer.fit(model=task, datamodule=dm)

    dist.destroy_process_group()


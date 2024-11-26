# Copyright contributors to the Terratorch project

import pytest
import yaml
from granitewxc.utils.config import get_config
from lightning.pytorch import Trainer
import os
from huggingface_hub import hf_hub_download
import torch.distributed as dist



from terratorch.models.wxc_model_factory import WxCModelFactory

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
    from terratorch.tasks.wxc_gravity_wave_task import WxCGravityWaveTask
    task = WxCGravityWaveTask(WxCModelFactory(), mode='eval')

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
    from terratorch.tasks.wxc_gravity_wave_task import WxCGravityWaveTask
    task = WxCGravityWaveTask(WxCModelFactory(), mode='train')

    trainer = Trainer(
        max_epochs=1,
    )
    dm = ERA5DataModule(train_data_path='.', valid_data_path='.')
    results = trainer.fit(model=task, datamodule=dm)

    dist.destroy_process_group()


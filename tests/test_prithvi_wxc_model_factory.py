# Copyright contributors to the Terratorch project

import pytest
import os
import torch.distributed as dist
import yaml
from granitewxc.utils.config import get_config



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
        config.download_path = '/home/romeokienzler/Downloads/'

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



def test_dummy():
    None


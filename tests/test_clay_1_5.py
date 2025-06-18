import gc
import torch
import yaml
from box import Box
import rasterio
import numpy as np
import pystac_client
import stackstac
import geopandas as gpd
import pandas as pd
from shapely import Point
from rasterio.enums import Resampling
from torchvision.transforms import v2
import math
from terratorch.models.clay1_5_model_factory import Clay1_5ModelFactory



def test_create_model():

    with open("../model/configs/metadata.yaml", "r") as f:
        metadata_contents = yaml.safe_load(f)

    #print(metadata_contents['sentinel-2-l2a']['bands'])

    #print(metadata_contents['sentinel-2-l2a'].bands.wavelength.values())
    
    model_args = {
        # ENCODER
        "dim": 192,
        "depth": 6,
        "heads": 4,
        "dim_head": 48,
        "mlp_ratio": 2,
        # DECODER
        "decoder_dim": 96,
        "decoder_depth": 3,
        "decoder_heads": 2,
        "decoder_dim_head": 48,
        "decoder_mlp_ratio": 2,
        "mask_ratio": 0.75,
        "norm_pix_loss": False,
        "patch_size": 8,
        "shuffle": True,
        "metadata": Box(metadata_contents),
        "teacher": "vit_large_patch14_reg4_dinov2.lvd142m",
        "dolls": [16, 32, 64, 128, 256, 768, 1024],
        "doll_weights": [1, 1, 1, 1, 1, 1, 1],
        "in_channels": None
    }

    clay_model = Clay1_5ModelFactory().build_model(None, None, None, **model_args)
    data_cube = {
        "pixels": torch.randn(64, 10, 64, 64), 
        "time": torch.stack([torch.zeros(4) for _ in range(64)]),
        "platform": ["sentinel-2-l2a"],
        "latlon": torch.zeros(64,4),
        "waves": torch.zeros(4),
        "gsd": torch.tensor(10),
    }

    clay_model.forward(data_cube)
    #clay_model.forward(torch.randn(1, 3, 224, 224))

    gc.collect()


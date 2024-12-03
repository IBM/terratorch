#!/usr/bin/env python
# coding: utf-8

# # Prithvi WxC Downscaling: Model Inference using TerraTorch
# This notebook is a walk through to use a finetuned downscaling model to generate inferences using TerraTorch. We show how to initalize the model, load weights, and use the model for inference using TerraTorch.
# 
# Note to set up your environment by running the following cells. (We recommend to run this notebook in an empty pyton 3.11 environment)  
# (e.g.,   
# python3.11 -m venv .venv  
# source .venv/bin/activate  
# )  
# 
# We assume that you've cloned terratorch with:  
# git clone https://github.com/IBM/terratorch.git  
# And you run this notebook from terratorch/examples/notebooks  

# 

import terratorch # this import is needed to initialize TT's factories
from lightning.pytorch import Trainer
import os, glob
from granitewxc.utils.config import get_config
import torch
from terratorch.tasks.wxc_downscaling_task import WxCDownscalingTask
from terratorch.datamodules.merra2_downscale import Merra2DownscaleNonGeoDataModule
from granitewxc.utils.data import _get_transforms
from huggingface_hub import hf_hub_download, snapshot_download

files = glob.glob("merra-2/*")

if not len(files):

    snapshot_download(
        repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
        allow_patterns="merra-2/MERRA2_sfc_2020010[1].nc",
        local_dir=".",
    )

    snapshot_download(
        repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
        allow_patterns="merra-2/MERRA_pres_2020010[1].nc",
        local_dir=".",
    )


# 
# ## Climatology
# 
# The PrithviWxC model was trained to calculate the output by producing a perturbation to the climatology at the target time. This mode of operation is set via the residual=climate option. This was chosen as climatology is typically a strong prior for long-range prediction. When using the residual=climate option, we have to provide the dataloader with the path of the climatology data.
# 

# 

files = glob.glob("climatology/*")

if not len(files):
    snapshot_download(
        repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
        allow_patterns="climatology/climate_surface_doy00[1]*.nc",
        local_dir=".",
    )

    snapshot_download(
        repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
        allow_patterns="climatology/climate_vertical_doy00[1]*.nc",
        local_dir=".",
    )

    hf_hub_download(
        repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
        filename=f"climatology/anomaly_variance_surface.nc",
        local_dir=".",
    )

    hf_hub_download(
        repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
        filename=f"climatology/anomaly_variance_vertical.nc",
        local_dir=".",
    )

    hf_hub_download(
        repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
        filename=f"climatology/musigma_surface.nc",
        local_dir=".",
    )

    hf_hub_download(
        repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
        filename=f"climatology/musigma_vertical.nc",
        local_dir=".",
    )

# 



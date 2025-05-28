import os
import gc
import pandas as pd
import pytest
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
import datetime
import torch.distributed as dist
from terratorch.datamodules.era5 import ERA5DataModule
from terratorch.datamodules.era5 import ERA5Dataset


import numpy as np
import xarray as xr
import datetime
import os

def create_era5_dummy_data(base_dir: str = '.', filename: str = 'era5dummy.nc') -> None:
    n_time = 4
    n_idim = 491
    n_odim = 366
    n_lat = 64
    n_lon = 128

    time = np.array([datetime.datetime(2020, 1, 1, 0) + datetime.timedelta(hours=i)
                     for i in range(n_time)], dtype='datetime64[ns]')
    idim = np.arange(n_idim, dtype=np.int32)
    odim = np.arange(n_odim, dtype=np.int32)
    lat = np.linspace(-0.9763, 0.9763, n_lat, dtype=np.float32)
    lon = np.linspace(0.0, 0.9922, n_lon, dtype=np.float32)

    features = np.random.randn(n_time, n_idim, n_lat, n_lon).astype(np.float32)
    output = np.random.randn(n_time, n_odim, n_lat, n_lon).astype(np.float32)

    ds = xr.Dataset(
        {
            "features": (("time", "idim", "lat", "lon"), features),
            "output": (("time", "odim", "lat", "lon"), output),
        },
        coords={
            "time": time,
            "idim": idim,
            "odim": odim,
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "title": "Dummy ERA5 Dataset (idim=491)",
            "source": "Synthetic",
        }
    )

    file_path = os.path.join(base_dir, filename)
    ds.to_netcdf(file_path)

def test_era5_datamodule():
    create_era5_dummy_data() 

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
    )


    datamodule = ERA5DataModule(
        train_data_path=".",
        valid_data_path=".",
        file_glob_pattern="era5dummy.nc",
        batch_size=2,
        num_data_workers=0,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()

    batch = next(iter(train_loader))
    assert "x" in batch
    assert "y" in batch
    assert "target" in batch
    assert "lead_time" in batch
    assert "static" in batch

    assert batch["x"].shape == (2, 1, 488, 64, 128)
    assert batch["y"].shape == (2, 488, 64, 128)
    assert batch["lead_time"].shape == (2, 1)
    assert batch["static"].shape == (2, 3, 64, 128)
    gc.collect()

    datamodule.setup("predict")
    
    predict_dataloader = datamodule.predict_dataloader()
    next(iter(predict_dataloader))

    val_dataloader = datamodule.val_dataloader()
    next(iter(val_dataloader))
    gc.collect()

    try:
        os.remove('era5dummy.nc')
    except FileNotFoundError:
        pass

    if dist.is_initialized():
        dist.destroy_process_group()

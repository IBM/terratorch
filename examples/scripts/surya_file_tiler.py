import glob
import os

import numpy as np
import xarray as xr

data_root = "experiment"
output_data_root = "experiment_tiled"
if not os.path.isdir(output_data_root):
    os.mkdir(output_data_root)

aimed_shape = (256, 256)
files = glob.glob(os.path.join(data_root, "*.nc"))

for file in files:
    data = xr.open_dataset(file, engine="h5netcdf")
    variables = list(data.variables.keys())
    shape = data[variables[0]].shape
    n_tiles_x = shape[0] // aimed_shape[0]
    n_tiles_y = shape[1] // aimed_shape[1]

    for i in range(1, n_tiles_x):
        for j in range(1, n_tiles_y):
            dataset = []
            for v in variables:
                data_sample_v = data[v][
                    aimed_shape[0] * (i - 1) : aimed_shape[0] * i, aimed_shape[1] * (j - 1) : aimed_shape[1] * j
                ]
                dataset.append(data_sample_v)
            datarray = xr.merge(dataset)
            filename = os.path.basename(file)
            datarray.to_netcdf(os.path.join(output_data_root, filename.replace(".nc", f"_tile_{i}_{j}.nc")), "w")

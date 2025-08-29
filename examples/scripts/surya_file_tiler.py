import xarray as xr
import glob
import os 

data_root = "experiment"
aimed_shape = (256,256)
files = glob.glob(os.path.join(data_root, "*.nc"))

for file in files:

    data = xr.open_dataset(file, engine="h5netcdf")
    shape = data.shape

    n_tiles_x = shape[0]//aimed_shape[0]
    n_tiles_y = shape[1]//aimed_shape[1]

    for i in range(1, n_tiles_x):
        for j in range(1, n_tiles_y):

            data_sample = data[aimed_shape[0]*(i-1):aimed_shape[0]*i,
                               aimed_shape[1]*(j-1):aimed_shape[1]*j]

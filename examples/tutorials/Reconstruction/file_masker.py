import argparse
import glob
import os

import numpy as np
import rasterio
import tifffile

parser = argparse.ArgumentParser("Arguments", add_help=False)

# data loader related
parser.add_argument("--input_dir", type=str)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

files = glob.glob(os.path.join(input_dir, "*.tif"))

for ff in files:
    filename = os.path.basename(ff)
    filename_mod = filename.replace(".tif", "_masked.tif")
    ff_mod = os.path.join(output_dir, filename_mod)

    data = tifffile.imread(ff)
    mask = np.random.choice([0, 1], data.shape[:-1], p=np.array([0.6, 0.4]))
    output = mask[..., None] * data
    output = output.transpose(1, 2, 0)

    with rasterio.open(
        ff_mod,
        "w",
        driver="GTiff",
        height=output.shape[1],
        width=output.shape[2],
        count=output.shape[0],
        dtype=output.dtype,
        crs="+proj=latlong",
    ) as dst:
        dst.write(output)

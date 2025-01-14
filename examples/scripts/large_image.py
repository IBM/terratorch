from PIL import Image
import os
import random
import numpy as np
import tifffile as tiff 
from argparse import ArgumentParser
from osgeo import gdal
from osgeo import osr

parser = ArgumentParser()

args = parser.parse_args()

# config
GDAL_DATA_TYPE = gdal.GDT_Int32
GEOTIFF_DRIVER_NAME = r'GTiff'
NO_DATA = 15
SPATIAL_REFERENCE_SYSTEM_WKID = 4326
n_bands = 10
img_size = 6_000
img_size = 2*(img_size,) + (n_bands,)

output_file  = os.path.join("large_image.tiff")

output = np.random.rand(*img_size)

 # create driver
driver = gdal.GetDriverByName(GEOTIFF_DRIVER_NAME)

output_raster = driver.Create(output_file,
                              output.shape[1],
                              output.shape[0],
                              output.shape[-1],
                              eType = GDAL_DATA_TYPE)


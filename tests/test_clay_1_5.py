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

def get_data_cube():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    lat, lon = 37.30939, -8.57207
    start = "2018-07-01"
    end = "2018-09-01"
    STAC_API = "https://earth-search.aws.element84.com/v1"
    COLLECTION = "sentinel-2-l2a"

    catalog = pystac_client.Client.open(STAC_API)
    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{start}/{end}",
        bbox=(lon - 1e-5, lat - 1e-5, lon + 1e-5, lat + 1e-5),
        max_items=100,
        query={"eo:cloud_cover": {"lt": 80}},
    )

    all_items = search.get_all_items()

    items = []
    dates = []
    for item in all_items:
        if item.datetime.date() not in dates:
            items.append(item)
            dates.append(item.datetime.date())

    print(f"Found {len(items)} items")

    asset_href = items[0].assets["red"].href

    with rasterio.open(asset_href) as src:
        epsg = src.crs.to_epsg()
        print(f"EPSG: {epsg}")

    poidf = gpd.GeoDataFrame(
        pd.DataFrame(),
        crs="EPSG:4326",
        geometry=[Point(lon, lat)],
    ).to_crs(epsg)

    coords = poidf.iloc[0].geometry.coords[0]

    size = 256
    gsd = 10
    bounds = (
        coords[0] - (size * gsd) // 2,
        coords[1] - (size * gsd) // 2,
        coords[0] + (size * gsd) // 2,
        coords[1] + (size * gsd) // 2,
    )

    dtype="float32"
    fill_value=0.0  # Python float (defaults to float64)

    stack = stackstac.stack(
        items,
        bounds=bounds,
        snap_bounds=False,
        epsg=epsg,
        resolution=gsd,
        dtype="float32",
        rescale=False,
        fill_value=np.float32(0.0),  # now matches float32
        assets=["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"],
    )

    # Extract mean, std, and wavelengths from metadata
    platform = "sentinel-2-l2a"
    metadata = Box(yaml.safe_load(open("../model/configs/metadata.yaml")))
    mean = []
    std = []
    waves = []
    # Use the band names to get the correct values in the correct order.
    for band in stack.band:
        mean.append(metadata[platform].bands.mean[str(band.values)])
        std.append(metadata[platform].bands.std[str(band.values)])
        waves.append(metadata[platform].bands.wavelength[str(band.values)])

    # Prepare the normalization transform function using the mean and std values.
    transform = v2.Compose(
        [
            v2.Normalize(mean=mean, std=std),
        ]
    )
    
    # Prep datetimes embedding using a normalization function from the model code.
    def normalize_timestamp(date):
        week = date.isocalendar().week * 2 * np.pi / 52
        hour = date.hour * 2 * np.pi / 24

        return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


    datetimes = stack.time.values.astype("datetime64[s]").tolist()
    times = [normalize_timestamp(dat) for dat in datetimes]
    week_norm = [dat[0] for dat in times]
    hour_norm = [dat[1] for dat in times]


    # Prep lat/lon embedding using the
    def normalize_latlon(lat, lon):
        lat = lat * np.pi / 180
        lon = lon * np.pi / 180

        return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))


    latlons = [normalize_latlon(lat, lon)] * len(times)
    lat_norm = [dat[0] for dat in latlons]
    lon_norm = [dat[1] for dat in latlons]

    # Normalize pixels
    pixels = torch.from_numpy(stack.values.astype(np.float32))

    #pixels = torch.from_numpy(stack.data.astype(np.float32))
    pixels = transform(pixels)

    datacube = {
        "platform": [platform],
        "time": torch.tensor(
            np.hstack((week_norm, hour_norm)),
            dtype=torch.float32,
            device=device,
        ),
        "latlon": torch.tensor(
            np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=device
        ),
        "pixels": pixels.to(device),
        "gsd": torch.tensor(stack.gsd.values, device=device),
        "waves": torch.tensor(waves, device=device),
    }
    return datacube

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
    data_cube = get_data_cube()
    print(f"Data cube: {data_cube.keys()}")
    print(f"Data cube pixels shape: {data_cube['pixels'].shape}")
    clay_model.forward(data_cube)
    #clay_model.forward(torch.randn(1, 3, 224, 224))

    gc.collect()


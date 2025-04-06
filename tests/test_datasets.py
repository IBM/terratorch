import binascii
import json
import pickle
import os
import random
import re
from pathlib import Path
from typing import Any

import albumentations as A
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from matplotlib.figure import Figure
from PIL import Image
from pytest import MonkeyPatch
from rasterio.transform import from_origin
from xarray import DataArray
from torchgeo.datasets import NonGeoDataset
from torchgeo.transforms import AugmentationSequential

from terratorch.datasets import (
    FireScarsNonGeo,
    MBeninSmallHolderCashewsNonGeo,
    MBigEarthNonGeo,
    MBrickKilnNonGeo,
    MChesapeakeLandcoverNonGeo,
    MEuroSATNonGeo,
    MForestNetNonGeo,
    MNeonTreeNonGeo,
    MNzCattleNonGeo,
    MPv4gerNonGeo,
    MPv4gerSegNonGeo,
    MSACropTypeNonGeo,
    MSo2SatNonGeo,
    MultiTemporalCropClassification,
    #Sen1Floods11NonGeo,
)
from terratorch.datasets.sen1floods11 import Sen1Floods11NonGeo
from terratorch.datasets.utils import pad_numpy, to_tensor

from terratorch.datasets.transforms import FlattenTemporalIntoChannels, UnflattenTemporalFromChannels

# Add necessary imports for OpenSentinelMap and testing utilities
import sys
# Add the terratorch root to the path to allow finding the module
# This might need adjustment depending on how tests are run
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terratorch.datasets import OpenSentinelMap


def create_dummy_tiff(path, width=100, height=100, count=6, dtype="uint8"):
    data = np.random.randint(0, 255, (count, height, width), dtype=dtype)
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform
    ) as dst:
        dst.write(data)

@pytest.fixture(scope="function")
def mpv4ger_data_root(tmp_path):
    data_root = tmp_path / "m_pv4ger"
    data_directory = data_root / "m-pv4ger"
    data_directory.mkdir(parents=True, exist_ok=True)

    partition = {
        "train": ["10.0,20.0", "11.0,21.0"],
        "val": [],
        "test": []
    }
    partition_file = data_directory / "default_partition.json"
    with open(partition_file, "w") as f:
        json.dump(partition, f)

    bands = ["BLUE", "GREEN", "RED"]

    for img_id in partition["train"]:
        file_path = data_directory / f"{img_id}.hdf5"
        with h5py.File(file_path, "w") as h5file:

            for band in bands:
                h5file.create_dataset(band, data=np.random.rand(100, 100).astype(np.float32))


            attr_dict = {"label": np.random.randint(0, 10)}
            serialized_attr = pickle.dumps(attr_dict)
            hex_attr = binascii.hexlify(serialized_attr).decode("ascii")
            h5file.attrs["pickle"] = hex_attr

    return str(data_root)

@pytest.fixture(scope="function")
def fire_scars_data_root(tmp_path):
    data_root = tmp_path / "fire_scars"
    split = "train"
    split_dir = data_root / FireScarsNonGeo.splits[split]
    split_dir.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        random_seq = "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), 5))
        year = 2021
        julian_day = i + 1
        date = f"{year}{julian_day:03d}"
        image_filename = f"subsetted_512x512_HLS.S30.T{random_seq}.{date}.v1.4_merged.tif"
        mask_filename = f"subsetted_512x512_HLS.S30.T{random_seq}.{date}.v1.4.mask.tif"
        image_path = split_dir / image_filename
        mask_path = split_dir / mask_filename

        create_dummy_tiff(image_path)
        create_dummy_tiff(mask_path, count=1, dtype="uint8")

    image_files = list(split_dir.glob("*_merged.tif"))
    mask_files = list(split_dir.glob("*.mask.tif"))

    assert len(image_files) == 5, f"Expected 5 image files, but found {len(image_files)}"
    assert len(mask_files) == 5, f"Expected 5 mask files, but found {len(mask_files)}"

    return str(data_root)

@pytest.fixture(scope="function")
def m_bigearth_data_root(tmp_path):
    data_root = tmp_path / "m_bigearthnet"
    data_directory = data_root / MBigEarthNonGeo.data_dir
    data_directory.mkdir(parents=True, exist_ok=True)

    label_map = {f"image_{i}": [0, 1] for i in range(2)}
    label_map_path = data_directory / MBigEarthNonGeo.label_map_file
    with open(label_map_path, "w") as f:
        json.dump(label_map, f)

    partition = {"train": ["image_0", "image_1"], "val": [], "test": []}
    partition_file = data_directory / MBigEarthNonGeo.partition_file_template.format(partition="default")
    with open(partition_file, "w") as f:
        json.dump(partition, f)

    bands = [
        "COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1",
        "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW",
        "WATER_VAPOR", "SWIR_1", "SWIR_2", "CLOUD_PROBABILITY"
    ]

    for img in ["image_0", "image_1"]:
        file_path = data_directory / f"{img}.hdf5"
        with h5py.File(file_path, "w") as h5file:
            for band in bands:
                h5file.create_dataset(band, data=np.random.rand(100, 100).astype(np.float32))

    image_files = list(data_directory.glob("*.hdf5"))
    assert len(image_files) == 2, f"Expected 2 image files, but found {len(image_files)}"

    return str(data_root)

@pytest.fixture(scope="function")
def m_forestnet_data_root(tmp_path):
    data_root = tmp_path / "m_forestnet"
    data_directory = data_root / MForestNetNonGeo.data_dir
    data_directory.mkdir(parents=True, exist_ok=True)

    partition = {
        "train": ["image_0", "image_1"],
        "val": [],
        "test": []
    }
    partition_file = data_directory / MForestNetNonGeo.partition_file_template.format(partition="default")
    with open(partition_file, "w") as f:
        json.dump(partition, f)

    bands = ["BLUE", "GREEN", "RED", "NIR", "SWIR_1", "SWIR_2"]
    label_map = {"image_0": 0, "image_1": 1}

    for img in ["image_0", "image_1"]:
        file_path = data_directory / f"{img}.hdf5"
        with h5py.File(file_path, "w") as h5file:
            for band in bands:
                h5file.create_dataset(band, data=np.random.rand(100, 100).astype(np.float32))

            label_dict = {"label": label_map[img]}
            pickle_bytes = pickle.dumps(label_dict)
            pickle_str = repr(pickle_bytes)
            h5file.attrs["pickle"] = pickle_str

    image_files = list(data_directory.glob("*.hdf5"))
    assert len(image_files) == 2, f"Expected 2 image files, but found {len(image_files)}"

    return str(data_root)

@pytest.fixture(scope="function")
def mnz_cattle_data_root(tmp_path):
    data_root = tmp_path / "m_nz_cattle"
    data_directory = data_root / MNzCattleNonGeo.data_dir
    data_directory.mkdir(parents=True, exist_ok=True)

    partition = {
        "train": ["image_0", "image_1"],
        "val": [],
        "test": []
    }
    partition_file = data_directory / MNzCattleNonGeo.partition_file_template.format(partition="default")
    with open(partition_file, "w") as f:
        json.dump(partition, f)

    bands = ["BLUE_2023-01-01", "GREEN_2023-01-01", "RED_2023-01-01"]

    for img in ["image_0", "image_1"]:
        file_path = data_directory / f"{img}.hdf5"
        with h5py.File(file_path, "w") as h5file:
            for band in bands:
                h5file.create_dataset(band, data=np.random.rand(100, 100).astype(np.float32))

            mask = np.random.randint(0, 2, size=(100, 100)).astype(np.uint8)
            h5file.create_dataset("label", data=mask)

    image_files = list(data_directory.glob("*.hdf5"))
    assert len(image_files) == 2, f"Expected 2 image files, but found {len(image_files)}"

    return str(data_root)

@pytest.fixture(scope="function")
def brickkiln_data_root(tmp_path):
    data_root = tmp_path / "m_brick_kiln"
    data_directory = data_root / MBrickKilnNonGeo.data_dir
    data_directory.mkdir(parents=True, exist_ok=True)

    partition = {
        "train": ["image_0", "image_1"],
        "val": [],
        "test": []
    }
    partition_file = data_directory / MBrickKilnNonGeo.partition_file_template.format(partition="default")
    with open(partition_file, "w") as f:
        json.dump(partition, f)

    bands = [
        "COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3",
        "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2"
    ]

    for img in ["image_0", "image_1"]:
        file_path = data_directory / f"{img}.hdf5"
        with h5py.File(file_path, "w") as h5file:
            for band in bands:
                h5file.create_dataset(band, data=np.random.rand(100, 100).astype(np.float32))

            attr_dict = {"label": 1 if img == "image_0" else 0}
            h5file.attrs["pickle"] = str(pickle.dumps(attr_dict))

    return str(data_root)

@pytest.fixture(scope="function")
def neontree_data_root(tmp_path):
    data_root = tmp_path / "m_neon_tree"
    data_directory = data_root / MNeonTreeNonGeo.data_dir
    data_directory.mkdir(parents=True, exist_ok=True)

    partition = {
        "train": ["image_0", "image_1"],
        "val": [],
        "test": []
    }
    partition_file = data_directory / MNeonTreeNonGeo.partition_file_template.format(partition="default")
    with open(partition_file, "w") as f:
        json.dump(partition, f)

    bands = ["BLUE", "CANOPY_HEIGHT_MODEL", "GREEN", "NEON", "RED"]

    for img in ["image_0", "image_1"]:
        file_path = data_directory / f"{img}.hdf5"
        with h5py.File(file_path, "w") as h5file:
            for band in bands:
                h5file.create_dataset(band, data=np.random.rand(100, 100).astype(np.float32))

            mask = np.random.randint(0, 3, size=(100, 100), dtype=np.uint8)
            h5file.create_dataset("label", data=mask)

    return str(data_root)

@pytest.fixture(scope="function")
def eurosat_data_root(tmp_path):
    data_root = tmp_path / "m_eurosat"
    data_directory = data_root / MEuroSATNonGeo.data_dir
    data_directory.mkdir(parents=True, exist_ok=True)

    partition = {
        "train": ["image_0", "image_1"],
        "val": [],
        "test": []
    }
    partition_file = data_directory / MEuroSATNonGeo.partition_file_template.format(partition="default")
    with open(partition_file, "w") as f:
        json.dump(partition, f)

    label_map = {
        "class_0": ["image_0"],
        "class_1": ["image_1"]
    }
    label_map_file = data_directory / MEuroSATNonGeo.label_map_file
    with open(label_map_file, "w") as f:
        json.dump(label_map, f)

    bands = [
        "COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3",
        "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2"
    ]

    for img in ["image_0", "image_1"]:
        file_path = data_directory / f"{img}.hdf5"
        with h5py.File(file_path, "w") as h5file:
            for band in bands:
                h5file.create_dataset(band, data=np.random.rand(100, 100).astype(np.float32))

            label_tensor = torch.tensor(0 if img == "image_0" else 1, dtype=torch.long)
            h5file.create_dataset("label", data=label_tensor.numpy())

    return str(data_root)

@pytest.fixture(scope="function")
def pv4gerseg_data_root(tmp_path):
    data_root = tmp_path / "m_pv4ger_seg"
    data_directory = data_root / MPv4gerSegNonGeo.data_dir
    data_directory.mkdir(parents=True, exist_ok=True)

    partition = {
        "train": ["52.5167,13.3833", "48.8566,2.3522"],
        "val": [],
        "test": []
    }
    partition_file = data_directory / MPv4gerSegNonGeo.partition_file_template.format(partition="default")
    with open(partition_file, "w") as f:
        json.dump(partition, f)

    bands = ["BLUE", "GREEN", "RED"]

    for img in ["52.5167,13.3833", "48.8566,2.3522"]:
        file_path = data_directory / f"{img}.hdf5"
        with h5py.File(file_path, "w") as h5file:
            for band in bands:
                h5file.create_dataset(band, data=np.random.rand(100, 100).astype(np.float32))

            mask = np.random.randint(0, 3, size=(100, 100), dtype=np.uint8)
            h5file.create_dataset("label", data=mask)

    return str(data_root)

@pytest.fixture(scope="function")
def pv4ger_data_root(tmp_path):
    data_root = tmp_path / "m_pv4ger"
    data_directory = data_root / MPv4gerNonGeo.data_dir
    data_directory.mkdir(parents=True, exist_ok=True)

    partition = {
        "train": ["52.5167,13.3833", "48.8566,2.3522"],
        "val": [],
        "test": []
    }
    partition_file = data_directory / MPv4gerNonGeo.partition_file_template.format(partition="default")
    with open(partition_file, "w") as f:
        json.dump(partition, f)

    bands = ["BLUE", "GREEN", "RED"]

    for img in ["52.5167,13.3833", "48.8566,2.3522"]:
        file_path = data_directory / f"{img}.hdf5"
        with h5py.File(file_path, "w") as h5file:
            for band in bands:
                h5file.create_dataset(band, data=np.random.rand(100, 100).astype(np.float32))

            attr_dict = {"label": 1 if img == "52.5167,13.3833" else 0}
            h5file.attrs["pickle"] = str(pickle.dumps(attr_dict))

    return str(data_root)

@pytest.fixture(scope="function")
def so2sat_data_root(tmp_path):
    data_root = tmp_path / "m_so2sat"
    data_directory = data_root / MSo2SatNonGeo.data_dir
    data_directory.mkdir(parents=True, exist_ok=True)

    partition = {
        "train": ["image_0", "image_1"],
        "val": [],
        "test": []
    }
    partition_file = data_directory / MSo2SatNonGeo.partition_file_template.format(partition="default")
    with open(partition_file, "w") as f:
        json.dump(partition, f)

    bands = [
        "VH_REAL", "BLUE", "VH_IMAGINARY", "GREEN", "VV_REAL", "RED",
        "VV_IMAGINARY", "VH_LEE_FILTERED", "RED_EDGE_1", "VV_LEE_FILTERED",
        "RED_EDGE_2", "VH_LEE_FILTERED_REAL", "RED_EDGE_3", "NIR_BROAD",
        "VV_LEE_FILTERED_IMAGINARY", "NIR_NARROW", "SWIR_1", "SWIR_2"
    ]

    for img in ["image_0", "image_1"]:
        file_path = data_directory / f"{img}.hdf5"
        with h5py.File(file_path, "w") as h5file:
            for band in bands:
                h5file.create_dataset(band, data=np.random.rand(100, 100).astype(np.float32))

            attr_dict = {"label": 1 if img == "image_0" else 0}
            h5file.attrs["pickle"] = str(pickle.dumps(attr_dict))

    return str(data_root)

@pytest.fixture(scope="function")
def cashews_data_root(tmp_path):
    data_root = tmp_path / "m_cashew_plant"
    data_directory = data_root / MBeninSmallHolderCashewsNonGeo.data_dir
    data_directory.mkdir(parents=True, exist_ok=True)

    partition = {
        "train": ["image_2023-01-15", "image_2023-02-20"],
        "val": [],
        "test": []
    }
    partition_file = data_directory / MBeninSmallHolderCashewsNonGeo.partition_file_template.format(partition="default")
    with open(partition_file, "w") as f:
        json.dump(partition, f)

    bands = [
        "COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2",
        "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "SWIR_1", "SWIR_2", "CLOUD_PROBABILITY"
    ]

    for img in ["image_2023-01-15", "image_2023-02-20"]:
        file_path = data_directory / f"{img}.hdf5"
        with h5py.File(file_path, "w") as h5file:
            for band in bands:
                date_band_name = f"{band}_2023-01-15" if img == "image_2023-01-15" else f"{band}_2023-02-20"
                h5file.create_dataset(date_band_name, data=np.random.rand(100, 100).astype(np.float32))

            mask = np.random.randint(0, 3, size=(100, 100), dtype=np.uint8)
            h5file.create_dataset("label", data=mask)

    return str(data_root)

@pytest.fixture(scope="function")
def sen1floods_data_root(tmp_path):
    data_root = tmp_path / "sen1floods11"
    (data_root / Sen1Floods11NonGeo.data_dir).mkdir(parents=True, exist_ok=True)
    (data_root / Sen1Floods11NonGeo.label_dir).mkdir(parents=True, exist_ok=True)
    (data_root / Sen1Floods11NonGeo.split_dir).mkdir(parents=True, exist_ok=True)

    for i in range(5):
        filename = f"tile_{i}_S2Hand.tif"
        label_filename = f"tile_{i}_LabelHand.tif"

        img_data = DataArray(
            np.random.rand(13, 64, 64).astype(np.float32),
            dims=["band", "y", "x"]
        )
        mask_data = DataArray(
            np.random.randint(0, 2, size=(1, 64, 64), dtype=np.uint8),
            dims=["band", "y", "x"]
        )

        img_data.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        mask_data.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

        image_path = data_root / Sen1Floods11NonGeo.data_dir / filename
        mask_path = data_root / Sen1Floods11NonGeo.label_dir / label_filename

        img_data.rio.to_raster(str(image_path))
        mask_data.rio.to_raster(str(mask_path))

    split_file = data_root / Sen1Floods11NonGeo.split_dir / "flood_train_data.txt"
    with open(split_file, "w") as f:
        f.write("\n".join([f"tile_{i}" for i in range(5)]))

    metadata = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"location": f"tile_{i}", "s2_date": "2021-01-01"},
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            }
            for i in range(5)
        ],
    }
    metadata_path = data_root / Sen1Floods11NonGeo.metadata_file
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    return str(data_root)

@pytest.fixture(scope="function")
def chesapeake_data_root(tmp_path):
    data_root = tmp_path / "m_chesapeake"
    data_directory = data_root / MChesapeakeLandcoverNonGeo.data_dir
    data_directory.mkdir(parents=True, exist_ok=True)

    partition = {
        "train": ["image_0", "image_1"],
        "val": [],
        "test": []
    }
    partition_file = data_directory / MChesapeakeLandcoverNonGeo.partition_file_template.format(partition="default")
    with open(partition_file, "w") as f:
        json.dump(partition, f)

    bands = ["BLUE", "GREEN", "NIR", "RED"]

    for img in ["image_0", "image_1"]:
        file_path = data_directory / f"{img}.hdf5"
        with h5py.File(file_path, "w") as h5file:
            for band in bands:
                h5file.create_dataset(band, data=np.random.rand(100, 100).astype(np.float32))

            mask = np.random.randint(0, 3, size=(100, 100), dtype=np.uint8)
            h5file.create_dataset("label", data=mask)

    return str(data_root)

@pytest.fixture(scope="function")
def crop_classification_data_root(tmp_path):
    data_root = tmp_path / "crop_classification"
    training_dir = data_root / "training_chips"
    validation_dir = data_root / "validation_chips"

    training_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    for directory in [training_dir, validation_dir]:
        for i in range(2):
            filename = f"chip_{i}_merged.tif"
            label_filename = f"chip_{i}.mask.tif"
            img_data = DataArray(np.random.rand(18, 64, 64).astype(np.float32), dims=["band", "y", "x"])
            mask_data = DataArray(np.random.randint(0, 13, size=(1, 64, 64), dtype=np.uint8), dims=["band", "y", "x"])

            img_data.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
            mask_data.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
            img_data.rio.write_crs("EPSG:4326", inplace=True)
            mask_data.rio.write_crs("EPSG:4326", inplace=True)
            image_path = directory / filename
            mask_path = directory / label_filename
            img_data.rio.to_raster(str(image_path))
            mask_data.rio.to_raster(str(mask_path))

    with open(data_root / "training_data.txt", "w") as f:
        f.write("\n".join([f"chip_{i}" for i in range(2)]))

    with open(data_root / "validation_data.txt", "w") as f:
        f.write("\n".join([f"chip_{i}" for i in range(2)]))

    metadata = pd.DataFrame({
        "chip_id": [f"chip_{i}" for i in range(2)],
        "first_img_date": ["2021-01-01", "2021-01-02"],
        "middle_img_date": ["2021-01-15", "2021-01-16"],
        "last_img_date": ["2021-02-01", "2021-02-02"],
    })
    metadata.to_csv(data_root / "chips_df.csv", index=False)

    return str(data_root)

# Fixture for creating mock OpenSentinelMap dataset
@pytest.fixture(scope="function") # Use function scope to ensure clean state for each test
def open_sentinel_map_data(tmp_path: Path) -> Path:
    """Create mock data directory for OpenSentinelMap tests."""
    root = tmp_path
    (root / "osm_sentinel_imagery" / "tile1" / "cell1").mkdir(parents=True)
    (root / "osm_sentinel_imagery" / "tile1" / "cell2").mkdir(parents=True)
    (root / "osm_label_images_v10" / "tile1").mkdir(parents=True)

    # Create dummy CSV
    csv_data = {
        "MGRS_tile": ["tile1", "tile1", "tile1"],
        "cell_id": [1, 2, 3], # Cell 3 will have no label file
        "split": ["training", "validation", "testing"]
    }
    pd.DataFrame(csv_data).to_csv(root / "spatial_cell_info.csv", index=False)

    # Create dummy JSON categories
    categories = {"0": "class_a", "1": "class_b"}
    with open(root / "osm_categories.json", "w") as f:
        json.dump(categories, f)

    # Create dummy image files (.npz) for cell1 (3 timestamps)
    for date in ["20230101", "20230105", "20230110"]:
        np.savez(
            root / "osm_sentinel_imagery" / "tile1" / "cell1" / f"{date}_data.npz",
            gsd_10=np.random.rand(192, 192, 12).astype(np.float32), # Match MAX_TEMPORAL_IMAGE_SIZE for simplicity
            gsd_20=np.random.rand(96, 96, 6).astype(np.float32),
            gsd_60=np.random.rand(32, 32, 2).astype(np.float32),
        )
    # Create dummy image files (.npz) for cell2 (1 timestamp)
    np.savez(
        root / "osm_sentinel_imagery" / "tile1" / "cell2" / "20230201_data.npz",
        gsd_10=np.random.rand(192, 192, 12).astype(np.float32),
        gsd_20=np.random.rand(96, 96, 6).astype(np.float32),
        gsd_60=np.random.rand(32, 32, 2).astype(np.float32),
    )

    # Create dummy label files (.png)
    Image.fromarray(np.random.randint(0, 2, size=(192, 192, 1), dtype=np.uint8)).save(
        root / "osm_label_images_v10" / "tile1" / "1.png"
    )
    Image.fromarray(np.random.randint(0, 2, size=(192, 192, 1), dtype=np.uint8)).save(
        root / "osm_label_images_v10" / "tile1" / "2.png"
    )
    # Note: No label file for cell 3, it should be skipped

    return root

# Test Class for OpenSentinelMap
class TestOpenSentinelMap:
    def test_init(self, open_sentinel_map_data: Path) -> None:
        """Test basic initialization."""
        ds = OpenSentinelMap(data_root=str(open_sentinel_map_data), split="train")
        assert ds.split == "training" # Internal mapping check
        assert ds.bands == ["gsd_10", "gsd_20", "gsd_60"] # Default bands
        assert len(ds) == 1 # Only cell 1 is in 'training' split and has a label

        ds_val = OpenSentinelMap(data_root=str(open_sentinel_map_data), split="val", bands=["gsd_10"])
        assert ds_val.split == "validation"
        assert ds_val.bands == ["gsd_10"]
        assert len(ds_val) == 1 # Only cell 2 is in 'validation' split

        ds_test = OpenSentinelMap(data_root=str(open_sentinel_map_data), split="test")
        assert ds_test.split == "testing"
        assert len(ds_test) == 0 # Cell 3 is in testing but has no label file

    def test_invalid_split(self, open_sentinel_map_data: Path) -> None:
        """Test error on invalid split."""
        with pytest.raises(ValueError, match="Split 'invalid_split' not recognized"):
            OpenSentinelMap(data_root=str(open_sentinel_map_data), split="invalid_split")

    def test_invalid_band(self, open_sentinel_map_data: Path) -> None:
        """Test error on invalid band."""
        with pytest.raises(ValueError, match="Band 'invalid_band' is not recognized"):
            OpenSentinelMap(data_root=str(open_sentinel_map_data), bands=["gsd_10", "invalid_band"])

    def test_len(self, open_sentinel_map_data: Path) -> None:
        """Test __len__ method."""
        ds_train = OpenSentinelMap(data_root=str(open_sentinel_map_data), split="train")
        assert len(ds_train) == 1
        ds_val = OpenSentinelMap(data_root=str(open_sentinel_map_data), split="val")
        assert len(ds_val) == 1
        ds_test = OpenSentinelMap(data_root=str(open_sentinel_map_data), split="test")
        assert len(ds_test) == 0 # Cell 3 has no label

    def test_getitem_interpolated_stacked(self, open_sentinel_map_data: Path) -> None:
        """Test __getitem__ with spatial_interpolate_and_stack_temporally=True."""
        ds = OpenSentinelMap(
            data_root=str(open_sentinel_map_data),
            split="train",
            bands=["gsd_10", "gsd_20"], # Use two bands for testing channel concat
            spatial_interpolate_and_stack_temporally=True,
            pick_random_pair=False # Load all timestamps for predictability
        )
        sample = ds[0] # Get the first sample (cell1)

        assert "image" in sample
        assert "mask" in sample
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["mask"], torch.Tensor)

        # Expected shape: (T, C, H, W)
        # T = 3 (timestamps for cell1)
        # C = 12 (gsd_10) + 6 (gsd_20) = 18
        # H = W = 192 (MAX_TEMPORAL_IMAGE_SIZE)
        assert sample["image"].shape == (3, 18, 192, 192)
        assert sample["mask"].shape == (192, 192)
        assert sample["image"].dtype == torch.float32
        assert sample["mask"].dtype == torch.int64 # Default tensor conversion type

    def test_getitem_not_interpolated_stacked(self, open_sentinel_map_data: Path) -> None:
        """Test __getitem__ with spatial_interpolate_and_stack_temporally=False."""
        ds = OpenSentinelMap(
            data_root=str(open_sentinel_map_data),
            split="train",
            bands=["gsd_10", "gsd_20"],
            spatial_interpolate_and_stack_temporally=False,
            pick_random_pair=False
        )
        sample = ds[0]

        assert "image" in sample
        assert "mask" in sample
        assert isinstance(sample["image"], dict) # Should be a dict of bands
        assert isinstance(sample["mask"], torch.Tensor)

        assert "gsd_10" in sample["image"]
        assert "gsd_20" in sample["image"]
        assert isinstance(sample["image"]["gsd_10"], torch.Tensor)
        assert isinstance(sample["image"]["gsd_20"], torch.Tensor)

        # Expected shape: (T, C, H, W)
        # T = 3
        # gsd_10: C=12, H=192, W=192
        # gsd_20: C=6, H=96, W=96 (original size, not interpolated)
        assert sample["image"]["gsd_10"].shape == (3, 12, 192, 192)
        assert sample["image"]["gsd_20"].shape == (3, 6, 96, 96)
        assert sample["mask"].shape == (192, 192)
        assert sample["image"]["gsd_10"].dtype == torch.float32
        assert sample["mask"].dtype == torch.int64

    def test_getitem_padding_truncation(self, open_sentinel_map_data: Path) -> None:
        """Test __getitem__ with padding and truncation."""
        ds = OpenSentinelMap(
            data_root=str(open_sentinel_map_data),
            split="train",
            bands=["gsd_10"],
            spatial_interpolate_and_stack_temporally=True,
            pad_image=5, # Pad to 5 timestamps
            truncate_image=2, # Truncate to last 2 timestamps
            pick_random_pair=False
        )
        sample = ds[0]

        # Original T=3. Truncate to 2 -> T=2. Pad to 5 -> T=5.
        # Expected shape: (T, C, H, W)
        # T = 5
        # C = 12 (gsd_10)
        # H = W = 192
        assert sample["image"].shape == (5, 12, 192, 192)

        # Test only padding
        ds_pad = OpenSentinelMap(
            data_root=str(open_sentinel_map_data),
            split="train",
            pad_image=4, # Pad original 3 to 4
            pick_random_pair=False,
            spatial_interpolate_and_stack_temporally=True,
        )
        sample_pad = ds_pad[0]
        assert sample_pad["image"].shape[0] == 4

        # Test only truncation
        ds_trunc = OpenSentinelMap(
            data_root=str(open_sentinel_map_data),
            split="train",
            truncate_image=1, # Truncate original 3 to 1
            pick_random_pair=False,
            spatial_interpolate_and_stack_temporally=True,
        )
        sample_trunc = ds_trunc[0]
        assert sample_trunc["image"].shape[0] == 1

    def test_getitem_pick_random_pair(self, open_sentinel_map_data: Path) -> None:
        """Test __getitem__ with pick_random_pair=True."""
        ds = OpenSentinelMap(
            data_root=str(open_sentinel_map_data),
            split="train",
            bands=["gsd_10"],
            spatial_interpolate_and_stack_temporally=True,
            pick_random_pair=True # Default, but explicit here
        )
        sample = ds[0]
        # Should pick 2 random timestamps from the 3 available
        assert sample["image"].shape[0] == 2

    def test_getitem_target_channel(self, open_sentinel_map_data: Path) -> None:
        """Test selecting specific target channel from mask."""
        # Modify the fixture to create a multi-channel mask
        mask_path = open_sentinel_map_data / "osm_label_images_v10" / "tile1" / "1.png"
        multi_channel_mask = np.random.randint(0, 2, size=(192, 192, 3), dtype=np.uint8)
        Image.fromarray(multi_channel_mask).save(mask_path)

        ds_target0 = OpenSentinelMap(data_root=str(open_sentinel_map_data), split="train", target=0)
        sample0 = ds_target0[0]
        assert sample0["mask"].shape == (192, 192)

        ds_target1 = OpenSentinelMap(data_root=str(open_sentinel_map_data), split="train", target=1)
        sample1 = ds_target1[0]
        assert sample1["mask"].shape == (192, 192)

        # Check that the masks are different (highly likely with random data)
        assert not torch.equal(sample0["mask"], sample1["mask"])

    def test_plot(self, open_sentinel_map_data: Path) -> None:
        """Test plot method."""
        ds = OpenSentinelMap(data_root=str(open_sentinel_map_data), split="train", bands=["gsd_10"])
        sample = ds[0]
        fig = ds.plot(sample)
        assert isinstance(fig, Figure)
        plt.close(fig) # Close the plot to avoid display issues

        # Test plot with no gsd_10 band (should return None or handle gracefully)
        ds_no_rgb = OpenSentinelMap(data_root=str(open_sentinel_map_data), split="train", bands=["gsd_20"])
        sample_no_rgb = ds_no_rgb[0]
        # The plot method might raise an error or return None if gsd_10 is missing.
        # Based on the code, it returns None gracefully.
        fig_no_rgb = ds_no_rgb.plot(sample_no_rgb)
        assert fig_no_rgb is None

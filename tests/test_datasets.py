import binascii
import json
import pickle

import albumentations as A
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from rasterio.transform import from_origin
from xarray import DataArray

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

from terratorch.datasets.transforms import FlattenTemporalIntoChannels, UnflattenTemporalFromChannels


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

class TestMNeonTreeNonGeo:
    def test_dataset_sample(self, neontree_data_root):
        transform = A.Compose([
            A.Resize(64, 64),
            ToTensorV2(),
        ])

        dataset = MNeonTreeNonGeo(data_root=neontree_data_root, split="train", transform=transform)
        sample = dataset[0]

        assert "image" in sample, "Sample does not contain 'image'"
        assert "mask" in sample, "Sample does not contain 'mask'"
        assert isinstance(sample["image"], torch.Tensor), "'image' is not a torch.Tensor"
        assert isinstance(sample["mask"], torch.Tensor), "'mask' is not a torch.Tensor"
        assert sample["image"].dtype == torch.float32, "'image' does not have dtype torch.float32"
        assert sample["mask"].dtype == torch.long, "'mask' does not have dtype torch.long"
        assert sample["image"].shape == (len(dataset.bands), 64, 64), f"'image' has incorrect shape: {sample['image'].shape}"
        assert sample["mask"].shape == (64, 64), f"'mask' has incorrect shape: {sample['mask'].shape}"

    def test_plot(self, neontree_data_root):
        transform = A.Compose([ToTensorV2()])

        dataset = MNeonTreeNonGeo(data_root=neontree_data_root, split="train", transform=transform)
        sample = dataset[0]

        fig = dataset.plot(sample, suptitle="Sample Plot")
        assert isinstance(fig, plt.Figure), "Plot method did not return a plt.Figure"
        plt.close(fig)


class TestMBrickKilnNonGeo:
    def test_dataset_sample(self, brickkiln_data_root):
        transform = A.Compose([
            A.Resize(64, 64),
            ToTensorV2(),
        ])

        dataset = MBrickKilnNonGeo(data_root=brickkiln_data_root, split="train", transform=transform)
        sample = dataset[0]

        assert "image" in sample, "Sample does not contain 'image'"
        assert "label" in sample, "Sample does not contain 'label'"
        assert isinstance(sample["image"], torch.Tensor), "'image' is not a torch.Tensor"
        assert isinstance(sample["label"], int), "'label' is not an int"
        assert sample["image"].dtype == torch.float32, "'image' does not have dtype torch.float32"
        assert sample["image"].shape == (len(dataset.bands), 64, 64), f"'image' has incorrect shape: {sample['image'].shape}"

    def test_plot(self, brickkiln_data_root):
        transform = A.Compose([ToTensorV2()])

        dataset = MBrickKilnNonGeo(data_root=brickkiln_data_root, split="train", transform=transform)
        sample = dataset[0]

        fig = dataset.plot(sample, suptitle="Sample Plot")
        assert isinstance(fig, plt.Figure), "Plot method did not return a plt.Figure"
        plt.close(fig)

class TestMEuroSATNonGeo:
    def test_dataset_sample(self, eurosat_data_root):
        transform = A.Compose([
            A.Resize(64, 64),
            ToTensorV2(),
        ])

        dataset = MEuroSATNonGeo(data_root=eurosat_data_root, split="train", transform=transform)
        sample = dataset[0]

        assert "image" in sample, "Sample does not contain 'image'"
        assert "label" in sample, "Sample does not contain 'label'"
        assert isinstance(sample["image"], torch.Tensor), "'image' is not a torch.Tensor"
        assert isinstance(sample["label"], int), "'label' is not an int"
        assert sample["image"].dtype == torch.float32, "'image' does not have dtype torch.float32"
        assert sample["image"].shape == (len(dataset.bands), 64, 64), f"'image' has incorrect shape: {sample['image'].shape}"

    def test_plot(self, eurosat_data_root):
        transform = A.Compose([ToTensorV2()])

        dataset = MEuroSATNonGeo(data_root=eurosat_data_root, split="train", transform=transform)
        sample = dataset[0]

        fig = dataset.plot(sample, suptitle="Sample Plot")
        assert isinstance(fig, plt.Figure), "Plot method did not return a plt.Figure"
        plt.close(fig)

class TestFireScarsNonGeo:
    def test_dataset_length(self, fire_scars_data_root):
        dataset = FireScarsNonGeo(data_root=fire_scars_data_root, split="train")
        expected_length = 5
        actual_length = len(dataset)
        assert actual_length == expected_length, f"Expected {expected_length}, but got {actual_length}"

    def test_dataset_sample(self, fire_scars_data_root):
        dataset = FireScarsNonGeo(data_root=fire_scars_data_root, split="train")
        sample = dataset[0]

        assert "image" in sample, "Sample does not contain 'image'"
        assert "mask" in sample, "Sample does not contain 'mask'"
        assert isinstance(sample["image"], torch.Tensor), "'image' is not a torch.Tensor"
        assert isinstance(sample["mask"], torch.Tensor), "'mask' is not a torch.Tensor"
        assert sample["image"].dtype == torch.float32, "'image' does not have dtype torch.float32"
        assert sample["mask"].dtype in [torch.float32, torch.long], "'mask' does not have expected dtype (torch.float32 or torch.long)"
        assert sample["image"].ndim == 3, "'image' does not have 3 dimensions (C, H, W)"
        assert sample["mask"].ndim == 2, "'mask' does not have 2 dimensions (H, W)"

    def test_plot(self, fire_scars_data_root):
        transform = A.Compose([ToTensorV2()])

        dataset = FireScarsNonGeo(
            data_root=fire_scars_data_root,
            split="train",
            transform=transform,
        )
        sample = dataset[0]

        fig = dataset.plot(sample, suptitle="Sample Plot")
        assert isinstance(fig, plt.Figure), "The plot method did not return a plt.Figure"
        plt.close(fig)

class TestMBigEarthNonGeo:
    def test_dataset_sample(self, m_bigearth_data_root):
        transform = A.Compose([
            ToTensorV2(),
        ])

        dataset = MBigEarthNonGeo(data_root=m_bigearth_data_root, split="train", transform=transform)
        sample = dataset[0]

        assert "image" in sample, "Sample does not contain 'image'"
        assert "label" in sample, "Sample does not contain 'label'"
        assert isinstance(sample["image"], torch.Tensor), "'image' is not a torch.Tensor"
        assert isinstance(sample["label"], torch.Tensor), "'label' is not a torch.Tensor"
        assert sample["image"].dtype == torch.float32, "'image' does not have dtype torch.float32"
        assert sample["label"].dtype == torch.float32, "'label' does not have dtype torch.float32"

    def test_plot(self, m_bigearth_data_root):
        transform = A.Compose([ToTensorV2()])

        dataset = MBigEarthNonGeo(
            data_root=m_bigearth_data_root,
            split="train",
            transform=transform,
        )
        sample = dataset[0]

        fig = dataset.plot(sample, suptitle="Sample Plot")
        assert isinstance(fig, plt.Figure), "The plot method did not return a plt.Figure"
        plt.close(fig)

class TestMForestNetNonGeo:
    def test_dataset_sample(self, m_forestnet_data_root):
        transform = A.Compose([
            ToTensorV2(),
        ])

        dataset = MForestNetNonGeo(data_root=m_forestnet_data_root, split="train", transform=transform)
        sample = dataset[0]

        assert "image" in sample, "Sample does not contain 'image'"
        assert "label" in sample, "Sample does not contain 'label'"
        assert isinstance(sample["image"], torch.Tensor), "'image' is not a torch.Tensor"
        assert isinstance(sample["label"], int), "'label' is not an int"
        assert sample["image"].dtype == torch.float32, "'image' does not have dtype torch.float32"
        assert isinstance(sample["label"], int), "'label' is not an int"

    def test_plot(self, m_forestnet_data_root):
        transform = A.Compose([ToTensorV2()])

        dataset = MForestNetNonGeo(
            data_root=m_forestnet_data_root,
            split="train",
            transform=transform,
        )
        sample = dataset[0]

        fig = dataset.plot(sample, suptitle="Sample Plot")
        assert isinstance(fig, plt.Figure), "The plot method did not return a plt.Figure"
        plt.close(fig)

class TestMNzCattleNonGeo:
    def test_dataset_sample(self, mnz_cattle_data_root):
        transform = A.Compose([
            A.Resize(64, 64),
            ToTensorV2(),
        ])

        dataset = MNzCattleNonGeo(data_root=mnz_cattle_data_root, split="train", transform=transform)
        sample = dataset[0]

        assert "image" in sample, "Sample does not contain 'image'"
        assert "mask" in sample, "Sample does not contain 'mask'"
        assert isinstance(sample["image"], torch.Tensor), "'image' is not a torch.Tensor"
        assert isinstance(sample["mask"], torch.Tensor), "'mask' is not a torch.Tensor"
        assert sample["image"].dtype == torch.float32, "'image' does not have dtype torch.float32"
        assert sample["mask"].dtype == torch.long, "'mask' does not have dtype torch.long"
        assert sample["image"].shape == (3, 64, 64), f"'image' has incorrect shape: {sample['image'].shape}"
        assert sample["mask"].shape == (64, 64), f"'mask' has incorrect shape: {sample['mask'].shape}"

    def test_plot(self, mnz_cattle_data_root):
        transform = A.Compose([ToTensorV2()])

        dataset = MNzCattleNonGeo(
            data_root=mnz_cattle_data_root,
            split="train",
            transform=transform,
        )
        sample = dataset[0]

        fig = dataset.plot(sample, suptitle="Sample Plot")
        assert isinstance(fig, plt.Figure), "The plot method did not return a plt.Figure"
        plt.close(fig)

class TestMPv4gerNonGeo:
    def test_dataset_sample(self, pv4ger_data_root):
        transform = A.Compose([
            A.Resize(64, 64),
            ToTensorV2(),
        ])

        dataset = MPv4gerNonGeo(data_root=pv4ger_data_root, split="train", transform=transform, use_metadata=True)
        sample = dataset[0]
        assert "image" in sample, "Sample does not contain 'image'"
        assert "label" in sample, "Sample does not contain 'label'"
        assert "location_coords" in sample, "Sample does not contain 'location_coords'"
        assert isinstance(sample["image"], torch.Tensor), "'image' is not a torch.Tensor"
        assert isinstance(sample["label"], int), "'label' is not an int"
        assert isinstance(sample["location_coords"], torch.Tensor), "'location_coords' is not a torch.Tensor"
        assert sample["image"].dtype == torch.float32, "'image' does not have dtype torch.float32"
        assert sample["location_coords"].dtype == torch.float32, "'location_coords' does not have dtype torch.float32"
        assert sample["image"].shape == (len(dataset.bands), 64, 64), f"'image' has incorrect shape: {sample['image'].shape}"
        assert sample["location_coords"].shape == (2,), f"'location_coords' has incorrect shape: {sample['location_coords'].shape}"

    def test_plot(self, pv4ger_data_root):

        transform = A.Compose([ToTensorV2()])


        dataset = MPv4gerNonGeo(data_root=pv4ger_data_root, split="train", transform=transform)
        sample = dataset[0]


        fig = dataset.plot(sample, suptitle="Sample Plot")
        assert isinstance(fig, plt.Figure), "Plot method did not return a plt.Figure"
        plt.close(fig)


class TestMPv4gerSegNonGeo:
    def test_dataset_sample(self, pv4gerseg_data_root):

        transform = A.Compose([
            A.Resize(64, 64),
            ToTensorV2(),
        ])

        dataset = MPv4gerSegNonGeo(data_root=pv4gerseg_data_root, split="train", transform=transform, use_metadata=True)
        sample = dataset[0]

        assert "image" in sample, "Sample does not contain 'image'"
        assert "mask" in sample, "Sample does not contain 'mask'"
        assert "location_coords" in sample, "Sample does not contain 'location_coords'"
        assert isinstance(sample["image"], torch.Tensor), "'image' is not a torch.Tensor"
        assert isinstance(sample["mask"], torch.Tensor), "'mask' is not a torch.Tensor"
        assert isinstance(sample["location_coords"], torch.Tensor), "'location_coords' is not a torch.Tensor"
        assert sample["image"].dtype == torch.float32, "'image' does not have dtype torch.float32"
        assert sample["mask"].dtype == torch.long, "'mask' does not have dtype torch.long"
        assert sample["location_coords"].dtype == torch.float32, "'location_coords' does not have dtype torch.float32"
        assert sample["image"].shape == (len(dataset.bands), 64, 64), f"'image' has incorrect shape: {sample['image'].shape}"
        assert sample["mask"].shape == (64, 64), f"'mask' has incorrect shape: {sample['mask'].shape}"
        assert sample["location_coords"].shape == (2,), f"'location_coords' has incorrect shape: {sample['location_coords'].shape}"

    def test_plot(self, pv4gerseg_data_root):
        transform = A.Compose([ToTensorV2()])

        dataset = MPv4gerSegNonGeo(data_root=pv4gerseg_data_root, split="train", transform=transform)
        sample = dataset[0]

        fig = dataset.plot(sample, suptitle="Sample Plot")
        assert isinstance(fig, plt.Figure), "Plot method did not return a plt.Figure"
        plt.close(fig)

@pytest.fixture(scope="function")
def sacroptype_data_root(tmp_path):

    data_root = tmp_path / "m_sa_crop_type"
    data_directory = data_root / MSACropTypeNonGeo.data_dir
    data_directory.mkdir(parents=True, exist_ok=True)


    partition = {
        "train": ["image_0", "image_1"],
        "val": [],
        "test": []
    }
    partition_file = data_directory / MSACropTypeNonGeo.partition_file_template.format(partition="default")
    with open(partition_file, "w") as f:
        json.dump(partition, f)


    bands = [
        "COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2",
        "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "SWIR_1", "SWIR_2", "CLOUD_PROBABILITY"
    ]


    for img in ["image_0", "image_1"]:
        file_path = data_directory / f"{img}.hdf5"
        with h5py.File(file_path, "w") as h5file:
            for band in bands:
                h5file.create_dataset(band, data=np.random.rand(100, 100).astype(np.float32))

            mask = np.random.randint(0, 3, size=(100, 100), dtype=np.uint8)
            h5file.create_dataset("label", data=mask)

    return str(data_root)


class TestMSACropTypeNonGeo:
    def test_dataset_sample(self, sacroptype_data_root):

        transform = A.Compose([
            A.Resize(64, 64),
            ToTensorV2(),
        ])

        dataset = MSACropTypeNonGeo(data_root=sacroptype_data_root, split="train", transform=transform)
        sample = dataset[0]

        assert "image" in sample, "Sample does not contain 'image'"
        assert "mask" in sample, "Sample does not contain 'mask'"
        assert isinstance(sample["image"], torch.Tensor), "'image' is not a torch.Tensor"
        assert isinstance(sample["mask"], torch.Tensor), "'mask' is not a torch.Tensor"
        assert sample["image"].dtype == torch.float32, "'image' does not have dtype torch.float32"
        assert sample["mask"].dtype == torch.long, "'mask' does not have dtype torch.long"
        assert sample["image"].shape == (len(dataset.bands), 64, 64), f"'image' has incorrect shape: {sample['image'].shape}"
        assert sample["mask"].shape == (64, 64), f"'mask' has incorrect shape: {sample['mask'].shape}"

    def test_plot(self, sacroptype_data_root):
        transform = A.Compose([ToTensorV2()])

        dataset = MSACropTypeNonGeo(data_root=sacroptype_data_root, split="train", transform=transform)
        sample = dataset[0]

        fig = dataset.plot(sample, suptitle="Sample Plot")
        assert isinstance(fig, plt.Figure), "Plot method did not return a plt.Figure"
        plt.close(fig)


class TestMSo2SatNonGeo:
    def test_dataset_sample(self, so2sat_data_root):
        transform = A.Compose([
            A.Resize(64, 64),
            ToTensorV2(),
        ])

        dataset = MSo2SatNonGeo(data_root=so2sat_data_root, split="train", transform=transform)
        sample = dataset[0]

        assert "image" in sample, "Sample does not contain 'image'"
        assert "label" in sample, "Sample does not contain 'label'"
        assert isinstance(sample["image"], torch.Tensor), "'image' is not a torch.Tensor"
        assert isinstance(sample["label"], int), "'label' is not an int"
        assert sample["image"].dtype == torch.float32, "'image' does not have dtype torch.float32"
        assert sample["image"].shape == (len(dataset.bands), 64, 64), f"'image' has incorrect shape: {sample['image'].shape}"

    def test_plot(self, so2sat_data_root):
        transform = A.Compose([ToTensorV2()])

        dataset = MSo2SatNonGeo(data_root=so2sat_data_root, split="train", transform=transform)
        sample = dataset[0]

        fig = dataset.plot(sample, suptitle="Sample Plot")
        assert isinstance(fig, plt.Figure), "Plot method did not return a plt.Figure"
        plt.close(fig)

class TestMBeninSmallHolderCashewsNonGeo:
    def test_dataset_sample(self, cashews_data_root):
        transform = A.Compose([
            A.Resize(64, 64),
            ToTensorV2(),
        ])

        dataset = MBeninSmallHolderCashewsNonGeo(data_root=cashews_data_root, split="train", transform=transform, use_metadata=True)
        sample = dataset[0]

        assert "image" in sample, "Sample does not contain 'image'"
        assert "mask" in sample, "Sample does not contain 'mask'"
        assert "temporal_coords" in sample, "Sample does not contain 'temporal_coords'"
        assert isinstance(sample["image"], torch.Tensor), "'image' is not a torch.Tensor"
        assert isinstance(sample["mask"], torch.Tensor), "'mask' is not a torch.Tensor"
        assert isinstance(sample["temporal_coords"], torch.Tensor), "'temporal_coords' is not a torch.Tensor"
        assert sample["image"].dtype == torch.float32, "'image' does not have dtype torch.float32"
        assert sample["mask"].dtype == torch.long, "'mask' does not have dtype torch.long"
        assert sample["temporal_coords"].dtype == torch.float32, "'temporal_coords' does not have dtype torch.float32"
        assert sample["image"].shape == (len(dataset.bands), 64, 64), f"'image' has incorrect shape: {sample['image'].shape}"
        assert sample["mask"].shape == (64, 64), f"'mask' has incorrect shape: {sample['mask'].shape}"
        assert sample["temporal_coords"].shape == (1, 2), f"'temporal_coords' has incorrect shape: {sample['temporal_coords'].shape}"

    def test_plot(self, cashews_data_root):
        transform = A.Compose([ToTensorV2()])

        dataset = MBeninSmallHolderCashewsNonGeo(data_root=cashews_data_root, split="train", transform=transform)
        sample = dataset[0]

        fig = dataset.plot(sample, suptitle="Sample Plot")
        assert isinstance(fig, plt.Figure), "Plot method did not return a plt.Figure"
        plt.close(fig)


class TestMChesapeakeLandcoverNonGeo:
    def test_dataset_sample(self, chesapeake_data_root):

        transform = A.Compose([
            A.Resize(64, 64),
            ToTensorV2(),
        ])

        dataset = MChesapeakeLandcoverNonGeo(data_root=chesapeake_data_root, split="train", transform=transform)
        sample = dataset[0]

        assert "image" in sample, "Sample does not contain 'image'"
        assert "mask" in sample, "Sample does not contain 'mask'"
        assert isinstance(sample["image"], torch.Tensor), "'image' is not a torch.Tensor"
        assert isinstance(sample["mask"], torch.Tensor), "'mask' is not a torch.Tensor"
        assert sample["image"].dtype == torch.float32, "'image' does not have dtype torch.float32"
        assert sample["mask"].dtype == torch.long, "'mask' does not have dtype torch.long"
        assert sample["image"].shape == (len(dataset.bands), 64, 64), f"'image' has incorrect shape: {sample['image'].shape}"
        assert sample["mask"].shape == (64, 64), f"'mask' has incorrect shape: {sample['mask'].shape}"

    def test_plot(self, chesapeake_data_root):
        transform = A.Compose([ToTensorV2()])

        dataset = MChesapeakeLandcoverNonGeo(data_root=chesapeake_data_root, split="train", transform=transform)
        sample = dataset[0]

        fig = dataset.plot(sample, suptitle="Sample Plot")
        assert isinstance(fig, plt.Figure), "Plot method did not return a plt.Figure"
        plt.close(fig)

class TestSen1Floods11NonGeo:
    def test_dataset_sample(self, sen1floods_data_root):
        transform = A.Compose([
            A.Resize(32, 32),
            ToTensorV2(),
        ])

        dataset = Sen1Floods11NonGeo(data_root=sen1floods_data_root, split="train", transform=transform, use_metadata=True)
        sample = dataset[0]

        assert "image" in sample, "Sample does not contain 'image'"
        assert "mask" in sample, "Sample does not contain 'mask'"
        assert "location_coords" in sample, "Sample does not contain 'location_coords'"
        assert "temporal_coords" in sample, "Sample does not contain 'temporal_coords'"
        assert isinstance(sample["image"], torch.Tensor), "'image' is not a torch.Tensor"
        assert isinstance(sample["mask"], torch.Tensor), "'mask' is not a torch.Tensor"
        assert isinstance(sample["location_coords"], torch.Tensor), "'location_coords' is not a torch.Tensor"
        assert isinstance(sample["temporal_coords"], torch.Tensor), "'temporal_coords' is not a torch.Tensor"
        assert sample["image"].dtype == torch.float32, "'image' does not have dtype torch.float32"
        assert sample["mask"].dtype == torch.long, "'mask' does not have dtype torch.long"
        assert sample["location_coords"].dtype == torch.float32, "'location_coords' does not have dtype torch.float32"
        assert sample["temporal_coords"].dtype == torch.float32, "'temporal_coords' does not have dtype torch.float32"
        num_bands = len(dataset.bands)
        assert sample["image"].shape == (num_bands, 32, 32), f"'image' has incorrect shape: {sample['image'].shape}"
        assert sample["mask"].shape == (32, 32), f"'mask' has incorrect shape: {sample['mask'].shape}"

    def test_plot(self, sen1floods_data_root):
        transform = A.Compose([ToTensorV2()])

        dataset = Sen1Floods11NonGeo(data_root=sen1floods_data_root, split="train", transform=transform)
        sample = dataset[0]

        fig = dataset.plot(sample, suptitle="Sample Plot")
        assert isinstance(fig, plt.Figure), "The plot method did not return a plt.Figure"
        plt.close(fig)

class TestMultiTemporalCropClassification:
    def test_dataset_sample(self, crop_classification_data_root):
        transform = A.Compose(
            [
                FlattenTemporalIntoChannels(),
                ToTensorV2(),
                UnflattenTemporalFromChannels(3),
            ],
            is_check_shapes=False
        )

        dataset = MultiTemporalCropClassification(
            data_root=crop_classification_data_root,
            split="train",
            transform=transform,
            use_metadata=True,
        )
        sample = dataset[0]

        assert "image" in sample, "Sample does not contain 'image'"
        assert "mask" in sample, "Sample does not contain 'mask'"
        assert isinstance(sample["image"], torch.Tensor), "'image' is not a torch.Tensor"
        assert isinstance(sample["mask"], torch.Tensor), "'mask' is not a torch.Tensor"
        assert sample["image"].dtype == torch.float32, "'image' does not have dtype torch.float32"
        assert sample["mask"].dtype == torch.long, "'mask' does not have dtype torch.long"
        assert sample["image"].shape == (6, 3, 64, 64), f"'image' has incorrect shape: {sample['image'].shape}"
        assert sample["mask"].shape == (64, 64), f"'mask' has incorrect shape: {sample['mask'].shape}"

    def test_plot(self, crop_classification_data_root):
        transform = A.Compose(
            [
                FlattenTemporalIntoChannels(),
                ToTensorV2(),
                UnflattenTemporalFromChannels(3),
            ],
            is_check_shapes=False
        )

        dataset = MultiTemporalCropClassification(
            data_root=crop_classification_data_root,
            split="train",
            transform=transform,
        )
        sample = dataset[0]

        fig = dataset.plot(sample, suptitle="Sample Plot")
        assert isinstance(fig, plt.Figure), "The plot method did not return a plt.Figure"
        plt.close(fig)

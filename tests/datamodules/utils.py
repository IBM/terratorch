import os
import pickle

import h5py
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin


def create_dummy_tiff(path: str, shape: tuple, pixel_values: int | list, min_size: int = None) -> None:

    if type(pixel_values) == int:
        pixel_values_ = [pixel_values]
    else:
        pixel_values_ = pixel_values

    if not all([type(i)==int for i in pixel_values_]):
        dtype = np.float32
    else:
        dtype = np.uint8

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if min_size is not None:
        if len(shape) == 3:
            h, w, c = shape
            h = max(h, min_size)
            w = max(w, min_size)
            shape = (h, w, c)
        elif len(shape) == 2:
            h, w = shape
            h = max(h, min_size)
            w = max(w, min_size)
            shape = (h, w)
    if len(shape) == 3:
        h, w, c = shape
        data = np.random.choice(pixel_values, size=(h, w, c), replace=True).astype(dtype)
        data = np.transpose(data, (2, 0, 1))
    elif len(shape) == 2:
        data = np.random.choice(pixel_values, size=shape, replace=True).astype(dtype)
        data = data[np.newaxis, ...]
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        transform=transform
    ) as dst:
        dst.write(data)

def create_dummy_image(path: str, shape: tuple, pixel_values: list[int]) -> None:

    if not all([type(i)==int for i in pixel_values]):
        dtype = np.float32
    else:
        dtype = np.uint8

    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if ext in [".tif", ".tiff"]:
        create_dummy_tiff(path, shape, pixel_values)
    else:
        if len(shape) == 3:
            data = np.random.choice(pixel_values, size=shape, replace=True).astype(dtype)
            img = Image.fromarray(data)
        elif len(shape) == 2:
            data = np.random.choice(pixel_values, size=shape, replace=True).astype(dtype)
            img = Image.fromarray(data)
        else:
            msg = "Invalid shape"
            raise ValueError(msg)
        img.save(path)

def create_dummy_h5_pickle(path: str, keys: list[str], shape: tuple, label: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        for key in keys:
            data = np.full(shape, fill_value=1, dtype=np.float32)
            f.create_dataset(key, data=data)
        attr = pickle.dumps({"label": label})
        f.attrs["pickle"] = repr(attr)

def create_dummy_h5(path: str, keys: list[str], shape: tuple, label: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        for key in keys:
            data = np.full(shape, fill_value=1, dtype=np.float32)
            f.create_dataset(key, data=data)
        f.create_dataset("label", data=np.full(shape, fill_value=label, dtype=np.int32))

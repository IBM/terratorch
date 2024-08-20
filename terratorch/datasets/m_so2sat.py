import ast
import json
import pickle
from collections.abc import Sequence
from pathlib import Path

import albumentations as A
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import to_tensor


class MSo2SatNonGeo(NonGeoDataset):
    all_band_names = (
        "VH_REAL",
        "BLUE",
        "VH_IMAGINARY",
        "GREEN",
        "VV_REAL",
        "RED",
        "VV_IMAGINARY",
        "VH_LEE_FILTERED",
        "RED_EDGE_1",
        "VV_LEE_FILTERED",
        "RED_EDGE_2",
        "VH_LEE_FILTERED_REAL",
        "RED_EDGE_3",
        "NIR_BROAD",
        "VV_LEE_FILTERED_IMAGINARY",
        "NIR_NARROW",
        "SWIR_1",
        "SWIR_2",
    )

    rgb_bands = ("RED", "GREEN", "BLUE")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    def __init__(
        self,
        data_root: str,
        bands: Sequence[str] = BAND_SETS["all"],
        transform: A.Compose | None = None,
        split="train",
        partition="default",
    ) -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            msg = "Split must be one of train, test, val."
            raise Exception(msg)
        if split == "val":
            split = "valid"

        self.transform = transform if transform else lambda **batch: to_tensor(batch)
        self._validate_bands(bands)
        self.bands = bands
        self.band_indices = np.array([self.all_band_names.index(b) for b in bands if b in self.all_band_names])
        self.split = split
        data_root = Path(data_root)
        self.data_directory = data_root / "m-so2sat"

        partition_file = self.data_directory / f"{partition}_partition.json"
        with open(partition_file, "r") as file:
            partitions = json.load(file)

        if split not in partitions:
            raise ValueError(f"Split '{split}' not found.")

        self.image_files = [self.data_directory / (filename + ".hdf5") for filename in partitions[split]]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_path = self.image_files[index]
        image_id = file_path.stem

        with h5py.File(file_path, "r") as h5file:
            keys = sorted(h5file.keys())
            keys = np.array([key for key in keys if key != "label"])[self.band_indices]
            bands = [np.array(h5file[key]) for key in keys]

            image = np.stack(bands, axis=-1)
            attr_dict = pickle.loads(ast.literal_eval(h5file.attrs["pickle"]))
            class_index = attr_dict["label"]

        output = {"image": image.astype(np.float32)}

        output = self.transform(**output)

        output["label"] = class_index

        return output

    def _validate_bands(self, bands: Sequence[str]) -> None:
        assert isinstance(bands, Sequence), "'bands' must be a sequence"
        for band in bands:
            if band not in self.all_band_names:
                raise ValueError(f"'{band}' is an invalid band name.")

    def __len__(self):
        return len(self.image_files)

    def plot(self, arg, suptitle: str | None = None) -> None:
        if isinstance(arg, int):
            sample = self.__getitem__(arg)
        elif isinstance(arg, dict):
            sample = arg
        else:
            raise TypeError("Argument must be an integer index or a sample dictionary.")

        image = sample["image"].numpy()
        label_index = sample["label"].numpy()

        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")

        rgb_image = image[rgb_indices, :, :]
        rgb_image = np.transpose(rgb_image, (1, 2, 0))
        rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))

        MSo2SatNonGeo._plot_sample(image=rgb_image, label_index=label_index, suptitle=suptitle)

    @staticmethod
    def _plot_sample(image, label_index, suptitle=None) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        ax.axis("off")

        class_name = str(label_index)
        title = f"Class: {class_name}"
        if suptitle:
            title = f"{suptitle} - {title}"
        ax.set_title(title)

        return fig

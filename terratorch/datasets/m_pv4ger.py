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


class MPv4gerNonGeo(NonGeoDataset):
    all_band_names = ("BLUE", "GREEN", "RED")

    rgb_bands = ("RED", "GREEN", "BLUE")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}  # noqa: RUF012

    def __init__(
        self,
        data_root: str,
        bands: Sequence[str] = BAND_SETS["all"],
        transform: A.Compose | None = None,
        split="train",
        partition="default",
        use_metadata=False,  # noqa: FBT002
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
        self.use_metadata = use_metadata
        data_root = Path(data_root)
        self.data_directory = data_root / "m-pv4ger"

        partition_file = self.data_directory / f"{partition}_partition.json"
        with open(partition_file) as file:
            partitions = json.load(file)

        if split not in partitions:
            msg = f"Split '{split}' not found."
            raise ValueError(msg)

        self.image_files = [self.data_directory / (filename + ".hdf5") for filename in partitions[split]]

    def _get_coords(self, image_id: str) -> torch.Tensor:
        lat_str, lon_str = image_id.split(",")
        latitude = float(lat_str)
        longitude = float(lon_str)

        location_coords = np.array([latitude, longitude], dtype=np.float32)
        return torch.tensor(location_coords)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_path = self.image_files[index]
        image_id = file_path.stem

        with h5py.File(file_path, "r") as h5file:
            keys = sorted(h5file.keys())
            keys = np.array([key for key in keys if key != "label"])[self.band_indices]
            bands = [np.array(h5file[key]) for key in keys]

            image = np.stack(bands, axis=-1)
            attr_dict = pickle.loads(ast.literal_eval(h5file.attrs["pickle"]))  # noqa: S301
            class_index = attr_dict["label"]

        output = {"image": image.astype(np.float32)}

        output = self.transform(**output)

        output["label"] = class_index

        if self.use_metadata:
            output["location_coords"] = self._get_coords(image_id)

        return output

    def _validate_bands(self, bands: Sequence[str]) -> None:
        assert isinstance(bands, Sequence), "'bands' must be a sequence"  # noqa: S101
        for band in bands:
            if band not in self.all_band_names:
                msg = f"'{band}' is an invalid band name."
                raise ValueError(msg)

    def __len__(self):
        return len(self.image_files)

    def plot(self, arg, suptitle: str | None = None) -> None:
        if isinstance(arg, int):
            sample = self.__getitem__(arg)
        elif isinstance(arg, dict):
            sample = arg
        else:
            msg = "Argument must be an integer index or a sample dictionary."
            raise TypeError(msg)

        image = sample["image"].numpy()
        label_index = sample["label"].numpy()

        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                msg = "Dataset doesn't contain some of the RGB bands"
                raise ValueError(msg)

        rgb_image = image[rgb_indices, :, :]
        rgb_image = np.transpose(rgb_image, (1, 2, 0))
        rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))

        self._plot_sample(image=rgb_image, label_index=label_index, suptitle=suptitle)

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

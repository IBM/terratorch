import json
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import albumentations as A
import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from matplotlib import colormaps
from matplotlib.colors import Normalize
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import to_tensor


class MNzCattleNonGeo(NonGeoDataset):
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
        self.data_directory = data_root / "m-nz-cattle"

        partition_file = self.data_directory / f"{partition}_partition.json"
        with open(partition_file) as file:
            partitions = json.load(file)

        if split not in partitions:
            msg = f"Split '{split}' not found."
            raise ValueError(msg)

        self.image_files = [self.data_directory / (filename + ".hdf5") for filename in partitions[split]]

    def _get_coords(self, file_name: str) -> torch.Tensor:
        coords_str = re.search(r"_(\-?\d+\.\d+),(\-?\d+\.\d+)", file_name).groups()
        longitude, latitude = map(float, coords_str)

        return torch.tensor([latitude, longitude], dtype=torch.float32)

    def _get_date(self, band_name: str) -> torch.Tensor:
        date_str = band_name.split("_")[-1]
        date = pd.to_datetime(date_str, format="%Y-%m-%d")  # noqa: DTZ007

        return torch.tensor([[date.year, date.dayofyear - 1]], dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_path = self.image_files[index]
        file_name = file_path.stem

        with h5py.File(file_path, "r") as h5file:
            keys = sorted(h5file.keys())

            data_keys = [key for key in keys if "label" not in key]
            label_keys = [key for key in keys if "label" in key]

            temporal_coords = self._get_date(data_keys[0])

            bands = [np.array(h5file[key]) for key in data_keys]
            image = np.stack(bands, axis=-1)

            mask = np.array(h5file[label_keys[0]])
            image = np.stack(bands, axis=-1)

        output = {"image": image.astype(np.float32), "mask": mask}

        output = self.transform(**output)
        output["mask"] = output["mask"].long()

        if self.use_metadata:
            location_coords = self._get_coords(file_name)
            output["location_coords"] = location_coords
            output["temporal_coords"] = temporal_coords

        return output

    def _validate_bands(self, bands: Sequence[str]) -> None:
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

        showing_predictions = sample["prediction"] if "prediction" in sample else None

        self.plot_sample(sample, showing_predictions, suptitle)

    def plot_sample(self, sample, prediction=None, suptitle: str | None = None, class_names=None):
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                msg = "Dataset doesn't contain some of the RGB bands"
                raise ValueError(msg)

        image = sample["image"].numpy()
        image = image[rgb_indices, :, :]
        image = np.transpose(image, (1, 2, 0))
        image = (image - image.min(axis=(0, 1))) * (1 / image.max(axis=(0, 1)))
        image = np.clip(image, 0, 1)

        mask = sample["mask"].numpy()
        num_classes = len(np.unique(mask))

        num_images = 4 if prediction is not None else 3
        fig, ax = plt.subplots(1, num_images, figsize=(num_images * 4, 4), tight_layout=True)

        cmap = colormaps["jet"]
        norm = Normalize(vmin=0, vmax=num_classes - 1)

        ax[0].imshow(image)
        ax[0].set_title("Image")
        ax[0].axis("off")

        ax[1].imshow(mask, cmap=cmap, norm=norm)
        ax[1].set_title("Ground Truth Mask")
        ax[1].axis("off")

        ax[2].imshow(image)
        ax[2].imshow(mask, cmap=cmap, alpha=0.3, norm=norm)
        ax[2].set_title("GT Mask on Image")
        ax[2].axis("off")

        if prediction is not None:
            prediction = prediction.numpy()
            ax[3].imshow(prediction, cmap=cmap, norm=norm)
            ax[3].set_title("Predicted Mask")
            ax[3].axis("off")

        if class_names:
            legend_handles = [mpatches.Patch(color=cmap(i), label=class_names[i]) for i in range(num_classes)]
            ax[0].legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc="upper left")

        if suptitle:
            plt.suptitle(suptitle)

        return fig

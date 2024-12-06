import glob
import os
from pathlib import Path
from typing import Any

import albumentations as A
import geopandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import torch
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch import Tensor
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import default_transform, filter_valid_files


class Sen1Floods11NonGeo(NonGeoDataset):
    """NonGeo dataset implementation for sen1floods11."""
    all_band_names = (
                "COASTAL_AEROSOL",
                "BLUE",
                "GREEN",
                "RED",
                "RED_EDGE_1",
                "RED_EDGE_2",
                "RED_EDGE_3",
                "NIR_BROAD",
                "NIR_NARROW",
                "WATER_VAPOR",
                "CIRRUS",
                "SWIR_1",
                "SWIR_2",
            )


    def __init__(
        self,
        data_root: str,
        split="train",
        bands: None | list[int] = None,
        transform: A.Compose | None = None,
    ) -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            msg = "Split must be one of train, test, val."
            raise Exception(msg)
        if split == "val":
            split = "valid"

        self.bands = bands
        data_root = Path(data_root)
        data_directory = data_root / "data/data/flood_events/HandLabeled/"
        input_directory = data_directory / "S2Hand"
        label_directory = data_directory / "LabelHand"

        # split_file = data_root / f"splits/splits/flood_handlabeled/flood_bolivia_data_S2.txt"
        split_file = data_root / f"splits/splits/flood_handlabeled/flood_{split}_data_S2_geodn.txt"
        # split_file = data_root / f"splits/splits/flood_handlabeled/flood_bolivia_data_S2.txt"
        metadata_file = data_root / "Sen1Floods11_Metadata.geojson"
        self.metadata = geopandas.read_file(metadata_file)

        self.image_files = sorted(glob.glob(os.path.join(input_directory, "*.tif")))
        self.segmentation_mask_files = sorted(glob.glob(os.path.join(label_directory, "*.tif")))

        with open(split_file) as f:
            split = f.readlines()
        valid_files = {rf"{substring.strip()}" for substring in split}
        self.image_files = filter_valid_files(
            self.image_files,
            valid_files=valid_files,
            ignore_extensions=True,
            allow_substring=True,
        )
        self.segmentation_mask_files = filter_valid_files(
            self.segmentation_mask_files,
            valid_files=valid_files,
            ignore_extensions=True,
            allow_substring=True,
        )

        self.rgb_indices = [2, 1, 0]
        self.transform = transform if transform else default_transform

    def __len__(self) -> int:
        return len(self.image_files)

    def _get_date(self, index) -> np.ndarray:
        # move this logic to the model?
        file_name = self.image_files[index]
        location = os.path.basename(file_name).split("_")[0]
        if self.metadata[self.metadata["location"] == location].shape[0] != 1:
            # msg = f"No date found for sample {file_name}"
            # raise Exception(msg)
            date = pd.to_datetime("13-10-1998", dayfirst=True)
        else:
            date = pd.to_datetime(self.metadata[self.metadata["location"] == location]["s2_date"].item())
        date_np = np.zeros((1, 3))
        date_np[0, 0] = date.year
        date_np[0, 1] = date.dayofyear - 1  # base 0
        # date_np[0, 2] = date.hour
        return date_np

    def _get_coords(self, index) -> np.ndarray:
        file_name = self.image_files[index]
        image = rioxarray.open_rasterio(file_name)
        # lons_lats = np.meshgrid(image.x / 180, image.y / 90)
        # coords = np.stack([np.stack(lon_lat) for lon_lat in lons_lats])
        # coords shape: batch_size, 2 (lon, lat), height, width

        lat_lon = np.array([image.y[image.shape[0]//2], image.x[image.shape[1]//2]])
        return lat_lon

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = self._load_file(self.image_files[index]).astype(np.float32) * 0.0001
        if self.bands:
            image = image[self.bands, ...]
        image = np.moveaxis(image, 0, -1)
        output = {
            "image": image,
            "mask": self._load_file(self.segmentation_mask_files[index]).astype(np.int64)[0],
        }
        if self.transform:
            output = self.transform(**output)

        output["location_coords"] = np.moveaxis(self._get_coords(index).astype(np.float32), 0, -1)
        output["temporal_coords"] = self._get_date(index).astype(np.float32)
        return output

    def _load_file(self, path: Path):
        data = rioxarray.open_rasterio(path)
        return data.to_numpy()

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample["image"]
        if torch.is_tensor(image):
            image = image.numpy()
        image = image.take(self.rgb_indices, axis=0)
        image = np.transpose(image, (1, 2, 0))
        image = (image - image.min(axis=(0, 1))) * (1 / image.max(axis=(0, 1)))
        image = np.clip(image, 0, 1)

        label_mask = sample["mask"]
        label_mask = np.transpose(label_mask, (1, 2, 0))

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_mask = sample["prediction"]

        return self._plot_sample(
            image,
            label_mask,
            prediction=prediction_mask if showing_predictions else None,
            suptitle=suptitle,
        )

    @staticmethod
    def _plot_sample(image, label, num_classes, prediction=None, suptitle=None, class_names=None):
        num_images = 5 if prediction else 4
        fig, ax = plt.subplots(1, num_images, figsize=(8, 6))

        # for legend
        ax[0].axis("off")

        norm = mpl.colors.Normalize(vmin=0, vmax=num_classes - 1)
        ax[1].axis("off")
        ax[1].title.set_text("Image")
        ax[1].imshow(image)

        ax[2].axis("off")
        ax[2].title.set_text("Ground Truth Mask")
        ax[2].imshow(label, cmap="jet", norm=norm)

        ax[3].axis("off")
        ax[3].title.set_text("GT Mask on Image")
        ax[3].imshow(image)
        ax[3].imshow(label, cmap="jet", alpha=0.3, norm=norm)

        if prediction:
            ax[4].title.set_text("Predicted Mask")
            ax[4].imshow(prediction, cmap="jet", norm=norm)

        cmap = cm.get_cmap("jet")
        legend_data = []
        for i, _ in enumerate(range(num_classes)):
            class_name = class_names[i] if class_names else str(i)
            data = [i, cmap(norm(i)), class_name]
            legend_data.append(data)
        handles = [Rectangle((0, 0), 1, 1, color=tuple(v for v in c)) for k, c, n in legend_data]
        labels = [n for k, c, n in legend_data]
        ax[0].legend(handles, labels, loc="center")
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig

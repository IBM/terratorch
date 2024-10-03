# Copyright contributors to the Terratorch project

import dataclasses
import glob
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import torch
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch import Tensor
from torchgeo.datasets import NonGeoDataset, RasterDataset


class FireScarsNonGeo(NonGeoDataset):
    """NonGeo dataset implementation for fire scars."""

    def __init__(self, transform, bands, data_root: Path, use_metadata: bool) -> None:  # noqa: FBT001
        super().__init__()
        self.data_root = data_root
        self.bands = bands
        self.image_files = sorted(glob.glob(os.path.join(data_root, "subsetted*_merged.tif")))
        self.segmentation_mask_files = sorted(glob.glob(os.path.join(data_root, "subsetted*.mask.tif")))
        self.rgb_indices = [0, 1, 2]
        self.use_metadata = use_metadata
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def _get_date(self, filename: str) -> torch.Tensor:
        base_filename = os.path.basename(filename)

        filename_regex = r"subsetted_512x512_HLS\.S30\.T[0-9A-Z]{5}\.(?P<date>[0-9]+)\.v1\.4_merged\.tif"
        match = re.match(filename_regex, base_filename)
        date_str = match.group("date")
        year = int(date_str[:4])
        julian_day = int(date_str[4:])

        temporal_coords = torch.tensor([year, julian_day], dtype=torch.float32).reshape(1, 2)
        return temporal_coords

    def __getitem__(self, index: int) -> dict[str, Any]:
        output = {
            "image": self._load_file(self.image_files[index]).astype(np.float32).transpose(1, 2, 0),
            "mask": self._load_file(self.segmentation_mask_files[index]).astype(np.int64).squeeze(),
        }
        if self.bands:
            output["image"] =  output["image"][..., self.bands]
        output = self.transform(**output)
        output["mask"] = output["mask"].long()
        if self.use_metadata:
            output["temporal_coords"] = self._get_date(self.image_files[index])

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
        image = sample["image"].take(self.rgb_indices, axis=0)
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

        cmap = plt.get_cmap("jet")
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


class FireScarsHLS(RasterDataset):
    """RasterDataset implementation for fire scars input images."""

    filename_glob = "subsetted*_merged.tif"
    filename_regex = r"subsetted_512x512_HLS\..30\..{6}\.(?P<date>[0-9]*)\.v1.4_merged.tif"
    date_format = "%Y%j"
    is_image = True
    separate_files = False
    all_bands = dataclasses.field(default_factory=["B02", "B03", "B04", "B8A", "B11", "B12"])
    rgb_bands = dataclasses.field(default_factory=["B04", "B03", "B02"])


class FireScarsSegmentationMask(RasterDataset):
    """RasterDataset implementation for fire scars segmentation mask.
    Can be easily merged with input images using the & operator.
    """

    filename_glob = "subsetted*.mask.tif"
    filename_regex = r"subsetted_512x512_HLS\..30\..{6}\.(?P<date>[0-9]*)\.v1.4.mask.tif"
    date_format = "%Y%j"
    is_image = False
    separate_files = False

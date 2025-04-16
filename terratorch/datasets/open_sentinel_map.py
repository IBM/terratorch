import json
import random
import re
from pathlib import Path

import albumentations as A  # noqa: N812
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import pad_numpy, to_tensor

MAX_TEMPORAL_IMAGE_SIZE = (192, 192)


class OpenSentinelMap(NonGeoDataset):
    """
    Pytorch Dataset class to load samples from the [OpenSentinelMap](https://visionsystemsinc.github.io/open-sentinel-map/) dataset, supporting
    multiple bands and temporal sampling strategies.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        bands: list[str] | None = None,
        transform: A.Compose | None = None,
        spatial_interpolate_and_stack_temporally: bool = True,  # noqa: FBT001, FBT002
        pad_image: int | None = None,
        truncate_image: int | None = None,
        target: int = 0,
        pick_random_pair: bool = True,  # noqa: FBT002, FBT001
    ) -> None:
        """

        Args:
            data_root (str): Path to the root directory of the dataset.
            split (str): Dataset split to load. Options are 'train', 'val', or 'test'. Defaults to 'train'.
            bands (list of str, optional): List of band names to load. Defaults to ['gsd_10', 'gsd_20', 'gsd_60'].
            transform (albumentations.Compose, optional): Albumentations transformations to apply to the data.
            spatial_interpolate_and_stack_temporally (bool): If True, the bands are interpolated and concatenated over time.
                Default is True.
            pad_image (int, optional): Number of timesteps to pad the time dimension of the image.
                If None, no padding is applied.
            truncate_image (int, optional): Number of timesteps to truncate the time dimension of the image.
                If None, no truncation is performed.
            target (int): Specifies which target class to use from the mask. Default is 0.
            pick_random_pair (bool): If True, selects two random images from the temporal sequence. Default is True.
        """
        split = "test"
        if bands is None:
            bands = ["gsd_10", "gsd_20", "gsd_60"]

        allowed_bands = {"gsd_10", "gsd_20", "gsd_60"}
        for band in bands:
            if band not in allowed_bands:
                msg = f"Band '{band}' is not recognized. Available values are: {', '.join(allowed_bands)}"
                raise ValueError(msg)

        if split not in ["train", "val", "test"]:
            msg = f"Split '{split}' not recognized. Use 'train', 'val', or 'test'."
            raise ValueError(msg)

        self.data_root = Path(data_root)
        split_mapping = {"train": "training", "val": "validation", "test": "testing"}
        split = split_mapping[split]
        self.imagery_root = self.data_root / "osm_sentinel_imagery"
        self.label_root = self.data_root / "osm_label_images_v10"
        self.auxiliary_data = pd.read_csv(self.data_root / "spatial_cell_info.csv")
        self.auxiliary_data = self.auxiliary_data[self.auxiliary_data["split"] == split]
        self.bands = bands
        self.transform = transform if transform else lambda **batch: to_tensor(batch)
        self.label_mappings = self._load_label_mappings()
        self.split_data = self.auxiliary_data[self.auxiliary_data["split"] == split]
        self.spatial_interpolate_and_stack_temporally = spatial_interpolate_and_stack_temporally
        self.pad_image = pad_image
        self.truncate_image = truncate_image
        self.target = target
        self.pick_random_pair = pick_random_pair

        self.image_files = []
        self.label_files = []

        for _, row in self.split_data.iterrows():
            mgrs_tile = row["MGRS_tile"]
            spatial_cell = str(row["cell_id"])

            label_file = self.label_root / mgrs_tile / f"{spatial_cell}.png"

            if label_file.exists():
                self.image_files.append((mgrs_tile, spatial_cell))
                self.label_files.append(label_file)

    def _load_label_mappings(self):
        with open(self.data_root / "osm_categories.json") as f:
            return json.load(f)

    def _extract_date_from_filename(self, filename: str) -> str:
        match = re.search(r"(\d{8})", filename)
        if match:
            return match.group(1)
        else:
            msg = f"Date not found in filename {filename}"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.image_files)

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None, show_axes: bool | None = False) -> Figure:
        if "gsd_10" not in self.bands:
            return None

        num_images = len([key for key in sample if key.startswith("image")])
        images = []

        for i in range(1, num_images + 1):
            image_dict = sample[f"image{i}"]
            image = image_dict["gsd_10"]
            if isinstance(image, Tensor):
                image = image.numpy()

            image = image.take(range(3), axis=2)
            image = image.squeeze()
            image = (image - image.min(axis=(0, 1))) * (1 / image.max(axis=(0, 1)))
            image = np.clip(image, 0, 1)
            images.append(image)

        label_mask = sample["mask"]
        if isinstance(label_mask, Tensor):
            label_mask = label_mask.numpy()

        return self._plot_sample(images, label_mask, suptitle=suptitle, show_axes=show_axes)

    def _plot_sample(
        self,
        images: list[np.ndarray],
        label: np.ndarray,
        suptitle: str | None = None,
        show_axes: bool = False,
    ) -> Figure:
        num_images = len(images)
        fig, ax = plt.subplots(1, num_images + 1, figsize=(15, 5))
        axes_visibility = "on" if show_axes else "off"

        for i, image in enumerate(images):
            ax[i].imshow(image)
            ax[i].set_title(f"Image {i + 1}")
            ax[i].axis(axes_visibility)

        ax[-1].imshow(label, cmap="gray")
        ax[-1].set_title("Ground Truth Mask")
        ax[-1].axis(axes_visibility)

        if suptitle:
            plt.suptitle(suptitle)

        return fig

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        mgrs_tile, spatial_cell = self.image_files[index]
        spatial_cell_path = self.imagery_root / mgrs_tile / spatial_cell

        npz_files = list(spatial_cell_path.glob("*.npz"))
        npz_files.sort(key=lambda x: self._extract_date_from_filename(x.stem))

        if self.pick_random_pair:
            npz_files = random.sample(npz_files, 2)
            npz_files.sort(key=lambda x: self._extract_date_from_filename(x.stem))

        output = {}

        if self.spatial_interpolate_and_stack_temporally:
            images_over_time = []
            for _, npz_file in enumerate(npz_files):
                data = np.load(npz_file)
                interpolated_bands = []
                for band in self.bands:
                    band_frame = data[band]
                    band_frame = torch.from_numpy(band_frame).float()
                    band_frame = band_frame.permute(2, 0, 1)
                    interpolated = F.interpolate(
                        band_frame.unsqueeze(0), size=MAX_TEMPORAL_IMAGE_SIZE, mode="bilinear", align_corners=False
                    ).squeeze(0)
                    interpolated_bands.append(interpolated)
                concatenated_bands = torch.cat(interpolated_bands, dim=0)
                images_over_time.append(concatenated_bands)

            images = torch.stack(images_over_time, dim=0).numpy()
            if self.truncate_image:
                images = images[-self.truncate_image :]
            if self.pad_image:
                images = pad_numpy(images, self.pad_image)

            output["image"] = images.transpose(0, 2, 3, 1)
        else:
            image_dict = {band: [] for band in self.bands}
            for _, npz_file in enumerate(npz_files):
                data = np.load(npz_file)
                for band in self.bands:
                    band_frames = data[band]
                    band_frames = band_frames.astype(np.float32)
                    band_frames = np.transpose(band_frames, (2, 0, 1))
                    image_dict[band].append(band_frames)

            final_image_dict = {}
            for band in self.bands:
                band_images = image_dict[band]
                if self.truncate_image:
                    band_images = band_images[-self.truncate_image :]
                if self.pad_image:
                    band_images = [pad_numpy(img, self.pad_image) for img in band_images]
                band_images = np.stack(band_images, axis=0)
                final_image_dict[band] = band_images

            output["image"] = final_image_dict

        label_file = self.label_files[index]
        mask = np.array(Image.open(label_file)).astype(int)

        # Map 'unlabel' (254) and 'none' (255) to unused classes 15 and 16 for processing
        mask[mask == 254] = 15  # noqa: PLR2004
        mask[mask == 255] = 16  # noqa: PLR2004
        output["mask"] = mask[:, :, self.target]

        if self.transform:
            output = self.transform(**output)

        return output

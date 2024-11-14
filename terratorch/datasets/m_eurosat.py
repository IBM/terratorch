import json
from collections.abc import Sequence
from pathlib import Path

import albumentations as A
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import (
    clip_image,
    default_transform,
    validate_bands,
)


class MEuroSATNonGeo(NonGeoDataset):
    """NonGeo dataset implementation for M-EuroSAT."""
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

    rgb_bands = ("RED", "GREEN", "BLUE")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    splits = {"train": "train", "val": "valid", "test": "test"}

    data_dir = "m-eurosat"
    partition_file_template = "{partition}_partition.json"
    label_map_file = "label_map.json"

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transform: A.Compose | None = None,
        partition: str = "default",
    ) -> None:
        """Initialize the dataset.

        Args:
            data_root (str): Path to the data root directory.
            split (str): One of 'train', 'val', or 'test'.
            bands (Sequence[str]): Bands to be used. Defaults to all bands.
            transform (A.Compose | None): Albumentations transform to be applied.
                Defaults to None, which applies default_transform().
            partition (str): Partition name for the dataset splits. Defaults to 'default'.
        """
        super().__init__()

        if split not in self.splits:
            msg = f"Incorrect split '{split}', please choose one of {list(self.splits.keys())}."
            raise ValueError(msg)
        split_name = self.splits[split]
        self.split = split

        validate_bands(bands, self.all_band_names)
        self.bands = bands
        self.band_indices = [self.all_band_names.index(b) for b in bands]\

        self.data_root = Path(data_root)
        self.data_directory = self.data_root / self.data_dir

        label_map_path = self.data_directory / self.label_map_file
        with open(label_map_path) as file:
            self.label_map = json.load(file)

        self.id_to_class = {img_id: cls for cls, ids in self.label_map.items() for img_id in ids}

        partition_file = self.data_directory / self.partition_file_template.format(partition=partition)
        with open(partition_file) as file:
            partitions = json.load(file)

        if split_name not in partitions:
            msg = f"Split '{split_name}' not found."
            raise ValueError(msg)

        self.image_files = [self.data_directory / f"{filename}.hdf5" for filename in partitions[split_name]]

        self.transform = transform if transform else default_transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_path = self.image_files[index]
        image_id = file_path.stem

        with h5py.File(file_path, "r") as h5file:
            keys = sorted(h5file.keys())
            keys = np.array([key for key in keys if key != "label"])[self.band_indices]
            bands = [np.array(h5file[key]) for key in keys]

            image = np.stack(bands, axis=-1)

        label_class = self.id_to_class[image_id]
        label_index = list(self.label_map.keys()).index(label_class)

        output = {"image": image.astype(np.float32)}

        if self.transform:
            output = self.transform(**output)

        output["label"] = label_index

        return output

    def plot(self, sample: dict[str, torch.Tensor], suptitle: str | None = None) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample (dict[str, torch.Tensor]): A sample returned by :meth:`__getitem__`.
            suptitle (str | None): Optional string to use as a suptitle.

        Returns:
            matplotlib.figure.Figure: A matplotlib Figure with the rendered sample.
        """
        rgb_indices = [self.bands.index(band) for band in self.rgb_bands if band in self.bands]

        if len(rgb_indices) != 3:
            msg = "Dataset doesn't contain some of the RGB bands"
            raise ValueError(msg)

        image = sample["image"]
        label_index = sample["label"]

        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).numpy()

        rgb_image = image[:, :, rgb_indices]
        rgb_image = clip_image(rgb_image)

        class_names = list(self.label_map.keys())
        class_name = class_names[label_index]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(rgb_image)
        ax.axis("off")
        ax.set_title(f"Class: {class_name}")

        if suptitle:
            plt.suptitle(suptitle)

        return fig

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Module containing generic dataset classes"""

import glob
import json
import logging
import os
import re
from abc import ABC
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib as mpl
import numpy as np
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch import Tensor
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import HLSBands, default_transform, filter_valid_files, generate_bands_intervals


class GenericPixelWiseDataset(NonGeoDataset, ABC):
    """
    This is a generic dataset class to be used for instantiating datasets from arguments.
    Ideally, one would create a dataset class specific to a dataset.
    """

    def __init__(
        self,
        data_root: Path,
        label_data_root: Path | None = None,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        rgb_indices: list[int] | None = None,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        constant_scale: float = 1,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
    ) -> None:
        """Constructor

        Args:
            data_root (Path): Path to data root directory
            label_data_root (Path, optional): Path to data root directory with labels.
                If not specified, will use the same as for images.
            image_grep (str, optional): Regular expression appended to data_root to find input images.
                Defaults to "*".
            label_grep (str, optional): Regular expression appended to data_root to find ground truth masks.
                Defaults to "*".
            split (Path, optional): Path to file containing files to be used for this split.
                The file should be a new-line separated prefixes contained in the desired files.
                Files will be seached using glob with the form Path(data_root).glob(prefix + [image or label grep])
            ignore_split_file_extensions (bool, optional): Whether to disregard extensions when using the split
                file to determine which files to include in the dataset.
                E.g. necessary for Eurosat, since the split files specify ".jpg" but files are
                actually ".jpg". Defaults to True.
            allow_substring_split_file (bool, optional): Whether the split files contain substrings
                that must be present in file names to be included (as in mmsegmentation), or exact
                matches (e.g. eurosat). Defaults to True.
            rgb_indices (list[str], optional): Indices of RGB channels. Defaults to [0, 1, 2].
            dataset_bands (list[HLSBands | int | tuple[int, int] | str] | None): Bands present in the dataset. This parameter names input channels (bands) using HLSBands, ints, int ranges, or strings, so that they can then be refered to by output_bands. Defaults to None.
            output_bands (list[HLSBands | int | tuple[int, int] | str] | None): Bands that should be output by the dataset as named by dataset_bands.
            constant_scale (float): Factor to multiply image values by. Defaults to 1.
            transform (Albumentations.Compose | None): Albumentations transform to be applied.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input images with this value. If none, does no replacement. Defaults to None.
            no_label_replace (int | None): Replace nan values in label with this value. If none, does no replacement. Defaults to -1.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Defaults to False.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
        """
        super().__init__()

        self.split_file = split

        label_data_root = label_data_root if label_data_root is not None else data_root
        self.image_files = sorted(glob.glob(os.path.join(data_root, image_grep)))
        self.constant_scale = constant_scale
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.segmentation_mask_files = sorted(glob.glob(os.path.join(label_data_root, label_grep)))
        self.reduce_zero_label = reduce_zero_label
        self.expand_temporal_dimension = expand_temporal_dimension

        if self.expand_temporal_dimension and output_bands is None:
            msg = "Please provide output_bands when expand_temporal_dimension is True"
            raise Exception(msg)
        if self.split_file is not None:
            with open(self.split_file) as f:
                split = f.readlines()
            valid_files = {rf"{substring.strip()}" for substring in split}
            self.image_files = filter_valid_files(
                self.image_files,
                valid_files=valid_files,
                ignore_extensions=ignore_split_file_extensions,
                allow_substring=allow_substring_split_file,
            )
            self.segmentation_mask_files = filter_valid_files(
                self.segmentation_mask_files,
                valid_files=valid_files,
                ignore_extensions=ignore_split_file_extensions,
                allow_substring=allow_substring_split_file,
            )

        # We don't define a split file for prediction
        if not self.split_file:
            # When prediction is enabled, we don't have mask files, so
            # we need to provide a way to run the dataloder in these cases.
            if not self.segmentation_mask_files:
                self.segmentation_mask_files = self.image_files
                # The masks can be `None` since they won't be used in fact. 

        self.rgb_indices = [0, 1, 2] if rgb_indices is None else rgb_indices

        self.dataset_bands = generate_bands_intervals(dataset_bands)
        self.output_bands = generate_bands_intervals(output_bands)

        if self.output_bands and not self.dataset_bands:
            msg = "If output bands provided, dataset_bands must also be provided"
            return Exception(msg)  # noqa: PLE0101

        # There is a special condition if the bands are defined as simple strings.
        if self.output_bands:
            if len(set(self.output_bands) & set(self.dataset_bands)) != len(self.output_bands):
                msg = "Output bands must be a subset of dataset bands"
                raise Exception(msg)

            self.filter_indices = [self.dataset_bands.index(band) for band in self.output_bands]

        else:
            self.filter_indices = None

        # If no transform is given, apply only to transform to torch tensor
        self.transform = transform if transform else default_transform
        # self.transform = transform if transform else ToTensorV2()

        import warnings

        import rasterio

        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = self._load_file(self.image_files[index], nan_replace=self.no_data_replace).to_numpy()
        # to channels last
        if self.expand_temporal_dimension:
            image = rearrange(image, "(channels time) h w -> channels time h w", channels=len(self.output_bands))
        image = np.moveaxis(image, 0, -1)

        if self.filter_indices:
            image = image[..., self.filter_indices]
        output = {
            "image": image.astype(np.float32) * self.constant_scale,
        }
        if self.segmentation_mask_files:
            mask = self._load_file(self.segmentation_mask_files[index], nan_replace=self.no_label_replace)
            output["mask"] = mask.to_numpy()[0]
            if self.reduce_zero_label:
                output["mask"] -= 1
        if self.transform:
            output = self.transform(**output)
        output["filename"] = self.image_files[index]

        return output

    def _load_file(self, path, nan_replace: int | float | None = None) -> xr.DataArray:
        data = rioxarray.open_rasterio(path, masked=True)
        if nan_replace is not None:
            data = data.fillna(nan_replace)
        return data


class GenericNonGeoSegmentationDataset(GenericPixelWiseDataset):
    """GenericNonGeoSegmentationDataset"""

    def __init__(
        self,
        data_root: Path,
        num_classes: int,
        label_data_root: Path | None = None,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        rgb_indices: list[str] | None = None,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        class_names: list[str] | None = None,
        constant_scale: float = 1,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
    ) -> None:
        """Constructor

        Args:
            data_root (Path): Path to data root directory
            num_classes (int): Number of classes in the dataset
            label_data_root (Path, optional): Path to data root directory with labels.
                If not specified, will use the same as for images.
            image_grep (str, optional): Regular expression appended to data_root to find input images.
                Defaults to "*".
            label_grep (str, optional): Regular expression appended to data_root to find ground truth masks.
                Defaults to "*".
            split (Path, optional): Path to file containing files to be used for this split.
                The file should be a new-line separated prefixes contained in the desired files.
                Files will be seached using glob with the form Path(data_root).glob(prefix + [image or label grep])
            ignore_split_file_extensions (bool, optional): Whether to disregard extensions when using the split
                file to determine which files to include in the dataset.
                E.g. necessary for Eurosat, since the split files specify ".jpg" but files are
                actually ".jpg". Defaults to True
            allow_substring_split_file (bool, optional): Whether the split files contain substrings
                that must be present in file names to be included (as in mmsegmentation), or exact
                matches (e.g. eurosat). Defaults to True.
            rgb_indices (list[str], optional): Indices of RGB channels. Defaults to [0, 1, 2].
            dataset_bands (list[HLSBands | int] | None): Bands present in the dataset.
            output_bands (list[HLSBands | int] | None): Bands that should be output by the dataset.
            class_names (list[str], optional): Class names. Defaults to None.
            constant_scale (float): Factor to multiply image values by. Defaults to 1.
            transform (Albumentations.Compose | None): Albumentations transform to be applied.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input images with this value. If none, does no replacement. Defaults to None.
            no_label_replace (int | None): Replace nan values in label with this value. If none, does no replacement. Defaults to None.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Defaults to False.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
        """
        super().__init__(
            data_root,
            label_data_root=label_data_root,
            image_grep=image_grep,
            label_grep=label_grep,
            split=split,
            ignore_split_file_extensions=ignore_split_file_extensions,
            allow_substring_split_file=allow_substring_split_file,
            rgb_indices=rgb_indices,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            expand_temporal_dimension=expand_temporal_dimension,
            reduce_zero_label=reduce_zero_label,
        )
        self.num_classes = num_classes
        self.class_names = class_names

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        if "mask" in item:
            item["mask"] = item["mask"].long()
        return item

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None, show_axes: bool | None = False) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle
            show_axes: whether to show axes or not

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        image = sample["image"]
        if len(image.shape) == 5:
            return
        if isinstance(image, Tensor):
            image = image.numpy()
        image = image.take(self.rgb_indices, axis=0)
        image = np.transpose(image, (1, 2, 0))
        image = (image - image.min(axis=(0, 1))) * (1 / image.max(axis=(0, 1)))
        image = np.clip(image, 0, 1)

        label_mask = sample["mask"]
        if isinstance(label_mask, Tensor):
            label_mask = label_mask.numpy()

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_mask = sample["prediction"]
            if isinstance(prediction_mask, Tensor):
                prediction_mask = prediction_mask.numpy()

        return self._plot_sample(
            image,
            label_mask,
            self.num_classes,
            prediction=prediction_mask if showing_predictions else None,
            suptitle=suptitle,
            class_names=self.class_names,
            show_axes=show_axes,
        )

    @staticmethod
    def _plot_sample(image, label, num_classes, prediction=None, suptitle=None, class_names=None, show_axes=False):
        num_images = 5 if prediction is not None else 4
        fig, ax = plt.subplots(1, num_images, figsize=(12, 10), layout="compressed")
        axes_visibility = "on" if show_axes else "off"

        # for legend
        ax[0].axis("off")

        norm = mpl.colors.Normalize(vmin=0, vmax=num_classes - 1)
        ax[1].axis(axes_visibility)
        ax[1].title.set_text("Image")
        ax[1].imshow(image)

        ax[2].axis(axes_visibility)
        ax[2].title.set_text("Ground Truth Mask")
        ax[2].imshow(label, cmap="jet", norm=norm)

        ax[3].axis(axes_visibility)
        ax[3].title.set_text("GT Mask on Image")
        ax[3].imshow(image)
        ax[3].imshow(label, cmap="jet", alpha=0.3, norm=norm)

        if prediction is not None:
            ax[4].axis(axes_visibility)
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


class GenericNonGeoPixelwiseRegressionDataset(GenericPixelWiseDataset):
    """GenericNonGeoPixelwiseRegressionDataset"""

    def __init__(
        self,
        data_root: Path,
        label_data_root: Path | None = None,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        rgb_indices: list[int] | None = None,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        constant_scale: float = 1,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
    ) -> None:
        """Constructor

        Args:
            data_root (Path): Path to data root directory
            label_data_root (Path, optional): Path to data root directory with labels.
                If not specified, will use the same as for images.
            image_grep (str, optional): Regular expression appended to data_root to find input images.
                Defaults to "*".
            label_grep (str, optional): Regular expression appended to data_root to find ground truth masks.
                Defaults to "*".
            split (Path, optional): Path to file containing files to be used for this split.
                The file should be a new-line separated prefixes contained in the desired files.
                Files will be seached using glob with the form Path(data_root).glob(prefix + [image or label grep])
            ignore_split_file_extensions (bool, optional): Whether to disregard extensions when using the split
                file to determine which files to include in the dataset.
                E.g. necessary for Eurosat, since the split files specify ".jpg" but files are
                actually ".jpg". Defaults to True.
            allow_substring_split_file (bool, optional): Whether the split files contain substrings
                that must be present in file names to be included (as in mmsegmentation), or exact
                matches (e.g. eurosat). Defaults to True.
            rgb_indices (list[str], optional): Indices of RGB channels. Defaults to [0, 1, 2].
            dataset_bands (list[HLSBands | int] | None): Bands present in the dataset.
            output_bands (list[HLSBands | int] | None): Bands that should be output by the dataset.
            constant_scale (float): Factor to multiply image values by. Defaults to 1.
            transform (Albumentations.Compose | None): Albumentations transform to be applied.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input images with this value. If none, does no replacement. Defaults to None.
            no_label_replace (int | None): Replace nan values in label with this value. If none, does no replacement. Defaults to None.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Defaults to False.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
        """
        super().__init__(
            data_root,
            label_data_root=label_data_root,
            image_grep=image_grep,
            label_grep=label_grep,
            split=split,
            ignore_split_file_extensions=ignore_split_file_extensions,
            allow_substring_split_file=allow_substring_split_file,
            rgb_indices=rgb_indices,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            expand_temporal_dimension=expand_temporal_dimension,
            reduce_zero_label=reduce_zero_label,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        if "mask" in item:
            item["mask"] = item["mask"].float()
        return item

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None, show_axes: bool | None = False) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample (dict[str, Tensor]): a sample returned by :meth:`__getitem__`
            suptitle (str|None): optional string to use as a suptitle
            show_axes (bool|None): whether to show axes or not

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        image = sample["image"]
        if len(image.shape) == 5:
            return
        if isinstance(image, Tensor):
            image = image.numpy()
        image = image.take(self.rgb_indices, axis=0)
        image = np.transpose(image, (1, 2, 0))
        image = (image - image.min(axis=(0, 1))) * (1 / image.max(axis=(0, 1)))
        image = np.clip(image, 0, 1)

        label_mask = sample["mask"]
        if isinstance(label_mask, Tensor):
            label_mask = label_mask.numpy()

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_mask = sample["prediction"]
            if isinstance(prediction_mask, Tensor):
                prediction_mask = prediction_mask.numpy()

        return self._plot_sample(
            image,
            label_mask,
            prediction=prediction_mask if showing_predictions else None,
            suptitle=suptitle,
            show_axes=show_axes,
        )

    @staticmethod
    def _plot_sample(image, label, prediction=None, suptitle=None, show_axes=False):
        num_images = 4 if prediction is not None else 3
        fig, ax = plt.subplots(1, num_images, figsize=(12, 10), layout="compressed")
        axes_visibility = "on" if show_axes else "off"

        norm = mpl.colors.Normalize(vmin=label.min(), vmax=label.max())
        ax[0].axis(axes_visibility)
        ax[0].title.set_text("Image")
        ax[0].imshow(image)

        ax[1].axis(axes_visibility)
        ax[1].title.set_text("Ground Truth Mask")
        ax[1].imshow(label, cmap="Greens", norm=norm)

        ax[2].axis(axes_visibility)
        ax[2].title.set_text("GT Mask on Image")
        ax[2].imshow(image)
        ax[2].imshow(label, cmap="Greens", alpha=0.3, norm=norm)
        # ax[2].legend()

        if prediction is not None:
            ax[3].axis(axes_visibility)
            ax[3].title.set_text("Predicted Mask")
            ax[3].imshow(prediction, cmap="Greens", norm=norm)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig

class GenericNonGeoPixelwiseDataset_Custom(GenericPixelWiseDataset):
    """GenericNonGeoPixelwiseRegressionDataset with JSON file support"""

    def __init__(
        self,
        data_root: Path,
        label_data_root: Path | None = None,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        rgb_indices: list[int] | None = None,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        constant_scale: float = 1,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        json_file: Path | None = None,
    ) -> None:
        """Constructor

        Args:
            data_root (Path): Path to data root directory
            label_data_root (Path, optional): Path to data root directory with labels.
                If not specified, will use the same as for images.
            image_grep (str, optional): Regular expression appended to data_root to find input images.
                Defaults to "*".
            label_grep (str, optional): Regular expression appended to data_root to find ground truth masks.
                Defaults to "*".
            split (Path, optional): Path to file containing files to be used for this split.
                The file should be a new-line separated prefixes contained in the desired files.
                Files will be seached using glob with the form Path(data_root).glob(prefix + [image or label grep])
            ignore_split_file_extensions (bool, optional): Whether to disregard extensions when using the split
                file to determine which files to include in the dataset.
                E.g. necessary for Eurosat, since the split files specify ".jpg" but files are
                actually ".jpg". Defaults to True.
            allow_substring_split_file (bool, optional): Whether the split files contain substrings
                that must be present in file names to be included (as in mmsegmentation), or exact
                matches (e.g. eurosat). Defaults to True.
            rgb_indices (list[str], optional): Indices of RGB channels. Defaults to [0, 1, 2].
            dataset_bands (list[HLSBands | int] | None): Bands present in the dataset.
            output_bands (list[HLSBands | int] | None): Bands that should be output by the dataset.
            constant_scale (float): Factor to multiply image values by. Defaults to 1.
            transform (Albumentations.Compose | None): Albumentations transform to be applied.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input images with this value. If none, does no replacement. Defaults to None.
            no_label_replace (int | None): Replace nan values in label with this value. If none, does no replacement. Defaults to None.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Defaults to False.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
            json_file (Path | None): Path to JSON file defining sample names and band file paths.
                If provided, overrides directory-based file discovery.
                Expected JSON format:
                {
                    "sample_name_1": {
                        "B01": "path/to/sample1_B01_60m.jp2",
                        "B02": "path/to/sample1_B02_10m.jp2",
                        ...
                        "B12": "path/to/sample1_B12_20m.jp2"
                    },
                    "sample_name_2": {
                        "B01": "path/to/sample2_B01_60m.jp2",
                        ...
                    }
                }
                Notes:
                - Sample names are used as identifiers and will be included in output filenames
                - Band names must match Sentinel-2 bands: B01-B12 (B8A instead of B08A)
                - File paths can be relative (to working directory) or absolute
                - Resolution suffix (10m/20m/60m) is automatically extracted from filenames
                - Missing or invalid file paths will cause that sample to be skipped with a warning
        """
        super().__init__(
            data_root,
            label_data_root=label_data_root,
            image_grep=image_grep,
            label_grep=label_grep,
            split=split,
            ignore_split_file_extensions=ignore_split_file_extensions,
            allow_substring_split_file=allow_substring_split_file,
            rgb_indices=rgb_indices,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            expand_temporal_dimension=expand_temporal_dimension,
            reduce_zero_label=reduce_zero_label,
        )

        # Load JSON file if provided
        self.json_file = json_file
        self.band_mapping = None
        if json_file is not None:
            logger = logging.getLogger("terratorch")

            with open(json_file, 'r') as f:
                raw_mapping = json.load(f)

            # Validate file paths and filter out samples with missing files
            self.band_mapping = {}
            skipped_samples = []

            for sample_name, band_dict in raw_mapping.items():
                missing_files = []
                for band, file_path in band_dict.items():
                    if not os.path.exists(file_path):
                        missing_files.append(f"{band}: {file_path}")

                if missing_files:
                    skipped_samples.append(sample_name)
                    logger.warning(
                        f"Skipping sample '{sample_name}' - missing {len(missing_files)} file(s):\n  " +
                        "\n  ".join(missing_files[:3]) +  # Show first 3 missing files
                        (f"\n  ... and {len(missing_files) - 3} more" if len(missing_files) > 3 else "")
                    )
                else:
                    # All files exist, include this sample
                    self.band_mapping[sample_name] = band_dict

            if skipped_samples:
                logger.info(f"Skipped {len(skipped_samples)} sample(s) due to missing files. Valid samples: {len(self.band_mapping)}")

            # Override image_files with validated sample names from JSON
            self.image_files = sorted(list(self.band_mapping.keys()))
            # For JSON-based loading, we don't use mask files
            self.segmentation_mask_files = self.image_files

    def _load_file(self, path, nan_replace: int | float | None = None) -> xr.DataArray:
        # Target order of Sentinel-2 bands expected by the model
        BAND_ORDER = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]

        # If using JSON mapping, path is the sample name
        if self.band_mapping is not None:
            logger = logging.getLogger("terratorch")

            sample_name = path
            if sample_name not in self.band_mapping:
                raise KeyError(f"Sample '{sample_name}' not found in JSON mapping")

            # Build candidates dict from JSON mapping
            candidates: dict[str, dict[int, str]] = {}
            for band, file_path in self.band_mapping[sample_name].items():
                # Verify file still exists (in case it was deleted after validation)
                if not os.path.exists(file_path):
                    logger.warning(f"File missing for sample '{sample_name}', band {band}: {file_path}")
                    continue

                # Extract resolution from filename
                m = re.search(r"_(10|20|60)m\.jp2$", file_path)
                if m:
                    res = int(m.group(1))
                    candidates.setdefault(band, {})[res] = file_path
                else:
                    # Default to 10m if resolution not in filename
                    candidates.setdefault(band, {})[10] = file_path
        else:
            # Original directory-based logic
            # Collect candidate files for 10m, 20m, 60m (e.g., Txx_yyy_B02_10m.jp2)
            patterns = [
                os.path.join(path, "*_B??_10m.jp2"),
                os.path.join(path, "*_B??_20m.jp2"),
                os.path.join(path, "*_B??_60m.jp2"),
                os.path.join(path, "*_B8A_10m.jp2"),
                os.path.join(path, "*_B8A_20m.jp2"),
                os.path.join(path, "*_B8A_60m.jp2"),
            ]
            all_paths: list[str] = []
            for pat in patterns:
                all_paths.extend(glob.glob(pat))

            # Parse band and resolution from filename
            def parse(p: str) -> tuple[str, int] | None:
                m = re.search(r"_B(\d{2}|8A)_(10|20|60)m\.jp2$", os.path.basename(p))
                if not m:
                    return None
                band = f"B{m.group(1)}"
                res = int(m.group(2))
                return band, res

            # Keep best available resolution per band: prefer 10m > 20m > 60m
            # Candidates: {"B02": {"10": "xyz.jp2"}, {"B09": {"60": "xyz.jp2"}, ...}
            candidates: dict[str, dict[int, str]] = {}
            for p in all_paths:
                parsed = parse(p)
                if not parsed:
                    continue
                band, res = parsed
                candidates.setdefault(band, {})[res] = p

        # Choose a 10m reference grid if available (prefer true 10m bands)
        ref_path: str | None = None
        for b in ["B02", "B03", "B04", "B08"] + BAND_ORDER:
            if b in candidates and 10 in candidates[b]:
                ref_path = candidates[b][10]
                break

        ref_da: xr.DataArray | None = None
        if ref_path is not None:
            ref_da = rioxarray.open_rasterio(ref_path, masked=True)
        else:
            # Fallback: build a 10m reference from any available band by resampling itself
            # (extent/CRS preserved, only resolution changes)
            if not candidates:
                raise FileNotFoundError(f"No band files found in: {path}")
            any_band = next(iter(candidates))
            any_res = sorted(candidates[any_band].keys())[0]
            base_da = rioxarray.open_rasterio(candidates[any_band][any_res], masked=True)
            # Reproject to 10m grid using the same CRS and bounds
            ref_da = base_da.rio.reproject(
                base_da.rio.crs,
                resolution=(10, 10),
                resampling=Resampling.bilinear,
            )

        das: list[xr.DataArray] = []
        template = ref_da
        if template is None:
            raise RuntimeError("Reference grid not initialized; cannot build band stack")

        # Build stack in strict BAND_ORDER; fill missing with zeros
        for b in BAND_ORDER:
            if b in candidates:
                # Prefer 10m, else 20m, else 60m
                path_b = candidates[b].get(10) or candidates[b].get(20) or candidates[b].get(60)
                da = rioxarray.open_rasterio(path_b, masked=True)
                # If resolution differs, upsample to match 10m reference
                try:
                    # returns (xres, yres) tuple like (10.0, 10.0)
                    da_res = da.rio.resolution()
                    ref_res = template.rio.resolution()
                except Exception:
                    da_res = ref_res = (None, None)

                if ref_res != da_res:
                    da = da.rio.reproject_match(template, resampling=Resampling.bilinear)
            else:
                # Missing band on disk; use zeros
                da = xr.zeros_like(template, dtype="float32")

            das.append(da)

        cube = xr.concat(das, dim="band")
        data = cube.astype("float32")

        if nan_replace is not None:
            data = data.fillna(nan_replace)
        return data
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        return item
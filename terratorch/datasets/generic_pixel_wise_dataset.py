# Copyright contributors to the Terratorch project

"""Module containing generic dataset classes
"""
import glob
import os
from abc import ABC
from functools import partial
from pathlib import Path
from typing import Any, List, Union

import albumentations as A
import matplotlib as mpl
import numpy as np
import rioxarray
import torch
import xarray as xr
from albumentations.pytorch import ToTensorV2
from einops import rearrange
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch import Tensor
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import HLSBands, filter_valid_files, to_tensor


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
        dataset_bands: list[HLSBands | int | tuple[int, int] | str ] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str ] | None = None,
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
        self.rgb_indices = [0, 1, 2] if rgb_indices is None else rgb_indices

        is_bands_by_interval = self._check_if_its_defined_by_interval(dataset_bands, output_bands) 

        # If the bands are defined by sub-intervals or not.
        if is_bands_by_interval:
            self.dataset_bands = self._generate_bands_intervals(dataset_bands)
            self.output_bands = self._generate_bands_intervals(output_bands)
        else:
            self.dataset_bands = dataset_bands
            self.output_bands = output_bands
                
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
        self.transform = transform if transform else lambda **batch: to_tensor(batch)
        # self.transform = transform if transform else ToTensorV2()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = self._load_file(self.image_files[index], nan_replace = self.no_data_replace).to_numpy()
        # to channels last
        if self.expand_temporal_dimension:
            image = rearrange(image, "(channels time) h w -> channels time h w", channels=len(self.output_bands))
        image = np.moveaxis(image, 0, -1)

        if self.filter_indices:
            image = image[..., self.filter_indices]
        output = {
            "image": image.astype(np.float32) * self.constant_scale,
            "mask": self._load_file(self.segmentation_mask_files[index], nan_replace = self.no_label_replace).to_numpy()[0],
            "filename": self.image_files[index],
        }

        if self.reduce_zero_label:
            output["mask"] -= 1
        if self.transform:
            output = self.transform(**output)
        return output
    
    def _load_file(self, path, nan_replace: int | float | None = None) -> xr.DataArray:
        data = rioxarray.open_rasterio(path, masked=True)
        if nan_replace is not None:
            data = data.fillna(nan_replace)
        return data

    def _generate_bands_intervals(self, bands_intervals:List[List[int]] = None):
        bands = list()
        for b_interval in bands_intervals:
            bands_sublist = np.arange(b_interval[0], b_interval[1] + 1).astype(int).tolist()
            bands.append(bands_sublist)
        return sorted(sum(bands, []))

    def _bands_as_int_or_str(self, dataset_bands, output_bands) -> type:

        band_type = [None, None]
        if not dataset_bands and not output_bands:
            return None
        else:
            for b, bands_list in enumerate([dataset_bands, output_bands]):
                if all([type(band)==int for band in bands_list]):
                    band_type[b] = int
                elif all([type(band)==str for band in bands_list]):
                    band_type[b] = str
                else:
                    pass 
            if band_type.count(band_type[0]) == len(band_type):
                return band_type[0]
            else:
                raise Exception("The bands must be or all str or all int.")

    def _check_if_its_defined_by_interval(self, dataset_bands: list[int] | list[tuple[int]] = None,
                                          output_bands: list[int] | list[tuple[int]] = None) -> bool:

        is_dataset_bands_defined = self._bands_defined_by_interval(bands_list=dataset_bands)
        is_output_bands_defined = self._bands_defined_by_interval(bands_list=output_bands)

        if is_dataset_bands_defined and is_output_bands_defined:
            return True
        elif not is_dataset_bands_defined and not is_output_bands_defined:
            return False
        else:
            raise Exception(f"Both dataset_bands and output_bands must have the same type, but received {dataset_bands} and {output_bands}")

    def _bands_defined_by_interval(self, bands_list: list[int] | list[tuple[int]] = None) -> bool:
        if not bands_list:
            return False
        elif all([type(band)==int or type(band)==str or isinstance(band, HLSBands) for band in bands_list]):
            return False
        elif all([isinstance(subinterval, tuple) for subinterval in bands_list]):
            bands_list_ = [list(subinterval) for subinterval in bands_list]
            if all([type(band)==int for band in sum(bands_list_, [])]):
                return True
            else:
                raise Exception(f"Whe using subintervals, the limits must be int.")
        else:
            raise Exception(f"Excpected List[int] or List[str] or List[tuple[int, int]], but received {type(bands_list)}.")

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
        dataset_bands: list[HLSBands | int | tuple[int, int] | str ] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str ] | None = None,
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
        item["mask"] = item["mask"].long()
        return item

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle

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
        )

    @staticmethod
    def _plot_sample(image, label, num_classes, prediction=None, suptitle=None, class_names=None):
        num_images = 5 if prediction is not None else 4
        fig, ax = plt.subplots(1, num_images, figsize=(12, 10), layout="compressed")

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

        if prediction is not None:
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
        dataset_bands: list[HLSBands | int | tuple[int, int] | str ] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str ] | None = None,
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
        item["mask"] = item["mask"].float()
        return item

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample (dict[str, Tensor]): a sample returned by :meth:`__getitem__`
            suptitle (str|None): optional string to use as a suptitle

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
        )

    @staticmethod
    def _plot_sample(image, label, prediction=None, suptitle=None):
        num_images = 4 if prediction is not None else 3
        fig, ax = plt.subplots(1, num_images, figsize=(12, 10), layout="compressed")

        norm = mpl.colors.Normalize(vmin=label.min(), vmax=label.max())
        ax[0].axis("off")
        ax[0].title.set_text("Image")
        ax[0].imshow(image)

        ax[1].axis("off")
        ax[1].title.set_text("Ground Truth Mask")
        ax[1].imshow(label, cmap="Greens", norm=norm)

        ax[2].axis("off")
        ax[2].title.set_text("GT Mask on Image")
        ax[2].imshow(image)
        ax[2].imshow(label, cmap="Greens", alpha=0.3, norm=norm)
        # ax[2].legend()

        if prediction is not None:
            ax[3].title.set_text("Predicted Mask")
            ax[3].imshow(prediction, cmap="Greens", norm=norm)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig

# Copyright contributors to the Terratorch project

"""Module containing generic dataset classes"""

import glob
import logging
import os
import torch
from abc import ABC
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib as mpl
import numpy as np
import rioxarray
import xarray as xr
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import HLSBands, default_transform, filter_valid_files, generate_bands_intervals
from terratorch.datasets.transforms import MultimodalTransforms


class MultimodalToTensor():
    def __init__(self, modalities):
        self.modalities = modalities
    def __call__(self, d):
        new_dict = {}
        for k, v in d.items():
            if not isinstance(v, np.ndarray):
                new_dict[k] = v
            else:
                # TODO: This code has hard assumptions on the data structure
                if k in self.modalities and len(v.shape) >= 3:  # Assuming raster modalities with 3+ dimensions
                    if len(v.shape) <= 4:
                        v = np.moveaxis(v, -1, 0)  # C, H, W or C, T, H, W
                    elif len(v.shape) == 5:
                        v = np.moveaxis(v, -1, 1)  # B, C, T, H, W
                    else:
                        raise ValueError(f'Unexpected shape for {k}: {v.shape}')
                new_dict[k] = torch.from_numpy(v)
        return new_dict


class GenericMultimodalDataset(NonGeoDataset, ABC):
    """
    This is a generic dataset class to be used for instantiating datasets from arguments.
    Ideally, one would create a dataset class specific to a dataset.
    """

    def __init__(
        self,
        data_root: dict[Path],
        label_data_root: Path | list[Path] | None = None,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        rgb_modality: str | None = None,
        rgb_indices: list[int] | None = None,
        allow_missing_modalities: bool = False,  # TODO: Not implemented on a data module level yet (collate_fn required).
        allow_substring_split_file: bool = False,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        constant_scale: dict[float] = None,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        channel_position: int = -1,
        *args, **kwargs,
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

        self.modalities = list(data_root.keys())
        assert 'mask' not in self.modalities, "Modality cannot be called 'mask'."

        # Convert path strings to lists
        for m, m_dir in data_root.items():
            if not isinstance(m_dir, list):
                data_root[m] = [m_dir]
        if label_data_root and not isinstance(label_data_root, list):
            label_data_root = [label_data_root]

        self.constant_scale = {m: constant_scale[m] or 1. for m in self.modalities}
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.reduce_zero_label = reduce_zero_label
        self.expand_temporal_dimension = expand_temporal_dimension
        self.channel_position = channel_position

        if self.expand_temporal_dimension and len(dataset_bands) != self.modalities:
            msg = "Please provide dataset_bands for each modality when expand_temporal_dimension is True"
            raise Exception(msg)

        # Load samples based on split file
        if self.split_file is not None:
            with open(self.split_file) as f:
                split = f.readlines()
            valid_files = {rf"{substring.strip()}" for substring in split}

        else:
            image_files = {}
            for m, m_dirs in data_root.items():
                dir_lists = [glob.glob(os.path.join(r, image_grep[m])) for r in m_dirs]
                image_files[m] = sorted([p for l in dir_lists for p in l])  # Concatenate

            if label_data_root:
                dir_lists = [glob.glob(os.path.join(r, label_grep)) for r in label_data_root]
                image_files['mask'] = sorted([p for l in dir_lists for p in l])  # Concatenate

            if allow_substring_split_file:
                # Get exact match of filenames
                get_file_id = lambda s: os.path.basename(s)
            else:
                # Remove file extensions
                get_file_id = lambda s: os.path.splitext(os.path.basename(s))[0]

            if allow_missing_modalities:
                valid_files = set([get_file_id(file) for file in np.concatenate(list(image_files.values()))])
            else:
                valid_files = [get_file_id(file) for file in image_files[self.modalities[0]]]

        self.samples = []
        num_modalities = len(self.modalities) + int(label_data_root is not None)
        # Iterate over all files in split
        for file in valid_files:
            sample = {}
            # Iterate over all modalities
            for m, m_dirs in data_root.items():
                # Iterate over all directories of the current modality
                for m_dir in m_dirs:
                    if allow_substring_split_file:
                        # Substring match with image_grep
                        m_files = glob.glob(os.path.join(m_dir, file + image_grep[m]))
                        if m_files:
                            sample[m] = m_files[0]
                            break
                    else:
                        # Exact match
                        file_path = os.path.join(m_dir, file)
                        if os.path.isfile(file_path):
                            sample[m] = file_path
                            break
            if label_data_root:
                for l_dir in label_data_root:
                    if allow_substring_split_file:
                        # Substring match with label_grep
                        l_files = glob.glob(os.path.join(l_dir, file + label_grep))
                        if l_files:
                            sample['mask'] = l_files[0]
                            break
                    else:
                        # Exact match
                        file_path = os.path.join(l_dir, file)
                        if os.path.isfile(file_path):
                            sample['mask'] = file_path
                            break
                if 'mask' not in sample:
                    # Only add sample if mask is present
                    break

            if len(sample) == num_modalities or allow_missing_modalities:
                self.samples.append(sample)

        self.rgb_modality = rgb_modality or self.modalities[0]
        self.rgb_indices = rgb_indices or [0, 1, 2]

        if dataset_bands is not None:
            self.dataset_bands = {m: generate_bands_intervals(m_bands)
                                  for m, m_bands in dataset_bands.items()}
        else:
            self.dataset_bands = None
        if output_bands is not None:
            self.output_bands = {m: generate_bands_intervals(m_bands)
                                  for m, m_bands in output_bands.items()}
            for modality in self.modalities:
                if modality in self.output_bands and modality not in self.dataset_bands:
                    msg = f"If output bands are provided, dataset_bands must also be provided (modality: {modality})"
                    raise Exception(msg)  # noqa: PLE0101
        else:
            self.output_bands = {}

        self.filter_indices = {}
        # There is a special condition if the bands are defined as simple strings.
        if self.output_bands:
            for m in self.output_bands.keys():
                if m not in self.output_bands or self.output_bands[m] == self.dataset_bands[m]:
                    continue
                if len(set(self.output_bands[m]) & set(self.dataset_bands[m])) != len(self.output_bands[m]):
                    msg = f"Output bands must be a subset of dataset bands (Modality: {m})"
                    raise Exception(msg)

                self.filter_indices[m] = [self.dataset_bands[m].index(band) for band in self.output_bands[m]]

        # If no transform is given, apply only to transform to torch tensor
        if isinstance(transform, A.Compose):
            self.transform = MultimodalTransforms(transform)
        elif transform is None:
            self.transform = MultimodalToTensor(self.modalities)
        else:
            # Modality-specific transforms
            transform = {m: transform[m] if m in transform else default_transform
                         for m in self.modalities}
            self.transform = MultimodalTransforms(transform, shared=False)

        import warnings
        import rasterio
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        output = {}
        if isinstance(index, tuple):
            # Load only sampled modalities instead of all modalities
            # (see sample_num_modalities in GenericMultiModalDataModule for details)
            index, modalities = index
            sample = {m: self.samples[index][m] for m in modalities}
        else:
            sample = self.samples[index]

        for modality, file in sample.items():
            data = self._load_file(
                file, nan_replace=self.no_label_replace if modality == 'mask' else self.no_data_replace).to_numpy()

            # Expand temporal dim
            if modality in self.filter_indices and self.expand_temporal_dimension:
                data = rearrange(data, "(channels time) h w -> channels time h w",
                                 channels=len(self.dataset_bands[modality]))

            if modality == 'mask':
                data = data[0]

            if len(data.shape) >= 3 and self.channel_position:
                # to channels last (required by albumentations)
                data = np.moveaxis(data, self.channel_position, -1)

            if modality in self.filter_indices:
                data = data[..., self.filter_indices[modality]]

            if modality != 'mask':
                data = data.astype(np.float32) * self.constant_scale[modality]

            output[modality] = data

        if self.reduce_zero_label:
            output["mask"] -= 1
        if self.transform:
            output = self.transform(output)

        # Tasks expect data to be stored in 'image', moving modalities to image dict
        output = {
            'image': {m: output[m] for m in self.modalities if m in output},
            'mask': output['mask'] if 'mask' in output else None,
            'filename': self.samples[index]
        }

        return output

    def _load_file(self, path, nan_replace: int | float | None = None) -> xr.DataArray:
        if path.endswith('.zarr') or path.endswith('.zarr.zip'):
            data = xr.open_zarr(path, mask_and_scale=True)
            data_var = list(data.data_vars)[0]  # TODO: Make data var configurable if required (e.g. for time/loc)
            data = data[data_var]
        elif path.endswith('.npy'):
            data = xr.DataArray(np.load(path))
        else:
            data = rioxarray.open_rasterio(path, masked=True)

        if nan_replace is not None:
            data = data.fillna(nan_replace)
        return data


class GenericMultimodalSegmentationDataset(GenericMultimodalDataset):
    """GenericNonGeoSegmentationDataset"""

    def __init__(
        self,
        data_root: Path,
        num_classes: int,
        label_data_root: Path,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        rgb_modality: str | None = None,
        rgb_indices: list[str] | None = None,
        allow_missing_modalities: bool = False,
        allow_substring_split_file: bool = False,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        class_names: list[str] | None = None,
        constant_scale: float = 1,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        channel_position: int = -3,
    ) -> None:
        """Constructor

        Args:
            TODO: Update docs
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
            rgb_modality=rgb_modality,
            rgb_indices=rgb_indices,
            allow_missing_modalities=allow_missing_modalities,
            allow_substring_split_file=allow_substring_split_file,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            expand_temporal_dimension=expand_temporal_dimension,
            reduce_zero_label=reduce_zero_label,
            channel_position=channel_position,
        )
        self.num_classes = num_classes
        self.class_names = class_names

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["mask"] = item["mask"].long()
        return item

    def plot(self, sample: dict[str, torch.Tensor], suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        raise NotImplementedError('Code is based on the generic single-modality dataset and not yet adapted. '
                                  'Set `export TERRATORCH_NUM_VAL_PLOTS=0` before running terratorch.')

        image = sample[self.rgb_modality]
        if len(image.shape) == 5:  # TODO: Fix plot code.
            return
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        image = image.take(self.rgb_indices, axis=0)
        image = np.transpose(image, (1, 2, 0))
        image = (image - image.min(axis=(0, 1))) * (1 / image.max(axis=(0, 1)))
        image = np.clip(image, 0, 1)

        label_mask = sample["mask"]
        if isinstance(label_mask, torch.Tensor):
            label_mask = label_mask.numpy()

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_mask = sample["prediction"]
            if isinstance(prediction_mask, torch.Tensor):
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


class GenericMultimodalPixelwiseRegressionDataset(GenericMultimodalDataset):
    """GenericNonGeoPixelwiseRegressionDataset"""

    def __init__(
        self,
        data_root: Path,
        label_data_root: Path,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        rgb_modality: str | None = None,
        rgb_indices: list[int] | None = None,
        allow_missing_modalities : bool = False,
        allow_substring_split_file: bool = False,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        constant_scale: float = 1,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        channel_position: int = -3,
    ) -> None:
        """Constructor

        Args:
            TODO: Update docs
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
            rgb_modality=rgb_modality,
            rgb_indices=rgb_indices,
            allow_missing_modalities=allow_missing_modalities,
            allow_substring_split_file=allow_substring_split_file,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            expand_temporal_dimension=expand_temporal_dimension,
            reduce_zero_label=reduce_zero_label,
            channel_position=channel_position,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["mask"] = item["mask"].float()
        return item

    def plot(self, sample: dict[str, torch.Tensor], suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample (dict[str, Tensor]): a sample returned by :meth:`__getitem__`
            suptitle (str|None): optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        raise NotImplementedError('Code is based on the generic single-modality dataset and not yet adapted. '
                                  'Set `export TERRATORCH_NUM_VAL_PLOTS=0` before running terratorch.')

        image = sample["image"]
        if len(image.shape) == 5:
            return
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        image = image.take(self.rgb_indices, axis=0)
        image = np.transpose(image, (1, 2, 0))
        image = (image - image.min(axis=(0, 1))) * (1 / image.max(axis=(0, 1)))
        image = np.clip(image, 0, 1)

        label_mask = sample["mask"]
        if isinstance(label_mask, torch.Tensor):
            label_mask = label_mask.numpy()

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_mask = sample["prediction"]
            if isinstance(prediction_mask, torch.Tensor):
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

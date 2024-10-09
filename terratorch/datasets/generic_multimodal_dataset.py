# Copyright contributors to the Terratorch project

"""Module containing generic dataset classes"""

import glob
import logging
import os
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
from torch import Tensor
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import HLSBands, default_transform, filter_valid_files, generate_bands_intervals


class GenericMultimodalDataset(NonGeoDataset, ABC):
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
        rgb_modality: str | None = None,
        rgb_indices: list[int] | None = None,
        allow_missing_modalities: bool = False,  # TODO: Not implemented on a data module level yet (collate_fn required).
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        constant_scale: dict[float] = None,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
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

        self.modalities = list(data_root.keys())
        assert 'mask' not in self.modalities, "Modality cannot be called 'mask'."
        self.label_data_root = label_data_root
        # Get files per modality
        if image_grep:
            self.image_files = {m: sorted(glob.glob(os.path.join(m_root, image_grep[m])))
                                for m, m_root in data_root.items()}
        else:
            self.image_files = {m: sorted(glob.glob(m_root)) for m, m_root in data_root.items()}
        self.constant_scale = constant_scale or {m: 1. for m in self.modalities}
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        if self.label_data_root:
            self.segmentation_mask_files = sorted(glob.glob(os.path.join(label_data_root, label_grep)))
        else:
            self.segmentation_mask_files = None
        self.reduce_zero_label = reduce_zero_label
        self.expand_temporal_dimension = expand_temporal_dimension

        if self.expand_temporal_dimension and len(dataset_bands) != self.modalities:
            msg = "Please provide dataset_bands for each modality when expand_temporal_dimension is True"
            raise Exception(msg)

        if self.split_file is not None:
            with open(self.split_file) as f:
                split = f.readlines()
            valid_files = {rf"{substring.strip()}" for substring in split}
            if not ignore_split_file_extensions or not allow_substring_split_file:
                # TODO: Only need for special cases, can we generalize the multi-modal samples and remove this part?
                for m, m_files in self.image_files.items():
                    self.image_files[m] = filter_valid_files(
                        m_files,
                        valid_files=valid_files,
                        ignore_extensions=ignore_split_file_extensions,
                        allow_substring=allow_substring_split_file,
                    )
                if self.segmentation_mask_files:
                    self.segmentation_mask_files = filter_valid_files(
                        self.segmentation_mask_files,
                        valid_files=valid_files,
                        ignore_extensions=ignore_split_file_extensions,
                        allow_substring=allow_substring_split_file,
                    )
        else:
            logging.warning('No split file provided. '
                            'This requires that all modalities have the same filename for aligned samples.')
            if allow_missing_modalities:
                all_files = [os.path.splitext(os.path.basename(file))[0]
                               for file in np.concatenate(list(self.image_files.values()))]
                valid_files = list(set(all_files))
            else:
                valid_files = [os.path.splitext(os.path.basename(file))[0]
                               for file in self.image_files[self.modalities[0]]]
            logging.info(f'Found {len(valid_files)} file names.')

        # Get multi-modal samples in form: {'modality': path, ..., 'mask': path}
        self.samples = []
        num_modalities = len(self.modalities) + int(self.segmentation_mask_files is not None)
        for split_file in valid_files:
            sample = {}
            for m, files in self.image_files.items():
                matching_files = [file for file in files if split_file in file]
                if matching_files:
                    sample[m] = matching_files[0]
            if self.segmentation_mask_files:
                matching_files = [file for file in self.segmentation_mask_files if split_file in file]
                if matching_files:
                    sample['mask'] = matching_files[0]
                else:
                    # Skip samples with missing labels
                    continue
            if allow_missing_modalities or len(sample) == num_modalities:
                self.samples.append(sample)

        self.rgb_modality = rgb_modality or self.modalities[0]
        self.rgb_indices = rgb_indices or [0, 1, 2]

        if dataset_bands is not None:
            self.dataset_bands = {m: generate_bands_intervals(m_bands)
                                  for m, m_bands in dataset_bands.items()}
        if output_bands is not None:
            self.output_bands = {m: generate_bands_intervals(m_bands)
                                  for m, m_bands in output_bands.items()}

        for modality in self.modalities:
            if modality in self.output_bands and modality not in self.dataset_bands:
                msg = f"If output bands are provided, dataset_bands must also be provided (modality: {modality})"
                raise Exception(msg)  # noqa: PLE0101

        self.filter_indices = {}
        # There is a special condition if the bands are defined as simple strings.
        if self.output_bands:
            for m in self.output_bands.keys():
                if len(set(self.output_bands[m]) & set(self.dataset_bands[m])) != len(self.output_bands[m]):
                    msg = f"Output bands must be a subset of dataset bands (Modality: {m})"
                    raise Exception(msg)
                if self.output_bands[m] == self.dataset_bands[m]:
                    continue

                self.filter_indices[m] = [self.dataset_bands[m].index(band) for band in self.output_bands[m]]

        # TODO: Implement multi-modal transforms
        # If no transform is given, apply only to transform to torch tensor
        self.transform = transform if transform else default_transform

        import warnings
        import rasterio
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        output = {}
        for modality, file in self.samples[index].items():
            data = self._load_file(
                file, nan_replace=self.no_label_replace if modality == 'mask' else self.no_data_replace).to_numpy()

            # Expand temporal dim
            if modality in self.filter_indices and self.expand_temporal_dimension:
                data = rearrange(data, "(channels time) h w -> channels time h w",
                                 channels=len(self.dataset_bands[modality]))

            if modality == 'mask':
                data = data[0]

            # TODO: Assumes all modalities with three dimension and more to be channel-first images
            if len(data.shape) >= 3:
                # to channels last
                data = np.moveaxis(data, -3, -1)

            if modality in self.filter_indices:
                data = data[..., self.filter_indices[modality]]

            if modality != 'mask':
                data = data.astype(np.float32) * self.constant_scale[modality]

            output[modality] = data

        if self.reduce_zero_label:
            output["mask"] -= 1
        if self.transform:
            output = self.transform(**output)
        output["filename"] = self.samples[index]

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
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        rgb_modality: str | None = None,
        rgb_indices: list[str] | None = None,
        allow_missing_modalities: bool = False,
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
            rgb_modality=rgb_modality,
            rgb_indices=rgb_indices,
            allow_missing_modalities=allow_missing_modalities,
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
        raise NotImplementedError('Code is based on the generic single-modality dataset and not yet adapted. '
                                  'Set `export TERRATORCH_NUM_VAL_PLOTS=0` before running terratorch.')

        image = sample[self.rgb_modality]
        if len(image.shape) == 5:  # TODO: Needed? Copied from generic dataest.
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


class GenericMultimodalPixelwiseRegressionDataset(GenericMultimodalDataset):
    """GenericNonGeoPixelwiseRegressionDataset"""

    def __init__(
        self,
        data_root: Path,
        label_data_root: Path,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        rgb_modality: str | None = None,
        rgb_indices: list[int] | None = None,
        allow_missing_modalities : bool = False,
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
            rgb_modality=rgb_modality,
            rgb_indices=rgb_indices,
            allow_missing_modalities=allow_missing_modalities,
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
        raise NotImplementedError('Code is based on the generic single-modality dataset and not yet adapted. '
                                  'Set `export TERRATORCH_NUM_VAL_PLOTS=0` before running terratorch.')

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

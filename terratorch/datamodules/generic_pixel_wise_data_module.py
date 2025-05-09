# Copyright contributors to the Terratorch project

"""
This module contains generic data modules for instantiation at runtime.
"""
import os
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import albumentations as A
import kornia.augmentation as K
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
from kornia.augmentation import AugmentationSequential

from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import GenericNonGeoPixelwiseRegressionDataset, GenericNonGeoSegmentationDataset, HLSBands
from terratorch.io.file import load_from_file_or_attribute

from .utils import check_dataset_stackability

logger = logging.getLogger("terratorch")

def wrap_in_compose_is_list(transform_list):
    # set check shapes to false because of the multitemporal case
    return A.Compose(transform_list, is_check_shapes=False) if isinstance(transform_list, Iterable) else transform_list


# def collate_fn_list_dicts(batch):
#     metadata = []
#     for i in range(len(batch)):
#         metadata.append(batch[i].pop("metadata"))
#     batch = default_collate(batch)
#     # print(len(batch["image"]), len(batch["mask"]))
#     batch["metadata"] = metadata
#     # print(len(batch["metadata"]))
#     if len(batch["image"]) != len(batch["metadata"]):
#         print(len(batch["image"]), len(batch["metadata"]))
#     return batch


class Normalize(Callable):
    def __init__(self, means, stds):
        super().__init__()
        self.means = means
        self.stds = stds

    def __call__(self, batch):
        # min_value = self.means - 2 * self.stds
        # max_value = self.means + 2 * self.stds
        # img = (batch["image"] - min_value) / (max_value - min_value)
        # img = torch.clip(img, 0, 1)
        # batch["image"] = img
        # return batch
        image = batch["image"]
        if len(image.shape) == 5:
            means = torch.tensor(self.means, device=image.device).view(1, -1, 1, 1, 1)
            stds = torch.tensor(self.stds, device=image.device).view(1, -1, 1, 1, 1)
        elif len(image.shape) == 4:
            means = torch.tensor(self.means, device=image.device).view(1, -1, 1, 1)
            stds = torch.tensor(self.stds, device=image.device).view(1, -1, 1, 1)
        else:
            msg = f"Expected batch to have 5 or 4 dimensions, but got {len(image.shape)}"
            raise Exception(msg)
        batch["image"] = (image - means) / stds
        return batch


class GenericNonGeoSegmentationDataModule(NonGeoDataModule):
    """
    This is a generic datamodule class for instantiating data modules at runtime.
    Composes several [GenericNonGeoSegmentationDatasets][terratorch.datasets.GenericNonGeoSegmentationDataset]
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_data_root: Path,
        val_data_root: Path,
        test_data_root: Path,
        means: list[float] | str,
        stds: list[float] | str,
        num_classes: int,
        img_grep: str = "*",
        label_grep: str = "*",
        predict_data_root: Path | None = None,
        train_label_data_root: Path | None = None,
        val_label_data_root: Path | None = None,
        test_label_data_root: Path | None = None,
        train_split: Path | None = None,
        val_split: Path | None = None,
        test_split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        predict_dataset_bands: list[HLSBands | int | tuple[int, int] | str ] | None = None,
        predict_output_bands: list[HLSBands | int | tuple[int, int] | str ] | None = None,
        constant_scale: float = 1,
        rgb_indices: list[int] | None = None,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        drop_last: bool = True,
        pin_memory: bool = False,
        check_stackability: bool = True,
        **kwargs: Any,
    ) -> None:
        """Constructor

        Args:
            batch_size (int): _description_
            num_workers (int): _description_
            train_data_root (Path): _description_
            val_data_root (Path): _description_
            test_data_root (Path): _description_
            predict_data_root (Path): _description_
            img_grep (str): _description_
            label_grep (str): _description_
            means (list[float]): _description_
            stds (list[float]): _description_
            num_classes (int): _description_
            train_label_data_root (Path | None, optional): _description_. Defaults to None.
            val_label_data_root (Path | None, optional): _description_. Defaults to None.
            test_label_data_root (Path | None, optional): _description_. Defaults to None.
            train_split (Path | None, optional): _description_. Defaults to None.
            val_split (Path | None, optional): _description_. Defaults to None.
            test_split (Path | None, optional): _description_. Defaults to None.
            ignore_split_file_extensions (bool, optional): Whether to disregard extensions when using the split
                file to determine which files to include in the dataset.
                E.g. necessary for Eurosat, since the split files specify ".jpg" but files are
                actually ".jpg". Defaults to True.
            allow_substring_split_file (bool, optional): Whether the split files contain substrings
                that must be present in file names to be included (as in mmsegmentation), or exact
                matches (e.g. eurosat). Defaults to True.
            dataset_bands (list[HLSBands | int] | None): Bands present in the dataset. Defaults to None.
            output_bands (list[HLSBands | int] | None): Bands that should be output by the dataset.
                Naming must match that of dataset_bands. Defaults to None.
            predict_dataset_bands (list[HLSBands | int] | None): Overwrites dataset_bands
                with this value at predict time.
                Defaults to None, which does not overwrite.
            predict_output_bands (list[HLSBands | int] | None): Overwrites output_bands
                with this value at predict time. Defaults to None, which does not overwrite.
            constant_scale (float, optional): _description_. Defaults to 1.
            rgb_indices (list[int] | None, optional): _description_. Defaults to None.
            train_transform (Albumentations.Compose | None): Albumentations transform
                to be applied to the train dataset.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            val_transform (Albumentations.Compose | None): Albumentations transform
                to be applied to the train dataset.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            test_transform (Albumentations.Compose | None): Albumentations transform
                to be applied to the train dataset.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input images with this value. If none, does no replacement. Defaults to None.
            no_label_replace (int | None): Replace nan values in label with this value. If none, does no replacement. Defaults to None.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Defaults to False.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
            drop_last (bool): Drop the last batch if it is not complete. Defaults to True.
            pin_memory (bool): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them. Defaults to False.
            check_stackability (bool): Check if all the files in the dataset has the same size and can be stacked.
        """
        super().__init__(GenericNonGeoSegmentationDataset, batch_size, num_workers, **kwargs)
        self.num_classes = num_classes
        self.img_grep = img_grep
        self.label_grep = label_grep
        self.train_root = train_data_root
        self.val_root = val_data_root
        self.test_root = test_data_root
        self.predict_root = predict_data_root
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.ignore_split_file_extensions = ignore_split_file_extensions
        self.allow_substring_split_file = allow_substring_split_file
        self.constant_scale = constant_scale
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.drop_last = drop_last
        self.pin_memory = pin_memory

        self.train_label_data_root = train_label_data_root
        self.val_label_data_root = val_label_data_root
        self.test_label_data_root = test_label_data_root

        self.dataset_bands = dataset_bands
        self.predict_dataset_bands = predict_dataset_bands if predict_dataset_bands else dataset_bands
        self.predict_output_bands = predict_output_bands if predict_output_bands else output_bands
        self.output_bands = output_bands
        self.rgb_indices = rgb_indices
        self.expand_temporal_dimension = expand_temporal_dimension
        self.reduce_zero_label = reduce_zero_label

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)

        # self.aug = AugmentationSequential(
        #     K.Normalize(means, stds),
        #     data_keys=["image"],
        # )
        means = load_from_file_or_attribute(means)
        stds = load_from_file_or_attribute(stds)

        self.aug = Normalize(means, stds)

        # self.aug = Normalize(means, stds)
        # self.collate_fn = collate_fn_list_dicts

        self.check_stackability = check_stackability
        
    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                self.train_root,
                self.num_classes,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.train_label_data_root,
                split=self.train_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.train_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                self.val_root,
                self.num_classes,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.val_label_data_root,
                split=self.val_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.val_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                self.test_root,
                self.num_classes,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.test_label_data_root,
                split=self.test_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
            )
        if stage in ["predict"] and self.predict_root:
            self.predict_dataset = self.dataset_class(
                self.predict_root,
                self.num_classes,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                dataset_bands=self.predict_dataset_bands,
                output_bands=self.predict_output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        if self.check_stackability:
            logger.info(f"Checking stackability for {split} split.")
            batch_size = check_dataset_stackability(dataset, batch_size)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
            pin_memory=self.pin_memory,
        )


class GenericNonGeoPixelwiseRegressionDataModule(NonGeoDataModule):
    """This is a generic datamodule class for instantiating data modules at runtime.
    Composes several
    [GenericNonGeoPixelwiseRegressionDataset][terratorch.datasets.GenericNonGeoPixelwiseRegressionDataset]
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_data_root: Path,
        val_data_root: Path,
        test_data_root: Path,
        means: list[float] | str,
        stds: list[float] | str,
        predict_data_root: Path | None = None,
        img_grep: str | None = "*",
        label_grep: str | None = "*",
        train_label_data_root: Path | None = None,
        val_label_data_root: Path | None = None,
        test_label_data_root: Path | None = None,
        train_split: Path | None = None,
        val_split: Path | None = None,
        test_split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str ] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str ] | None = None,
        predict_dataset_bands: list[HLSBands | int | tuple[int, int] | str ] | None = None,
        predict_output_bands: list[HLSBands | int | tuple[int, int] | str ] | None = None,
        constant_scale: float = 1,
        rgb_indices: list[int] | None = None,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        drop_last: bool = True,
        pin_memory: bool = False,
        check_stackability: bool = True,
        **kwargs: Any,
    ) -> None:
        """Constructor

        Args:
            batch_size (int): _description_
            num_workers (int): _description_
            train_data_root (Path): _description_
            val_data_root (Path): _description_
            test_data_root (Path): _description_
            predict_data_root (Path): _description_
            img_grep (str): _description_
            label_grep (str): _description_
            means (list[float]): _description_
            stds (list[float]): _description_
            train_label_data_root (Path | None, optional): _description_. Defaults to None.
            val_label_data_root (Path | None, optional): _description_. Defaults to None.
            test_label_data_root (Path | None, optional): _description_. Defaults to None.
            train_split (Path | None, optional): _description_. Defaults to None.
            val_split (Path | None, optional): _description_. Defaults to None.
            test_split (Path | None, optional): _description_. Defaults to None.
            ignore_split_file_extensions (bool, optional): Whether to disregard extensions when using the split
                file to determine which files to include in the dataset.
                E.g. necessary for Eurosat, since the split files specify ".jpg" but files are
                actually ".jpg". Defaults to True.
            allow_substring_split_file (bool, optional): Whether the split files contain substrings
                that must be present in file names to be included (as in mmsegmentation), or exact
                matches (e.g. eurosat). Defaults to True.
            dataset_bands (list[HLSBands | int] | None): Bands present in the dataset. Defaults to None.
            output_bands (list[HLSBands | int] | None): Bands that should be output by the dataset.
                Naming must match that of dataset_bands. Defaults to None.
            predict_dataset_bands (list[HLSBands | int] | None): Overwrites dataset_bands
                with this value at predict time.
                Defaults to None, which does not overwrite.
            predict_output_bands (list[HLSBands | int] | None): Overwrites output_bands
                with this value at predict time. Defaults to None, which does not overwrite.
            constant_scale (float, optional): _description_. Defaults to 1.
            rgb_indices (list[int] | None, optional): _description_. Defaults to None.
            train_transform (Albumentations.Compose | None): Albumentations transform
                to be applied to the train dataset.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            val_transform (Albumentations.Compose | None): Albumentations transform
                to be applied to the train dataset.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            test_transform (Albumentations.Compose | None): Albumentations transform
                to be applied to the train dataset.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input images with this value. If none, does no replacement. Defaults to None.
            no_label_replace (int | None): Replace nan values in label with this value. If none, does no replacement. Defaults to None.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Defaults to False.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
            drop_last (bool): Drop the last batch if it is not complete. Defaults to True.
            pin_memory (bool): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them. Defaults to False.
            check_stackability (bool): Check if all the files in the dataset has the same size and can be stacked.
        """
        super().__init__(GenericNonGeoPixelwiseRegressionDataset, batch_size, num_workers, **kwargs)
        self.img_grep = img_grep
        self.label_grep = label_grep
        self.train_root = train_data_root
        self.val_root = val_data_root
        self.test_root = test_data_root
        self.predict_root = predict_data_root
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.ignore_split_file_extensions = ignore_split_file_extensions
        self.allow_substring_split_file = allow_substring_split_file
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.expand_temporal_dimension = expand_temporal_dimension
        self.reduce_zero_label = reduce_zero_label

        self.train_label_data_root = train_label_data_root
        self.val_label_data_root = val_label_data_root
        self.test_label_data_root = test_label_data_root

        self.constant_scale = constant_scale

        self.dataset_bands = dataset_bands
        self.predict_dataset_bands = predict_dataset_bands if predict_dataset_bands else dataset_bands
        self.predict_output_bands = predict_output_bands if predict_output_bands else output_bands
        self.output_bands = output_bands
        self.rgb_indices = rgb_indices

        # self.aug = AugmentationSequential(
        #     K.Normalize(means, stds),
        #     data_keys=["image"],
        # )
        means = load_from_file_or_attribute(means)
        stds = load_from_file_or_attribute(stds)

        self.aug = Normalize(means, stds)
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)

        self.check_stackability = check_stackability

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                self.train_root,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.train_label_data_root,
                split=self.train_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.train_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                self.val_root,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.val_label_data_root,
                split=self.val_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.val_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                self.test_root,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.test_label_data_root,
                split=self.test_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
            )

        if stage in ["predict"] and self.predict_root:
            self.predict_dataset = self.dataset_class(
                self.predict_root,
                image_grep=self.img_grep,
                dataset_bands=self.predict_dataset_bands,
                output_bands=self.predict_output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
            )
       
    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        if self.check_stackability:
            logger.info("Checking stackability.")
            batch_size = check_dataset_stackability(dataset, batch_size)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
            pin_memory=self.pin_memory,
        )

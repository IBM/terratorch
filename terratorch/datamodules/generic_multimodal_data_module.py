# Copyright contributors to the Terratorch project

"""
This module contains generic data modules for instantiation at runtime.
"""
import os
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
from torchgeo.transforms import AugmentationSequential

from terratorch.datasets import (GenericMultimodalDataset, GenericMultimodalSegmentationDataset,
                                 GenericMultimodalPixelwiseRegressionDataset, HLSBands)
from terratorch.io.file import load_from_file_or_attribute


def collate_chunk_dicts(batch_list):
    batch = {}
    for key, value in batch_list[0].items():  # TODO: Handle missing modalities when is allow_missing_modalities set.
        if isinstance(value, torch.Tensor):
            batch[key] = torch.concat([chunk[key] for chunk in batch_list])
        else:
            batch[key] = [chunk[key] for chunk in batch_list]
    return batch


def wrap_in_compose_is_list(transform_list):
    # set check shapes to false because of the multitemporal case
    return A.Compose(transform_list, is_check_shapes=False) if isinstance(transform_list, Iterable) else transform_list


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


class GenericMultiModalDataModule(NonGeoDataModule):
    """
    This is a generic datamodule class for instantiating data modules at runtime.
    Composes several [GenericNonGeoSegmentationDatasets][terratorch.datasets.GenericNonGeoSegmentationDataset]
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        modalities: list[str],
        task: str | None = None,
        num_classes: int | None = None,
        label_grep: str | None = None,
        train_label_data_root: Path | None = None,
        val_label_data_root: Path | None = None,
        test_label_data_root: Path | None = None,
        predict_data_root: Path | None = None,
        train_split: Path | None = None,
        val_split: Path | None = None,
        test_split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True, # TODO: Check if covered
        predict_dataset_bands: list[HLSBands | int | tuple[int, int] | str ] | None = None,
        predict_output_bands: list[HLSBands | int | tuple[int, int] | str ] | None = None,
        predict_modality: str | None = None,
        rgb_modality: str | None = None,
        rgb_indices: list[int] | None = None,
        # TODO: Check how to handle transforms
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        drop_last: bool = True,
        chunk_data: bool = False,
        **kwargs: Any,
    ) -> None:
        """Constructor

        Args:
            # TODO: Update docs
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
        """
        if task == 'segmentation':
            dataset_class = GenericMultimodalSegmentationDataset
        elif task == 'regression':
            dataset_class = GenericMultimodalPixelwiseRegressionDataset
        elif task is None:
            dataset_class = GenericMultimodalDataset
        else:
            raise ValueError(f'Unknown task {task}, only segmentation and regression are supported.')

        super().__init__(dataset_class, batch_size, num_workers, **kwargs)
        self.num_classes = num_classes
        self.modalities = modalities
        self.img_grep = {m: kwargs.get('{m}_grep', '*') for m in modalities}
        self.label_grep = label_grep
        self.train_root = {m: kwargs[f'train_{m}_data_root'] for m in modalities}
        self.val_root = {m: kwargs[f'val_{m}_data_root'] for m in modalities}
        self.test_root = {m: kwargs[f'test_{m}_data_root'] for m in modalities}
        self.train_label_data_root = train_label_data_root
        self.val_label_data_root = val_label_data_root
        self.test_label_data_root = test_label_data_root
        self.predict_root = predict_data_root
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.ignore_split_file_extensions = ignore_split_file_extensions
        self.allow_substring_split_file = allow_substring_split_file
        self.constant_scale = {m: kwargs.get(f'{m}_constant_scale', 1.) for m in modalities}
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.drop_last = drop_last

        self.dataset_bands = {m: kwargs.get(f'{m}_dataset_bands') for m in modalities if f'{m}_dataset_bands' in kwargs}
        self.output_bands = {m: kwargs.get(f'{m}_output_bands') for m in modalities if f'{m}_output_bands' in kwargs}

        self.predict_dataset_bands = predict_dataset_bands or self.dataset_bands[predict_modality] \
            if predict_modality in self.dataset_bands else None
        self.predict_output_bands = predict_output_bands or self.output_bands[predict_modality] \
            if predict_modality in self.output_bands else None

        self.rgb_modality = rgb_modality or modalities[0]
        self.rgb_indices = rgb_indices
        self.expand_temporal_dimension = expand_temporal_dimension
        self.reduce_zero_label = reduce_zero_label

        assert train_transform is None, "transforms are not implemented yet" # TODO Handle transforms
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)

        # self.aug = AugmentationSequential(
        #     K.Normalize(means, stds),
        #     data_keys=["image"],
        # )
        means = {m: load_from_file_or_attribute(kwargs[f'{m}_means']) for m in modalities}
        stds = {m: load_from_file_or_attribute(kwargs[f'{m}_stds']) for m in modalities}

        self.aug = {m: Normalize(means[m], stds[m]) for m in modalities}

        self.chunk_data = chunk_data
        if chunk_data:
            self.collate_fn = collate_chunk_dicts
        # self.collate_fn = collate_fn_list_dicts

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                data_root=self.train_root,
                num_classes=self.num_classes,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.train_label_data_root,
                split=self.train_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_modality=self.rgb_modality,
                rgb_indices=self.rgb_indices,
                transform=self.train_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                data_root=self.val_root,
                num_classes=self.num_classes,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.val_label_data_root,
                split=self.val_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_modality=self.rgb_modality,
                rgb_indices=self.rgb_indices,
                transform=self.val_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                data_root=self.test_root,
                num_classes=self.num_classes,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.test_label_data_root,
                split=self.test_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_modality=self.rgb_modality,
                rgb_indices=self.rgb_indices,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
            )
        if stage in ["predict"] and self.predict_root:
            self.predict_dataset = self.dataset_class(
                data_root=self.predict_root,
                num_classes=self.num_classes,
                dataset_bands=self.predict_dataset_bands,
                output_bands=self.predict_output_bands,
                constant_scale=self.constant_scale,
                rgb_modality=self.rgb_modality,
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
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
        )

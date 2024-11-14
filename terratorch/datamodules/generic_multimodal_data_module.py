# Copyright contributors to the Terratorch project

"""
This module contains generic data modules for instantiation at runtime.
"""
import os
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Iterator

import albumentations as A
import kornia.augmentation as K
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, SequentialSampler
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential

from terratorch.datasets import (GenericMultimodalDataset, GenericMultimodalSegmentationDataset,
                                 GenericMultimodalPixelwiseRegressionDataset, HLSBands)
from terratorch.io.file import load_from_file_or_attribute


def collate_chunk_dicts(batch_list):
    batch = {}
    for key, value in batch_list[0].items():  # TODO: Handle missing modalities when allow_missing_modalities is set.
        if isinstance(value, torch.Tensor):
            batch[key] = torch.concat([chunk[key] for chunk in batch_list])
        elif isinstance(value, dict):
            batch[key] = collate_chunk_dicts([chunk[key] for chunk in batch_list])
        else:
            batch[key] = [chunk[key] for chunk in batch_list]
    return batch


def wrap_in_compose_is_list(transform_list, image_modalities=None, sequence_modalities=None):
    additional_targets = {}
    if image_modalities:
        for modality in image_modalities:
            additional_targets[modality] = 'image'
    if sequence_modalities:
        # Global label values are ignored and need to be processed separately
        for modality in sequence_modalities:
            additional_targets[modality] = "global_label"
    # set check shapes to false because of the multitemporal case
    return A.Compose(transform_list, is_check_shapes=False, additional_targets=additional_targets) \
        if isinstance(transform_list, Iterable) else transform_list


class MultimodalNormalize(Callable):
    def __init__(self, means, stds):
        super().__init__()
        self.means = means
        self.stds = stds

    def __call__(self, batch):
        for m in self.means.keys():
            if m not in batch['image']:
                continue
            image = batch['image'][m]
            if len(image.shape) == 5:
                # B, C, T, H, W
                means = torch.tensor(self.means[m], device=image.device).view(1, -1, 1, 1, 1)
                stds = torch.tensor(self.stds[m], device=image.device).view(1, -1, 1, 1, 1)
            elif len(image.shape) == 4:
                # B, C, H, W
                means = torch.tensor(self.means[m], device=image.device).view(1, -1, 1, 1)
                stds = torch.tensor(self.stds[m], device=image.device).view(1, -1, 1, 1)
            elif len(self.means[m]) == 1:
                # B, (T,) H, W
                means = torch.tensor(self.means[m], device=image.device)
                stds = torch.tensor(self.stds[m], device=image.device)
            elif len(image.shape) == 3:  # No batch dim
                # C, H, W
                means = torch.tensor(self.means[m], device=image.device).view(-1, 1, 1)
                stds = torch.tensor(self.stds[m], device=image.device).view(-1, 1, 1)
            else:
                msg = (f"Expected batch with 5 or 4 dimensions (B, C, (T,) H, W), sample with 3 dimensions (C, H, W) "
                       f"or a single channel, but got {len(image.shape)}")
                raise Exception(msg)
            batch['image'][m] = (image - means) / stds
        return batch


class MultiModalBatchSampler(BatchSampler):
    """
    Sample a defined number of modalities per batch (see sample_num_modalities and sample_replace)
    """
    def __init__(self, modalities, sample_num_modalities, sample_replace, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modalities = modalities
        self.sample_num_modalities = sample_num_modalities
        self.sample_replace = sample_replace

    def __iter__(self) -> Iterator[list[int]]:
        """
        Code similar to BatchSampler but samples tuples in the format (idx, ['m1', 'm2', ...])
        """
        # Select sampled modalities per batch
        sampled_modalities = np.random.choice(self.modalities, self.sample_num_modalities, replace=self.sample_replace)

        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [(next(sampler_iter), sampled_modalities) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = (idx, sampled_modalities)
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]


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
        train_data_root: dict,
        val_data_root: dict,
        test_data_root: dict,
        means: dict,
        stds: dict,
        task: str | None = None,
        num_classes: int | None = None,
        img_grep: str | dict | None = None,
        label_grep: str | None = None,
        train_label_data_root: Path | None = None,
        val_label_data_root: Path | None = None,
        test_label_data_root: Path | None = None,
        predict_data_root: Path | None = None,
        train_split: Path | None = None,
        val_split: Path | None = None,
        test_split: Path | None = None,
        dataset_bands: dict | None = None,
        output_bands: dict | None = None,
        predict_dataset_bands: list[HLSBands | int | tuple[int, int] | str ] | None = None,
        predict_output_bands: list[HLSBands | int | tuple[int, int] | str ] | None = None,
        image_modalities: list[str] | None = None,
        rgb_modality: str | None = None,
        rgb_indices: list[int] | None = None,
        allow_substring_split_file: bool = False,
        constant_scale: dict | float = 1.,
        train_transform: dict | A.Compose | None | list[A.BasicTransform] = None,
        val_transform: dict | A.Compose | None | list[A.BasicTransform] = None,
        test_transform: dict | A.Compose | None | list[A.BasicTransform] = None,
        shared_transforms: list | bool = True,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        drop_last: bool = True,
        pin_memory: bool = False,
        chunk_data: bool = False,
        sample_num_modalities: int | None = None,
        sample_replace: bool = False,
        channel_position: int = -3,
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
        self.image_modalities = image_modalities or modalities
        self.sequence_modalities = list(set(self.modalities) - set(image_modalities))
        if isinstance(img_grep, dict):
            self.img_grep = {m: img_grep[m] if m in img_grep else '*' for m in modalities}
        else:
            self.img_grep = {m: img_grep or '*' for m in modalities}
        self.label_grep = label_grep or '*'
        self.train_root = train_data_root
        self.val_root = val_data_root
        self.test_root = test_data_root
        self.train_label_data_root = train_label_data_root
        self.val_label_data_root = val_label_data_root
        self.test_label_data_root = test_label_data_root
        self.predict_root = predict_data_root
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.allow_substring_split_file = allow_substring_split_file
        self.constant_scale = constant_scale
        if isinstance(self.constant_scale, dict):
            # Fill in missing modalities
            self.constant_scale = {m: self.constant_scale[m] if m in self.constant_scale else 1.
                                   for m in modalities}
        else:
            # Create dict
            self.constant_scale = {m: constant_scale for m in modalities}
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.sample_num_modalities = sample_num_modalities
        self.sample_replace = sample_replace

        self.dataset_bands = dataset_bands
        self.output_bands = output_bands
        self.predict_dataset_bands = predict_dataset_bands
        self.predict_output_bands = predict_output_bands

        self.rgb_modality = rgb_modality or modalities[0]
        self.rgb_indices = rgb_indices
        self.expand_temporal_dimension = expand_temporal_dimension
        self.reduce_zero_label = reduce_zero_label
        self.channel_position = channel_position

        if isinstance(train_transform, dict):
            self.train_transform = {m: wrap_in_compose_is_list(train_transform[m]) if m in train_transform else None
                                    for m in modalities}
        elif shared_transforms:
            self.train_transform = wrap_in_compose_is_list(train_transform,
                                                           image_modalities=self.image_modalities,
                                                           sequence_modalities=self.sequence_modalities)
        else:
            self.train_transform = {m: wrap_in_compose_is_list(train_transform)
                                    for m in modalities}

        if isinstance(val_transform, dict):
            self.val_transform = {m: wrap_in_compose_is_list(val_transform[m]) if m in val_transform else None
                                    for m in modalities}
        elif shared_transforms:
            self.val_transform = wrap_in_compose_is_list(val_transform,
                                                         image_modalities=self.image_modalities,
                                                         sequence_modalities=self.sequence_modalities)
        else:
            self.val_transform = {m: wrap_in_compose_is_list(val_transform)
                                    for m in modalities}

        if isinstance(test_transform, dict):
            self.test_transform = {m: wrap_in_compose_is_list(test_transform[m]) if m in test_transform else None
                                    for m in modalities}
        elif shared_transforms:
            self.test_transform = wrap_in_compose_is_list(test_transform,
                                                          image_modalities=self.image_modalities,
                                                          sequence_modalities=self.sequence_modalities)
        else:
            self.test_transform = {m: wrap_in_compose_is_list(test_transform)
                                    for m in modalities}

        means = {m: load_from_file_or_attribute(means[m]) for m in means.keys()}
        stds = {m: load_from_file_or_attribute(stds[m]) for m in stds.keys()}

        self.aug = MultimodalNormalize(means, stds)

        self.chunk_data = chunk_data
        if chunk_data:
            self.collate_fn = collate_chunk_dicts

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                data_root=self.train_root,
                num_classes=self.num_classes,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.train_label_data_root,
                split=self.train_split,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                image_modalities=self.image_modalities,
                rgb_modality=self.rgb_modality,
                rgb_indices=self.rgb_indices,
                transform=self.train_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
                channel_position=self.channel_position,
            )
            logging.info(f'Train dataset: {len(self.train_dataset)}')
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                data_root=self.val_root,
                num_classes=self.num_classes,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.val_label_data_root,
                split=self.val_split,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                image_modalities=self.image_modalities,
                rgb_modality=self.rgb_modality,
                rgb_indices=self.rgb_indices,
                transform=self.val_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
                channel_position=self.channel_position,
            )
            logging.info(f'Val dataset: {len(self.val_dataset)}')
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                data_root=self.test_root,
                num_classes=self.num_classes,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.test_label_data_root,
                split=self.test_split,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                image_modalities=self.image_modalities,
                rgb_modality=self.rgb_modality,
                rgb_indices=self.rgb_indices,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
                channel_position=self.channel_position,
            )
            logging.info(f'Test dataset: {len(self.test_dataset)}')
        if stage in ["predict"] and self.predict_root:
            self.predict_dataset = self.dataset_class(
                data_root=self.predict_root,
                num_classes=self.num_classes,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.predict_dataset_bands,
                output_bands=self.predict_output_bands,
                constant_scale=self.constant_scale,
                image_modalities=self.image_modalities,
                rgb_modality=self.rgb_modality,
                rgb_indices=self.rgb_indices,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
                channel_position=self.channel_position,
            )
            logging.info(f'Predict dataset: {len(self.predict_dataset)}')

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
        if self.sample_num_modalities:
            # Custom batch sampler for sampling modalities per batch
            batch_sampler = MultiModalBatchSampler(
                self.modalities, self.sample_num_modalities, self.sample_replace,
                RandomSampler(dataset) if split == "train" else SequentialSampler(dataset),
                batch_size=batch_size,
                drop_last=split == "train" and self.drop_last
            )
        else:
            batch_sampler = BatchSampler(
                RandomSampler(dataset) if split == "train" else SequentialSampler(dataset),
                batch_size=batch_size,
                drop_last=split == "train" and self.drop_last
            )

        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
        )

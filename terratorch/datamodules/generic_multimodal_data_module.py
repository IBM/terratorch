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
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, SequentialSampler, default_collate
from torchgeo.datamodules import NonGeoDataModule

from terratorch.datasets import (GenericMultimodalDataset, GenericMultimodalSegmentationDataset,
                                 GenericMultimodalPixelwiseRegressionDataset, GenericMultimodalScalarDataset, HLSBands)
from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from terratorch.io.file import load_from_file_or_attribute

logger = logging.getLogger("terratorch")

def collate_chunk_dicts(batch_list):
    batch = {}
    for key, value in batch_list[0].items():  # TODO: Handle missing modalities when allow_missing_modalities is set.
        if isinstance(value, torch.Tensor):
            batch[key] = torch.concat([chunk[key] for chunk in batch_list])
        elif isinstance(value, np.ndarray):
            batch[key] = np.concatenate([chunk[key] for chunk in batch_list])
        elif isinstance(value, dict):
            batch[key] = collate_chunk_dicts([chunk[key] for chunk in batch_list])
        else:
            batch[key] = [chunk[key] for chunk in batch_list]
    return batch


def collate_samples(batch_list):
    """
    Wrapper for default_collate as it cannot handle some datatypes such as np.datetime64.
    """
    batch = {}
    for key, value in batch_list[0].items():  # TODO: Handle missing modalities when allow_missing_modalities is set.
        if isinstance(value, dict):
            batch[key] = collate_samples([chunk[key] for chunk in batch_list])
        else:
            try:
                batch[key] = default_collate([chunk[key] for chunk in batch_list])
            except TypeError:
                # Fallback to numpy or simple list
                if isinstance(value, np.ndarray):
                    batch[key] = np.stack([chunk[key] for chunk in batch_list])
                else:
                    batch[key] = [chunk[key] for chunk in batch_list]
    return batch


def wrap_in_compose_is_list(transform_list, image_modalities=None, non_image_modalities=None):
    additional_targets = {}
    if image_modalities:
        for modality in image_modalities:
            additional_targets[modality] = "image"
    if non_image_modalities:
        # Global label values are ignored and need to be processed separately
        for modality in non_image_modalities:
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
            if m not in batch["image"]:
                continue
            image = batch["image"][m]
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

            elif len(image.shape) == 2:
                means = torch.tensor(self.means[m], device=image.device)
                stds = torch.tensor(self.stds[m], device=image.device)

            elif len(image.shape) == 1:
                means = torch.tensor(self.means[m], device=image.device)
                stds = torch.tensor(self.stds[m], device=image.device)

            else:
                msg = (f"Expected batch with 5 or 4 dimensions (B, C, (T,) H, W), sample with 3 dimensions (C, H, W) "
                       f"or a single channel, but got {len(image.shape)}")
                raise Exception(msg)
            batch["image"][m] = (image - means) / stds
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
        Code similar to BatchSampler but samples tuples in the format (idx, ["m1", "m2", ...])
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
        modalities: list[str],
        train_data_root: dict[str, Path],
        val_data_root: dict[str, Path],
        test_data_root: dict[str, Path],
        means: dict[str, list],
        stds: dict[str, list],
        task: str | None = None,
        num_classes: int | None = None,
        image_grep: str | dict[str, str] | None = None,
        label_grep: str | None = None,
        train_label_data_root: Path | str | None = None,
        val_label_data_root: Path | str | None = None,
        test_label_data_root: Path | str | None = None,
        predict_data_root: dict[str, Path] | str | None = None,
        train_split: Path | str | None = None,
        val_split: Path | str | None = None,
        test_split: Path| str | None = None,
        dataset_bands: dict[str, list] | None = None,
        output_bands: dict[str, list] | None = None,
        predict_dataset_bands: dict[str, list] | None = None,
        predict_output_bands: dict[str, list] | None = None,
        image_modalities: list[str] | None = None,
        rgb_modality: str | None = None,
        rgb_indices: list[int] | None = None,
        allow_substring_file_names: bool = False,
        class_names: list[str] | None = None,
        constant_scale: dict[float] = None,
        train_transform: dict | A.Compose | None | list[A.BasicTransform] = None,
        val_transform: dict | A.Compose | None | list[A.BasicTransform] = None,
        test_transform: dict | A.Compose | None | list[A.BasicTransform] = None,
        shared_transforms: list | bool = True,
        expand_temporal_dimension: bool = False,
        no_data_replace: float | None = None,
        no_label_replace: float | None = -1,
        reduce_zero_label: bool = False,
        drop_last: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_with_sample_dim: bool = False,
        sample_num_modalities: int | None = None,
        sample_replace: bool = False,
        channel_position: int = -3,
        concat_bands: bool = False,
        **kwargs: Any,
    ) -> None:
        """Constructor

        Args:
            batch_size (int): Number of samples in per batch.
            modalities (list[str]): List of modalities.
            train_data_root (dict[Path]): Dictionary of paths to training data root directory or csv/parquet files with 
                image-level data, with modalities as keys.
            val_data_root (dict[Path]): Dictionary of paths to validation data root directory or csv/parquet files with 
                image-level data, with modalities as keys.
            test_data_root (dict[Path]): Dictionary of paths to test data root directory or csv/parquet files with 
                image-level data, with modalities as keys.
            means (dict[list]): Dictionary of mean values as lists with modalities as keys.
            stds (dict[list]): Dictionary of std values as lists with modalities as keys.
            task (str, optional): Selected task form segmentation, regression (pixel-wise), classification,
                multilabel_classification, scalar_regression, scalar (custom image-level task), or None (no targets).
                Defaults to None.
            num_classes (int, optional): Number of classes in classification or segmentation tasks.
            predict_data_root (dict[Path], optional): Dictionary of paths to data root directory or csv/parquet files
                with image-level data, with modalities as keys.
            image_grep (dict[str], optional): Dictionary with regular expression appended to data_root to find input
                images, with modalities as keys. Defaults to "*". Ignored when allow_substring_file_names is False.
            label_grep (str, optional): Regular expression appended to label_data_root to find labels or mask files.
                Defaults to "*". Ignored when allow_substring_file_names is False.
            train_label_data_root (Path | None, optional): Path to data root directory with training labels or
                csv/parquet files with labels. Required for supervised tasks.
            val_label_data_root (Path | None, optional): Path to data root directory with validation labels or
                csv/parquet files with labels. Required for supervised tasks.
            test_label_data_root (Path | None, optional): Path to data root directory with test labels or
                csv/parquet files with labels. Required for supervised tasks.
            train_split (Path, optional): Path to file containing training samples prefixes to be used for this split.
                The file can be a csv/parquet file with the prefixes in the index or a txt file with new-line separated
                sample prefixes. File names must be exact matches if allow_substring_file_names is False. Otherwise,
                files are searched using glob with the form Path(data_root).glob(prefix + [image or label grep]).
                If not specified, search samples based on files in data_root. Defaults to None.
            val_split (Path, optional): Path to file containing validation samples prefixes to be used for this split.
                The file can be a csv/parquet file with the prefixes in the index or a txt file with new-line separated
                sample prefixes. File names must be exact matches if allow_substring_file_names is False. Otherwise,
                files are searched using glob with the form Path(data_root).glob(prefix + [image or label grep]).
                If not specified, search samples based on files in data_root. Defaults to None.
            test_split (Path, optional): Path to file containing test samples prefixes to be used for this split.
                The file can be a csv/parquet file with the prefixes in the index or a txt file with new-line separated
                sample prefixes. File names must be exact matches if allow_substring_file_names is False. Otherwise,
                files are searched using glob with the form Path(data_root).glob(prefix + [image or label grep]).
                If not specified, search samples based on files in data_root. Defaults to None.
            dataset_bands (dict[list], optional): Bands present in the dataset, provided in a dictionary with modalities
                as keys. This parameter names input channels (bands) using HLSBands, ints, int ranges, or strings, so
                that they can then be referred to by output_bands. Needs to be superset of output_bands. Can be a subset
                of all modalities. Defaults to None.
            output_bands (dict[list], optional): Bands that should be output by the dataset as named by dataset_bands,
                provided as a dictionary with modality keys. Can be subset of all modalities. Defaults to None.
            predict_dataset_bands (list[dict], optional): Overwrites dataset_bands with this value at predict time.
                Defaults to None, which does not overwrite.
            predict_output_bands (list[dict], optional): Overwrites output_bands with this value at predict time.
                Defaults to None, which does not overwrite.
            image_modalities(list[str], optional): List of pixel-level raster modalities. Defaults to data_root.keys().
                The difference between all modalities and image_modalities are non-image modalities which are treated
                differently during the transforms and are not modified but only converted into a tensor if possible.
            rgb_modality (str, optional): Modality used for RGB plots. Defaults to first modality in data_root.keys().
            rgb_indices (list[int] | None, optional): _description_. Defaults to None.
            allow_substring_file_names
            class_names (list[str], optional): Names of the classes. Defaults to None.
            constant_scale (dict[float]): Factor to multiply data values by, provided as a dictionary with modalities as
                keys. Can be subset of all modalities. Defaults to None.
            train_transform (Albumentations.Compose | dict | None): Albumentations transform to be applied to all image
                modalities. Should end with ToTensorV2() and not include normalization. The transform is not applied to 
                non-image data, which is only converted to tensors if possible. If dict, can include separate transforms 
                per modality (no shared parameters between modalities). 
                Defaults to None, which simply applies ToTensorV2().
            val_transform (Albumentations.Compose | dict | None): Albumentations transform to be applied to all image
                modalities. Should end with ToTensorV2() and not include normalization. The transform is not applied to 
                non-image data, which is only converted to tensors if possible. If dict, can include separate transforms 
                per modality (no shared parameters between modalities). 
                Defaults to None, which simply applies ToTensorV2().
            test_transform (Albumentations.Compose | dict | None): Albumentations transform to be applied to all image
                modalities. Should end with ToTensorV2() and not include normalization. The transform is not applied to 
                non-image data, which is only converted to tensors if possible. If dict, can include separate transforms 
                per modality (no shared parameters between modalities). 
                Defaults to None, which simply applies ToTensorV2().
            shared_transforms (bool): transforms are shared between all image modalities (e.g., similar crop). 
                This setting is ignored if transforms are defined per modality. Defaults to True.  
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Only works with image modalities. Is only applied to modalities with defined dataset_bands.
                Defaults to False.
            no_data_replace (float | None): Replace nan values in input images with this value. If none, does no 
                replacement. Defaults to None.
            no_label_replace (int | None): Replace nan values in label with this value. If none, does no replacement. 
                Defaults to None.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
            drop_last (bool): Drop the last batch if it is not complete. Defaults to True.
            num_workers (int): Number of parallel workers. Defaults to 0 for single threaded process.
            pin_memory (bool): If ``True``, the data loader will copy Tensors into device/CUDA pinned memory before 
                returning them. Defaults to False.
            data_with_sample_dim (bool): Use a specific collate function to concatenate samples along a existing sample
                dimension instead of stacking the samples. Defaults to False.
            sample_num_modalities (int, optional): Load only a subset of modalities per batch. Defaults to None.
            sample_replace (bool): If sample_num_modalities is set, sample modalities with replacement.
                Defaults to False.
            channel_position (int): Position of the channel dimension in the image modalities. Defaults to -3.
            concat_bands (bool): Concatenate all image modalities along the band dimension into a single "image", so
                that it can be processed by single-modal models. Concatenate in the order of provided modalities.
                Works with image modalities only. Does not work with allow_missing_modalities. Defaults to False.
        """
        if task == "segmentation":
            dataset_class = GenericMultimodalSegmentationDataset
        elif task == "regression":
            dataset_class = GenericMultimodalPixelwiseRegressionDataset
        elif task in ["classification", "multilabel_classification", "scalar_regression", "scalar"]:
            dataset_class = GenericMultimodalScalarDataset
            task = "scalar"
        elif task is None:
            dataset_class = GenericMultimodalDataset
        else:
            raise ValueError(f"Unknown task {task}, only segmentation and regression are supported.")

        super().__init__(dataset_class, batch_size, num_workers, **kwargs)
        self.num_classes = num_classes
        self.class_names = class_names
        self.modalities = modalities
        self.image_modalities = image_modalities or modalities
        self.non_image_modalities = list(set(self.modalities) - set(self.image_modalities))
        if task == "scalar":
            self.non_image_modalities += ["label"]
        if isinstance(image_grep, dict):
            self.image_grep = {m: image_grep[m] if m in image_grep else "*" for m in modalities}
        else:
            self.image_grep = {m: image_grep or "*" for m in modalities}
        self.label_grep = label_grep or "*"
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
        self.allow_substring_file_names = allow_substring_file_names
        self.constant_scale = constant_scale
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
        self.concat_bands = concat_bands

        if isinstance(train_transform, dict):
            self.train_transform = {m: wrap_in_compose_is_list(train_transform[m]) if m in train_transform else None
                                    for m in modalities}
        elif shared_transforms:
            self.train_transform = wrap_in_compose_is_list(train_transform,
                                                           image_modalities=self.image_modalities,
                                                           non_image_modalities=self.non_image_modalities)
        else:
            self.train_transform = {m: wrap_in_compose_is_list(train_transform)
                                    for m in modalities}

        if isinstance(val_transform, dict):
            self.val_transform = {m: wrap_in_compose_is_list(val_transform[m]) if m in val_transform else None
                                    for m in modalities}
        elif shared_transforms:
            self.val_transform = wrap_in_compose_is_list(val_transform,
                                                         image_modalities=self.image_modalities,
                                                         non_image_modalities=self.non_image_modalities)
        else:
            self.val_transform = {m: wrap_in_compose_is_list(val_transform)
                                    for m in modalities}

        if isinstance(test_transform, dict):
            self.test_transform = {m: wrap_in_compose_is_list(test_transform[m]) if m in test_transform else None
                                    for m in modalities}
        elif shared_transforms:
            self.test_transform = wrap_in_compose_is_list(test_transform,
                                                          image_modalities=self.image_modalities,
                                                          non_image_modalities=self.non_image_modalities,                                                          
                                                          )
        else:
            self.test_transform = {m: wrap_in_compose_is_list(test_transform)
                                    for m in modalities}

        if self.concat_bands:
            # Concatenate mean and std values
            means = load_from_file_or_attribute(np.concatenate([means[m] for m in self.image_modalities]).tolist())
            stds = load_from_file_or_attribute(np.concatenate([stds[m] for m in self.image_modalities]).tolist())

            self.aug = Normalize(means, stds)
        else:
            # Apply standardization per modality
            means = {m: load_from_file_or_attribute(means[m]) for m in means.keys()}
            stds = {m: load_from_file_or_attribute(stds[m]) for m in stds.keys()}

            self.aug = MultimodalNormalize(means, stds)

        self.data_with_sample_dim = data_with_sample_dim

        self.collate_fn = collate_chunk_dicts if data_with_sample_dim else collate_samples


    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                data_root=self.train_root,
                num_classes=self.num_classes,
                image_grep=self.image_grep,
                label_grep=self.label_grep,
                label_data_root=self.train_label_data_root,
                split=self.train_split,
                allow_substring_file_names=self.allow_substring_file_names,
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
                concat_bands=self.concat_bands ,
            )
            logger.info(f"Train dataset: {len(self.train_dataset)}")
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                data_root=self.val_root,
                num_classes=self.num_classes,
                image_grep=self.image_grep,
                label_grep=self.label_grep,
                label_data_root=self.val_label_data_root,
                split=self.val_split,
                allow_substring_file_names=self.allow_substring_file_names,
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
                concat_bands=self.concat_bands,
            )
            logger.info(f"Val dataset: {len(self.val_dataset)}")
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                data_root=self.test_root,
                num_classes=self.num_classes,
                image_grep=self.image_grep,
                label_grep=self.label_grep,
                label_data_root=self.test_label_data_root,
                split=self.test_split,
                allow_substring_file_names=self.allow_substring_file_names,
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
                concat_bands=self.concat_bands,
            )
            logger.info(f"Test dataset: {len(self.test_dataset)}")
        if stage in ["predict"] and self.predict_root:
            self.predict_dataset = self.dataset_class(
                data_root=self.predict_root,
                num_classes=self.num_classes,
                allow_substring_file_names=self.allow_substring_file_names,
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
                concat_bands=self.concat_bands,
            )
            logger.info(f"Predict dataset: {len(self.predict_dataset)}")

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either "train", "val", "test", or "predict".

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

from pathlib import Path
from typing import Any

import albumentations as A
import torch

from terratorch.datasets import (
    GenericNonGeoPixelwiseRegressionDataset,
    GenericNonGeoSegmentationDataModule,
    HLSBands,
)
from terratorch.io.file import load_from_file_or_attribute


def wrap_in_compose_is_list(transform_list):
    # Set check_shapes to False because of the multi-temporal case
    return A.Compose(transform_list, is_check_shapes=False) if isinstance(transform_list, list) else transform_list


class Normalize:
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds

    def __call__(self, batch):
        image = batch["image"]
        if len(image.shape) == 5:
            # Shape: (batch_size, channels, time, height, width)
            means = torch.tensor(self.means, device=image.device).view(1, -1, 1, 1, 1)
            stds = torch.tensor(self.stds, device=image.device).view(1, -1, 1, 1, 1)
        elif len(image.shape) == 4:
            # Shape: (batch_size, channels, height, width)
            means = torch.tensor(self.means, device=image.device).view(1, -1, 1, 1)
            stds = torch.tensor(self.stds, device=image.device).view(1, -1, 1, 1)
        else:
            msg = f"Expected image to have 4 or 5 dimensions, but got {len(image.shape)}"
            raise Exception(msg)
        batch["image"] = (image - means) / stds
        return batch


class MultiTemporalCropClassificationDataModule(GenericNonGeoSegmentationDataModule):
    """
    DataModule for multi-temporal crop classification.
    Inherits from GenericNonGeoSegmentationDataModule and modifies only what's necessary.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_data_root: Path,
        val_data_root: Path,
        test_data_root: Path,
        img_grep: str,
        label_column: str,
        means: list[float] | str,
        stds: list[float] | str,
        predict_data_root: Path | None = None,
        train_split: Path | None = None,
        val_split: Path | None = None,
        test_split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        predict_dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        predict_output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        constant_scale: float = 1,
        rgb_indices: list[int] | None = None,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        expand_temporal_dimension: bool = False,
        no_data_replace: float | None = None,
        drop_last: bool = True,
        metadata_file: Path | str | None = None,
        metadata_columns: list[str] | None = None,
        metadata_index_col: str | None = "chip_id",
        filename_regex: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            train_data_root=train_data_root,
            val_data_root=val_data_root,
            test_data_root=test_data_root,
            predict_data_root=predict_data_root,
            img_grep=img_grep,
            label_grep=None,
            means=means,
            stds=stds,
            num_classes=None,
            train_label_data_root=None,
            val_label_data_root=None,
            test_label_data_root=None,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            ignore_split_file_extensions=ignore_split_file_extensions,
            allow_substring_split_file=allow_substring_split_file,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            predict_dataset_bands=predict_dataset_bands,
            predict_output_bands=predict_output_bands,
            constant_scale=constant_scale,
            rgb_indices=rgb_indices,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            expand_temporal_dimension=expand_temporal_dimension,
            reduce_zero_label=False,
            no_data_replace=no_data_replace,
            no_label_replace=None,
            drop_last=drop_last,
            **kwargs,
        )

        # Specify the dataset_class as GenericNonGeoPixelwiseRegressionDataset
        self.dataset_class = GenericNonGeoPixelwiseRegressionDataset

        self.label_column = label_column
        self.metadata_file = metadata_file
        self.metadata_columns = metadata_columns
        self.metadata_index_col = metadata_index_col
        self.filename_regex = filename_regex

        # Load means and standard deviations for normalization
        self.means = load_from_file_or_attribute(means)
        self.stds = load_from_file_or_attribute(stds)
        self.aug = Normalize(self.means, self.stds)

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                data_root=self.train_root,
                label_column=self.label_column,
                image_grep=self.img_grep,
                split=self.train_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.train_transform,
                no_data_replace=self.no_data_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                metadata_file=self.metadata_file,
                metadata_columns=self.metadata_columns,
                metadata_index_col=self.metadata_index_col,
                filename_regex=self.filename_regex,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                data_root=self.val_root,
                label_column=self.label_column,
                image_grep=self.img_grep,
                split=self.val_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.val_transform,
                no_data_replace=self.no_data_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                metadata_file=self.metadata_file,
                metadata_columns=self.metadata_columns,
                metadata_index_col=self.metadata_index_col,
                filename_regex=self.filename_regex,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                data_root=self.test_root,
                label_column=self.label_column,
                image_grep=self.img_grep,
                split=self.test_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                metadata_file=self.metadata_file,
                metadata_columns=self.metadata_columns,
                metadata_index_col=self.metadata_index_col,
                filename_regex=self.filename_regex,
            )
        if stage in ["predict"] and self.predict_root:
            self.predict_dataset = self.dataset_class(
                data_root=self.predict_root,
                label_column=self.label_column,
                image_grep=self.img_grep,
                dataset_bands=self.predict_dataset_bands,
                output_bands=self.predict_output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                metadata_file=self.metadata_file,
                metadata_columns=self.metadata_columns,
                metadata_index_col=self.metadata_index_col,
                filename_regex=self.filename_regex,
            )

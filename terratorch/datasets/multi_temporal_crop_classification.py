from datetime import datetime
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torchgeo.datasets import GenericNonGeoSegmentationDataset

from terratorch.datasets.utils import HLSBands


class MultiTemporalCropClassification(GenericNonGeoSegmentationDataset):
    """
    Dataset class for multi-temporal crop classification.
    Inherits from GenericNonGeoSegmentationDataset and adds temporal coordinates support.
    """

    def __init__(
        self,
        data_root: Path,
        label_column: str,
        image_grep: str = "*",
        split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        rgb_indices: list[int] | None = None,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        constant_scale: float = 1,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        expand_temporal_dimension: bool = False,
        metadata_file: Path | str | None = None,
        metadata_columns: list[str] | None = None,
        metadata_index_col: str | None = "chip_id",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data_root=data_root,
            label_data_root=None,
            image_grep=image_grep,
            label_grep=None,
            split=split,
            ignore_split_file_extensions=ignore_split_file_extensions,
            allow_substring_split_file=allow_substring_split_file,
            rgb_indices=rgb_indices,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            no_label_replace=None,
            expand_temporal_dimension=expand_temporal_dimension,
        )

        self.label_column = label_column

        if metadata_file:
            self.metadata_file = metadata_file
            self.metadata_columns = metadata_columns
            self.metadata_index_col = metadata_index_col
            self._load_metadata()

    def _load_metadata(self):
        """Load metadata CSV file."""
        self.metadata_df = pd.read_csv(self.metadata_file)

        if self.metadata_index_col not in self.metadata_df.columns:
            msg = f"Metadata file must contain column '{self.metadata_index_col}'"
            raise ValueError(msg)

        if self.label_column not in self.metadata_df.columns:
            msg = f"Metadata file must contain column '{self.label_column}'"
            raise ValueError(msg)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data_df.iloc[index]
        image_file = row["image_file"]
        label = row[self.label_column]

        image = self._load_file(image_file, nan_replace=self.no_data_replace).to_numpy()

        # Handle temporal dimension if required
        if self.expand_temporal_dimension:
            image = self._expand_temporal_dimension(image)
        else:
            image = np.moveaxis(image, 0, -1)  # Move channels to the last dimension

        if self.filter_indices:
            image = image[..., self.filter_indices]

        output = {
            "image": image.astype(np.float32) * self.constant_scale,
            "label": label,
        }

        if self.transform:
            output = self.transform(**output)

        output["filename"] = image_file

        if self.metadata_file:
            temporal_coords = self._get_temporal_coords(row)
            output["temporal_coords"] = temporal_coords

        if self.metadata_columns:
            metadata = row[self.metadata_columns].to_dict()
            output.update(metadata)

        return output

    def _get_temporal_coords(self, row) -> torch.Tensor:
        """Extract and format temporal coordinates (year, day_of_year) from metadata."""
        date_columns = ["first_img_date", "middle_img_date", "last_img_date"]
        temporal_coords = []
        for col in date_columns:
            date_str = row[col]
            date = datetime.strptime(date_str, "%Y-%m-%d")  # noqa: DTZ007
            year = date.year
            day_of_year = date.timetuple().tm_yday
            temporal_coords.append([year, day_of_year])
        temporal_coords = np.array(temporal_coords, dtype=np.float32)
        return torch.tensor(temporal_coords)

    def _expand_temporal_dimension(self, image: np.ndarray) -> np.ndarray:
        """Reshape the image to separate the temporal dimension if required."""
        channels = len(self.output_bands) if self.output_bands else image.shape[0]
        time_steps = image.shape[0] // channels
        return image.reshape((channels, time_steps, *image.shape[1:]))

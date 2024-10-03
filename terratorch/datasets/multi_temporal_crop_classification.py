from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from einops import rearrange
import albumentations as A

from terratorch.datasets import GenericNonGeoSegmentationDataset
from terratorch.datasets.utils import HLSBands


class MultiTemporalCropClassification(GenericNonGeoSegmentationDataset):
    """
    Dataset class for multi-temporal crop classification.
    Inherits from GenericNonGeoSegmentationDataset and adds temporal coordinates support.
    """

    def __init__(
        self,
        data_root: Path,
        num_classes: int,
        image_grep: str = "*_merged.tif",
        label_grep: str = "*.mask.tif",
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
        reduce_zero_label: bool = False,
    ) -> None:
        super().__init__(
            data_root=data_root,
            num_classes=num_classes,
            label_data_root=data_root,
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
            no_label_replace=None,
            expand_temporal_dimension=expand_temporal_dimension,
        )

        self.metadata_file = metadata_file
        self.metadata_columns = metadata_columns
        self.metadata_index_col = metadata_index_col
        self.reduce_zero_label = reduce_zero_label

        if metadata_file:
            self._load_metadata()
            self._build_image_metadata_mapping()
        else:
            self.metadata_df = None

    def _load_metadata(self):
        """Load metadata CSV file."""
        self.metadata_df = pd.read_csv(self.metadata_file)

        if self.metadata_index_col not in self.metadata_df.columns:
            msg = f"Metadata file must contain column '{self.metadata_index_col}'"
            raise ValueError(msg)

    def _build_image_metadata_mapping(self):
        """Build a mapping from image filenames to metadata indices."""
        self.image_to_metadata_index = {}

        for idx, image_file in enumerate(self.image_files):
            image_filename = Path(image_file).name
            image_id = image_filename.replace("_merged.tif", "").replace(".tif", "")
            metadata_indices = self.metadata_df.index[
                self.metadata_df[self.metadata_index_col] == image_id
            ].tolist()
            self.image_to_metadata_index[idx] = metadata_indices[0]


    def __getitem__(self, index: int) -> dict[str, Any]:
        image_file = self.image_files[index]
        mask_file = str(image_file).replace("_merged.tif", ".mask.tif")

        image = self._load_file(image_file, nan_replace=self.no_data_replace).to_numpy().transpose(1, 2, 0)
        mask = self._load_file(mask_file, nan_replace=self.no_label_replace).to_numpy()[0]

        output = {
            "image": image.astype(np.float32) * self.constant_scale,
            "mask": mask,
        }

        if self.reduce_zero_label:
            output["mask"] -= 1

        if self.transform:
            output = self.transform(**output)

        output["mask"] = output["mask"].long()
        output["filename"] = str(image_file)

        if self.metadata_df is not None:
            metadata_idx = self.image_to_metadata_index.get(index, None)
            if metadata_idx is not None:
                row = self.metadata_df.iloc[metadata_idx]
                temporal_coords = self._get_temporal_coords(row)
                output["temporal_coords"] = temporal_coords

                if self.metadata_columns:
                    metadata = row[self.metadata_columns].to_dict()
                    output.update(metadata)

        return output

    def _get_temporal_coords(self, row) -> torch.Tensor:
        """Extract and format temporal coordinates (T, date) from metadata."""
        date_columns = ["first_img_date", "middle_img_date", "last_img_date"]
        temporal_coords = []
        for idx, col in enumerate(date_columns):
            date_str = row[col]
            date = datetime.strptime(date_str, "%Y-%m-%d")  # noqa: DTZ007
            temporal_coords.append([idx, date.timestamp()])
        temporal_coords = np.array(temporal_coords, dtype=np.float32)
        return torch.from_numpy(temporal_coords)

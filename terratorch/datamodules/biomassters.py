from collections.abc import Sequence
from typing import Any

import albumentations as A
from torch.utils.data import DataLoader

from terratorch.datamodules.generic_multimodal_data_module import MultimodalNormalize, wrap_in_compose_is_list
from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from terratorch.datasets import BioMasstersNonGeo
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential

MEANS = {
    "AGBM": 63.4584,
    "S1": {
        "VV_Asc": 0.08871397,
        "VH_Asc": 0.02172604,
        "VV_Desc": 0.08556002,
        "VH_Desc": 0.02795591,
        "RVI_Asc": 0.75507677,
        "RVI_Desc": 0.6600374
    },
    "S2": {
        "BLUE": 1633.0802,
        "GREEN": 1610.0035,
        "RED": 1599.557,
        "RED_EDGE_1": 1916.7083,
        "RED_EDGE_2": 2478.8325,
        "RED_EDGE_3": 2591.326,
        "NIR_BROAD": 2738.5837,
        "NIR_NARROW": 2685.8281,
        "SWIR_1": 1023.90204,
        "SWIR_2": 696.48755,
        "CLOUD_PROBABILITY": 21.177078
    }
}

STDS = {
    "AGBM": 72.21242,
    "S1": {
        "VV_Asc": 0.16714208,
        "VH_Asc": 0.04876742,
        "VV_Desc": 0.19260046,
        "VH_Desc": 0.10272296,
        "RVI_Asc": 0.24945821,
        "RVI_Desc": 0.3590119
    },
    "S2": {
        "BLUE": 2499.7146,
        "GREEN": 2308.5298,
        "RED": 2388.2268,
        "RED_EDGE_1": 2389.6375,
        "RED_EDGE_2": 2209.6467,
        "RED_EDGE_3": 2104.572,
        "NIR_BROAD": 2194.209,
        "NIR_NARROW": 2031.7762,
        "SWIR_1": 934.0556,
        "SWIR_2": 759.8444,
        "CLOUD_PROBABILITY": 49.352486
    }
}

class BioMasstersNonGeoDataModule(NonGeoDataModule):
    """NonGeo datamodule implementation for BioMassters."""

    default_metadata_filename = "The_BioMassters_-_features_metadata.csv.csv"

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 0,
        bands: dict[str, Sequence[str]] | Sequence[str] = BioMasstersNonGeo.all_band_names,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        predict_transform: A.Compose | None | list[A.BasicTransform] = None,
        aug: AugmentationSequential = None,
        drop_last: bool = True,
        sensors: Sequence[str] = ["S1", "S2"],
        as_time_series: bool = False,
        metadata_filename: str = default_metadata_filename,
        max_cloud_percentage: float | None = None,
        max_red_mean: float | None = None,
        include_corrupt: bool = True,
        subset: float = 1,
        seed: int = 42,
        use_four_frames: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(BioMasstersNonGeo, batch_size, num_workers, **kwargs)
        self.data_root = data_root
        self.sensors = sensors
        if isinstance(bands, dict):
            self.bands = bands
        else:
            sens = sensors[0]
            self.bands = {sens: bands}

        self.means = {}
        self.stds = {}
        for sensor in self.sensors:
            self.means[sensor] = [MEANS[sensor][band] for band in self.bands[sensor]]
            self.stds[sensor] = [STDS[sensor][band] for band in self.bands[sensor]]

        self.mask_mean = MEANS["AGBM"]
        self.mask_std = STDS["AGBM"]
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.predict_transform = wrap_in_compose_is_list(predict_transform)
        if len(sensors) == 1:
            self.aug = Normalize(self.means[sensors[0]], self.stds[sensors[0]]) if aug is None else aug
        else:
            MultimodalNormalize(self.means, self.stds) if aug is None else aug
        self.drop_last = drop_last
        self.as_time_series = as_time_series
        self.metadata_filename = metadata_filename
        self.max_cloud_percentage = max_cloud_percentage
        self.max_red_mean = max_red_mean
        self.include_corrupt = include_corrupt
        self.subset = subset
        self.seed = seed
        self.use_four_frames = use_four_frames

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train",
                root=self.data_root,
                transform=self.train_transform,
                bands=self.bands,
                mask_mean=self.mask_mean,
                mask_std=self.mask_std,
                sensors=self.sensors,
                as_time_series=self.as_time_series,
                metadata_filename=self.metadata_filename,
                max_cloud_percentage=self.max_cloud_percentage,
                max_red_mean=self.max_red_mean,
                include_corrupt=self.include_corrupt,
                subset=self.subset,
                seed=self.seed,
                use_four_frames=self.use_four_frames,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="test",
                root=self.data_root,
                transform=self.val_transform,
                bands=self.bands,
                mask_mean=self.mask_mean,
                mask_std=self.mask_std,
                sensors=self.sensors,
                as_time_series=self.as_time_series,
                metadata_filename=self.metadata_filename,
                max_cloud_percentage=self.max_cloud_percentage,
                max_red_mean=self.max_red_mean,
                include_corrupt=self.include_corrupt,
                subset=self.subset,
                seed=self.seed,
                use_four_frames=self.use_four_frames,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test",
                root=self.data_root,
                transform=self.test_transform,
                bands=self.bands,
                mask_mean=self.mask_mean,
                mask_std=self.mask_std,
                sensors=self.sensors,
                as_time_series=self.as_time_series,
                metadata_filename=self.metadata_filename,
                max_cloud_percentage=self.max_cloud_percentage,
                max_red_mean=self.max_red_mean,
                include_corrupt=self.include_corrupt,
                subset=self.subset,
                seed=self.seed,
                use_four_frames=self.use_four_frames,
            )
        if stage in ["predict"]:
            self.predict_dataset = self.dataset_class(
                split="test",
                root=self.data_root,
                transform=self.predict_transform,
                bands=self.bands,
                mask_mean=self.mask_mean,
                mask_std=self.mask_std,
                sensors=self.sensors,
                as_time_series=self.as_time_series,
                metadata_filename=self.metadata_filename,
                max_cloud_percentage=self.max_cloud_percentage,
                max_red_mean=self.max_red_mean,
                include_corrupt=self.include_corrupt,
                subset=self.subset,
                seed=self.seed,
                use_four_frames=self.use_four_frames,
            )

    def _dataloader_factory(self, split: str):
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split =="train" and self.drop_last,
        )

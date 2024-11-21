# Copyright contributors to the Terratorch project

from collections.abc import Sequence
from typing import Any

import albumentations as A
import kornia.augmentation as K  # noqa: N812
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential

from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import Sen1Floods11NonGeo

MEANS = {
    "COASTAL_AEROSOL": 0.16450718,
    "BLUE": 0.1412956,
    "GREEN": 0.13795798,
    "RED": 0.12353792,
    "RED_EDGE_1": 0.1481099,
    "RED_EDGE_2": 0.23991728,
    "RED_EDGE_3": 0.28587557,
    "NIR_BROAD": 0.26345379,
    "NIR_NARROW": 0.30902815,
    "WATER_VAPOR": 0.04911151,
    "CIRRUS": 0.00652506,
    "SWIR_1": 0.2044958,
    "SWIR_2": 0.11912015,
}

STDS = {
    "COASTAL_AEROSOL": 0.06977374,
    "BLUE": 0.07406382,
    "GREEN": 0.07370365,
    "RED": 0.08692279,
    "RED_EDGE_1": 0.07778555,
    "RED_EDGE_2": 0.09105416,
    "RED_EDGE_3": 0.10690993,
    "NIR_BROAD": 0.10096586,
    "NIR_NARROW": 0.11798815,
    "WATER_VAPOR": 0.03380113,
    "CIRRUS": 0.01463465,
    "SWIR_1": 0.09772074,
    "SWIR_2": 0.07659938,
}

class Sen1Floods11NonGeoDataModule(NonGeoDataModule):
    """NonGeo Fire Scars data module implementation"""

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 0,
        bands: Sequence[str] = Sen1Floods11NonGeo.all_band_names,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        drop_last: bool = True,
        constant_scale: float = 0.0001,
        no_data_replace: float | None = 0,
        no_label_replace: int | None = -1,
        use_metadata: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(Sen1Floods11NonGeo, batch_size, num_workers, **kwargs)
        self.data_root = data_root

        means = [MEANS[b] for b in bands]
        stds = [STDS[b] for b in bands]
        self.bands = bands,
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.aug = AugmentationSequential(K.Normalize(means, stds), data_keys=["image"])
        self.drop_last = drop_last
        self.constant_scale = constant_scale
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.use_metadata = use_metadata

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train",
                data_root=self.data_root,
                transform=self.train_transform,
                bands=self.bands,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                use_metadata=self.use_metadata,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                transform=self.val_transform,
                bands=self.bands,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                use_metadata=self.use_metadata,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test",
                data_root=self.data_root,
                transform=self.test_transform,
                bands=self.bands,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                use_metadata=self.use_metadata,
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

from collections.abc import Callable, Sequence
from typing import Any

import albumentations as A
from kornia.augmentation import AugmentationSequential
from torchgeo.datamodules import NonGeoDataModule

from terratorch.datamodules.generic_multimodal_data_module import wrap_in_compose_is_list
from terratorch.datasets import HelioNetCDFDataset


class ByPassNormalize(Callable):
    def __init__(self):
        super().__init__()

    def __call__(self, batch):
        return batch


class HelioNetCDFDataModule(NonGeoDataModule):
    """NonGeo LightningDataModule implementation for the Heliophysics datamodule."""

    def __init__(
        self,
        train_index_path: str = None,
        val_index_path: str = None,
        test_index_path: str = None,
        predict_index_path: str = None,
        batch_size: int = 4,
        num_workers: int = 0,
        aug: AugmentationSequential = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(HelioNetCDFDataset, batch_size, num_workers, **kwargs)

        self.train_index_path = train_index_path
        self.val_index_path = val_index_path
        self.test_index_path = test_index_path
        self.predict_index_path = predict_index_path

        self.aug = ByPassNormalize() if aug is None else aug
        self.extra_arguments = kwargs

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either fit, validate, test, or predict.
        """
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                phase="train",
                index_path=self.train_index_path,
                **self.extra_arguments,
            )

        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(phase="test", index_path=self.val_index_path, **self.extra_arguments)

        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                phase="test", index_path=self.test_index_path, **self.extra_arguments
            )

        if stage in ["predict"]:
            self.predict_dataset = self.dataset_class(
                phase="predict",
                index_path=self.predict_index_path,
                **self.extra_arguments,
            )

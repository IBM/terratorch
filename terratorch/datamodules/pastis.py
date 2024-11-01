from typing import Any

import albumentations as A  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule

from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import PASTIS


class PASTISDataModule(NonGeoDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        data_root: str = "./",
        truncate_image: int | None = None,
        pad_image: int | None = None,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            PASTIS,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        self.truncate_image = truncate_image
        self.pad_image = pad_image
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.data_root = data_root
        self.kwargs = kwargs

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = PASTIS(
                folds=[1, 2, 3],
                data_root=self.data_root,
                transform=self.train_transform,
                truncate_image=self.truncate_image,
                pad_image=self.pad_image,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = PASTIS(
                folds=[4],
                data_root=self.data_root,
                transform=self.val_transform,
                truncate_image=self.truncate_image,
                pad_image=self.pad_image,
                **self.kwargs,
            )
        if stage in ["test"]:
            self.test_dataset = PASTIS(
                folds=[5],
                data_root=self.data_root,
                transform=self.test_transform,
                truncate_image=self.truncate_image,
                pad_image=self.pad_image,
                **self.kwargs,
            )

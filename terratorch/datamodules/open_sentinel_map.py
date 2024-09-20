from typing import Any

import albumentations as A  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule

from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import OpenSentinelMap


class OpenSentinelMapDataModule(NonGeoDataModule):
    def __init__(
        self,
        bands: list[str] | None = None,
        batch_size: int = 8,
        num_workers: int = 0,
        data_root: str = "./",
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        spatial_interpolate_and_stack_temporally: bool = True,  # noqa: FBT001, FBT002
        pad_image: int | None = None,
        truncate_image: int | None = None,
        target: int = 0,
        pick_random_pair: bool = True,  # noqa: FBT002, FBT001
        **kwargs: Any,
    ) -> None:
        super().__init__(
            OpenSentinelMap,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        self.bands = bands
        self.spatial_interpolate_and_stack_temporally = spatial_interpolate_and_stack_temporally
        self.pad_image = pad_image
        self.truncate_image = truncate_image
        self.target = target
        self.pick_random_pair = pick_random_pair
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.data_root = data_root
        self.kwargs = kwargs

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = OpenSentinelMap(
                split="train",
                data_root=self.data_root,
                transform=self.train_transform,
                bands=self.bands,
                spatial_interpolate_and_stack_temporally = self.spatial_interpolate_and_stack_temporally,
                pad_image = self.pad_image,
                truncate_image = self.truncate_image,
                target = self.target,
                pick_random_pair = self.pick_random_pair,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = OpenSentinelMap(
                split="val",
                data_root=self.data_root,
                transform=self.val_transform,
                bands=self.bands,
                spatial_interpolate_and_stack_temporally = self.spatial_interpolate_and_stack_temporally,
                pad_image = self.pad_image,
                truncate_image = self.truncate_image,
                target = self.target,
                pick_random_pair = self.pick_random_pair,
                **self.kwargs,
            )
        if stage in ["test"]:
            self.test_dataset = OpenSentinelMap(
                split="test",
                data_root=self.data_root,
                transform=self.test_transform,
                bands=self.bands,
                spatial_interpolate_and_stack_temporally = self.spatial_interpolate_and_stack_temporally,
                pad_image = self.pad_image,
                truncate_image = self.truncate_image,
                target = self.target,
                pick_random_pair = self.pick_random_pair,
                **self.kwargs,
            )

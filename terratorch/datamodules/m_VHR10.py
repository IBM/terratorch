from torchgeo.datasets.utils import (
    Path,
    check_integrity,
    download_and_extract_archive,
    download_url,
    lazy_import,
    percentile_normalization,
)

from collections.abc import Callable
from typing import Any, ClassVar

from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import patches
from functools import partial

from terratorch.datasets import mVHR10

from torch.utils.data._utils.collate import default_collate
from torchgeo.datamodules import NonGeoDataModule

import albumentations as A
from albumentations.pytorch import transforms as T

from torch.utils.data import DataLoader

import torch

import numpy as np

def custom_collate(batch):
    
    images = [item['image'] for item in batch]
    bboxes = [item['boxes'] for item in batch]
    masks = [item['masks'] for item in batch]
    
    # Apply transforms to each sample in the batch
    transformed = [transforms(image, {'boxes': bbox, 'masks': mask}) for image, bbox, mask in zip(images, bboxes, masks)]
    
    # Separate transformed images and targets
    transformed_images = [t[0] for t in transformed]
    transformed_targets = [t[1] for t in transformed]
    
    # Collate the transformed data
    collated_images = default_collate(transformed_images)
    collated_targets = [{k: default_collate([d[k] for d in transformed_targets]) for k in transformed_targets[0]}]
    
    return {'image': collated_images, 'bbox': collated_targets[0]['boxes'], 'mask': collated_targets[0]['masks'], 'labels': batch['labels']}


def get_transform(train):
    transforms = []
    if train:
        transforms.append(A.RandomCrop(width=224, height=224))
        transforms.append(A.HorizontalFlip(p=0.5))
    else:
        transforms.append(A.CenterCrop(width=224, height=224))
    transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transforms.append(A.ToFloat())
    transforms.append(T.ToTensorV2())
    # return A.Compose(transforms, additional_targets={'boxes': 'bboxes', 'masks': 'mask'})
    return A.Compose(transforms, bbox_params=A.BboxParams(format="pascal_voc", label_fields=['labels']), is_check_shapes=False)

def apply_transforms(sample, transforms):
    
    sample['image']=sample['image'].permute(1, 2, 0)
    transformed = transforms(image=np.array(sample["image"]), masks=np.array(sample["masks"]), bboxes=np.array(sample["boxes"]), labels=np.array(sample["labels"]))
    transformed['boxes'] = torch.tensor(transformed['bboxes'])
    transformed['labels'] = torch.tensor(transformed['labels'], dtype=torch.int8)
    del transformed['bboxes']

    return transformed


class mVHR10DataModule(NonGeoDataModule):
    def __init__(
        self,
        root: Path = 'data',
        split: str = 'positive',
        train_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = partial(apply_transforms,transforms=get_transform(True)),
        val_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = partial(apply_transforms,transforms=get_transform(False)),
        test_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = partial(apply_transforms,transforms=get_transform(False)),
        download: bool = False,
        checksum: bool = False,
        second_level_split="train",
        second_level_split_proportions = (0.7, 0.15, 0.15),
        batch_size: int = 4,
        num_workers: int = 0,
        collate_fn = None,
        *args,
        **kwargs):

        super().__init__(mVHR10,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         root=root, 
                         split=split,
                         download=download, 
                         checksum=checksum,
                         second_level_split=second_level_split,
                         second_level_split_proportions=second_level_split_proportions,
                         **kwargs)

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.root = root
        self.split = split
        self.second_level_split = second_level_split
        self.second_level_split_proportions = second_level_split_proportions
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = self.collate_fn if collate_fn is None else collate_fn
        self.download = download
        self.checksum = checksum


    def setup(self, stage: str) -> None:

        if stage in ["fit"]:
            self.train_dataset = mVHR10(
                root = self.root,
                split = self.split, 
                transforms = self.train_transform,
                download = self.download, 
                checksum = self.checksum,
                second_level_split="train",
                second_level_split_proportions = self.second_level_split_proportions,
            )            
        if stage in ["fit", "validate"]:
            self.val_dataset = mVHR10(
                root = self.root,
                split = self.split, 
                transforms = self.train_transform,
                download = self.download, 
                checksum = self.checksum,
                second_level_split="val",
                second_level_split_proportions = self.second_level_split_proportions,
            )
        if stage in ["test"]:
            self.test_dataset = mVHR10(
                root = self.root,
                split = self.split, 
                transforms = self.train_transform,
                download = self.download, 
                checksum = self.checksum,
                second_level_split="test",
                second_level_split_proportions = self.second_level_split_proportions,
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
        batch_size = self.batch_size
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )


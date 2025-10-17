from typing import Optional

import lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CocoDetection

from terratorch.datasets.od_aed_elephant import ElephantCocoDataset
from terratorch.datasets.od_tiled_dataset_wrapper import TiledDataset


class ElephantDataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_folder_train: str,
        ann_file_train: str,
        img_folder_val: str,
        ann_file_val: str,
        min_size: tuple = (5472, 3648),
        tile_size: tuple = (512, 512),
        overlap: int = 128,
        batch_size: int = 8,
        num_workers: int = 8,
        train_test_split: float = 0.8,
        tile_cache_test: str | None = "tile_cache_test",
        tile_cache_train: str | None = "tile_cache_train",
    ):
        if not 0.0 <= train_test_split <= 1.0:
            raise ValueError(f"train_test_split must be between 0 and 1, got {train_test_split}")

        super().__init__()

        self.dataset_test = TiledDataset(
            base_dataset=ElephantCocoDataset(img_folder=img_folder_val, ann_file=ann_file_val),
            min_size=min_size,
            tile_size=tile_size,
            overlap=overlap,
            cache_dir=tile_cache_test,
            skip_empty_boxes=False,
        )

        train_val_dataset = TiledDataset(
            base_dataset=ElephantCocoDataset(img_folder=img_folder_train, ann_file=ann_file_train),
            min_size=min_size,
            tile_size=tile_size,
            overlap=overlap,
            cache_dir=tile_cache_train,
        )

        train_size = int(train_test_split * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size

        self.dataset_train, self.dataset_val = random_split(train_val_dataset, [train_size, val_size])

        self.batch_size = batch_size
        self.num_workers = num_workers

        # Basic transforms (resize + normalize)
        self.train_transform = T.Compose(
            [
                T.ToTensor(),
                """ T.Resize((512, 512)),   # adjust if needed
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) """,
            ]
        )

    def detection_collate_fn(self, batch):
        """
        Collate function returning:
        - images: Tensor [B,C,H,W]
        - boxes: List[Tensor[N_i,4]]
        - labels: List[Tensor[N_i]]
        """
        # images: stack along batch dimension
        images = torch.stack([b["image"] for b in batch])  # [B,C,H,W]

        # boxes and labels: keep as list of tensors (variable N_i)
        boxes = [b["boxes"] for b in batch]
        labels = [b["labels"] for b in batch]

        return {
            "image": images,
            "boxes": boxes,
            "labels": labels,
        }

    def setup(self, stage: str | None = None):
        # nothing to do
        pass

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.detection_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.detection_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.detection_collate_fn,
        )

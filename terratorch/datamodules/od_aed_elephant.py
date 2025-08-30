from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
import lightning as pl
from torch.utils.data import Dataset
from terratorch.datasets.od_tiled_dataset_wrapper import TiledDataset
from terratorch.datasets.od_aed_elephant import ElephantCocoDataset


class ElephantDataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        min_size: tuple = (5472,3648),
        tile_size: tuple = (512, 512),
        overlap: int = 128,
        batch_size: int = 8,
        num_workers: int = 8,
        save_png: bool = False
    ):       
        super().__init__()
        self.dataset = TiledDataset(
            base_dataset=ElephantCocoDataset(img_folder=img_folder, ann_file=ann_file),
            min_size=min_size,
            tile_size=tile_size,
            overlap=overlap,
            save_png=save_png,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Basic transforms (resize + normalize)
        self.train_transform = T.Compose([
            T.ToTensor(),
            """ T.Resize((512, 512)),   # adjust if needed
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) """
        ])

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





    def setup(self, stage: Optional[str] = None):
        # TODO rkie implement train/val split
        self.dataset_train = self.dataset
        self.dataset_val = self.dataset


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


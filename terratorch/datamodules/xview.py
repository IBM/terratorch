import os
import logging
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms

try:
    from terratorch.datasets.xview import XviewDataset
except ImportError as e:
    logging.error(
        "\n[ERROR] XviewDataset not found.\n"
        "Please copy the xView-PyTorch dataset file to your environment:\n"
        "    wget https://raw.githubusercontent.com/akihironitta/xView-PyTorch/refs/heads/master/datasets.py\n"
        "    mv datasets.py path_to_environment/terratorch/datasets/xview.py\n"
    )
    raise e


class XviewDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, ann_file, batch_size=8, num_workers=4, img_transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.ann_file = ann_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_transform = img_transform or transforms.ToTensor()

    def setup(self, stage=None):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not os.path.isfile(self.ann_file):
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")

        self.xv_dataset = XviewDataset(
            root=self.data_dir,
            annFile=self.ann_file,
            transform=self.img_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.xv_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.xv_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.xv_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

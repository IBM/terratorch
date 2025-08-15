import os
import logging
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from torchvision import transforms
import torch
import torch.nn.functional as F
from torchvision.models.detection.transform import GeneralizedRCNNTransform



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


class XviewDataModule(LightningDataModule):

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


 #   def detection_collate(self, batch):
 #       images = [item[0] for item in batch]
 #       annots_batch = [item[1] for item in batch]#

        # pad images to max H, W
#       max_h = max(img.shape[1] for img in images)
 #       max_w = max(img.shape[2] for img in images)
  #      padded_images = []
   #     for img in images:
    #        c, h, w = img.shape
     #       pad = (0, max_w - w, 0, max_h - h)  # (left, right, top, bottom)
      #      padded_images.append(F.pad(img, pad, value=0.0))
       # images_tensor = torch.stack(padded_images, dim=0)  # [B,C,H,W]

#        boxes_list = []
#        labels_list = []

#        for annots in annots_batch:
#            boxes, labels = [], []
#            for obj in annots:
#                if isinstance(obj, dict):
#                    x1, y1, x2, y2 = map(float, obj['bbox'].split(','))
#                    type_id = int(obj['type_id'])
#                elif isinstance(obj, str):
#                    parts = list(map(float, obj.split(',')))  # x1,y1,x2,y2,type_id
#                    x1, y1, x2, y2, type_id = parts
#                else:
#                    raise TypeError(f"Unexpected annotation type: {type(obj)}")
#                boxes.append([x1, y1, x2, y2])
#                labels.append(type_id)#
#
#            boxes_list.append(
#                torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
#            )
#            labels_list.append(
#                torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
#            )
#
#        return {
#            'image': images_tensor,  # [B,C,Hmax,Wmax]
#            'boxes': boxes_list,     # list of tensors per image
#            'labels': labels_list    # list of tensors per image
#        }


    def detection_collate(self, batch):
        """
        Collate function for TorchVision object detection datasets.
        Returns:
            images: Tensor [B,C,Hmax,Wmax]
            targets: List[dict], each dict has 'boxes' [N,4] and 'labels' [N]
        """
        images = [item[0] for item in batch]
        annots_batch = [item[1] for item in batch]

        # pad images to max H,W
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        padded_images = []
        for img in images:
            c, h, w = img.shape
            pad = (0, max_w - w, 0, max_h - h)  # left, right, top, bottom
            padded_images.append(F.pad(img, pad, value=0.0))
        images_tensor = torch.stack(padded_images, dim=0)  # [B,C,Hmax,Wmax]

        # prepare targets
        targets = []
        #for annots in annots_batch:
        #    boxes = torch.tensor([
        #        list(map(float, obj['bbox'].split(','))) for obj in annots
        #    ], dtype=torch.float32) if annots else torch.zeros((0, 4), dtype=torch.float32)
#
 #           labels = torch.tensor([
  #              int(obj['type_id']) for obj in annots
   #         ], dtype=torch.int64) if annots else torch.zeros((0,), dtype=torch.int64)
#
 #           targets.append({
  #              'boxes': boxes,
   #             'labels': labels
    #        })


        return { 'image': images_tensor, 'boxes': [d["boxes"] for d in annots_batch], 'labels': [d["labels"] for d in annots_batch] }



    def train_dataloader(self):
        return DataLoader(
            self.xv_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.detection_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.xv_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.detection_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.xv_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.detection_collate,
        )

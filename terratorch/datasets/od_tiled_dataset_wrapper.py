import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image


class TiledDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        min_size=(1, 1),
        tile_size=(512, 512),
        overlap: int = 0,
    ):
        """
        Args:
            base_dataset: dataset returning (image, target) where
                image is a PIL.Image or tensor (C, H, W),
                target is a dict with 'boxes' in [x1,y1,x2,y2] format.
            min_size: (min_h, min_w). Skip images smaller than this.
            tile_size: (tile_h, tile_w).
            overlap: overlap in pixels (applied both H and W).
        """
        self.base_dataset = base_dataset
        self.min_h, self.min_w = min_size
        self.tile_h, self.tile_w = tile_size
        self.overlap = overlap

        self.tiles = []
        for idx in range(len(base_dataset)):
            print('preprocessing image:', idx)
            img= base_dataset[idx]['image']
            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
            else:  
                raise RuntimeError(f'Only supported torch.Tensor (b,c,h,w), but got {type(img)}')

            # skip if smaller than required min size
            if h < self.min_h or w < self.min_w:
                print(' image too small, skipping:', idx, h, w)
                continue
            print(' image large enough:', idx)

            step_h = self.tile_h - self.overlap
            step_w = self.tile_w - self.overlap

            # only keep tiles that fully fit into image (no remainder)
            for y0 in range(0, h - self.tile_h + 1, step_h):
                for x0 in range(0, w - self.tile_w + 1, step_w):
                    print('appending tile')
                    self.tiles.append((idx, x0, y0))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        base_idx, x0, y0 = self.tiles[idx]
        img = self.base_dataset[base_idx]['image']
        boxes = self.base_dataset[base_idx]['boxes']
        labels = self.base_dataset[base_idx]['labels']

        if isinstance(img, torch.Tensor):
            c, h, w = img.shape
        else:
            w, h = img.size

        # crop tile
        tile = F.crop(img, top=y0, left=x0,
                      height=self.tile_h, width=self.tile_w)

        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes, dtype=torch.float32)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # shift relative to crop
        x1 = x1 - x0
        x2 = x2 - x0
        y1 = y1 - y0
        y2 = y2 - y0
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        # keep only boxes that intersect tile
        # compute intersection area with tile
        inter_x1 = boxes[:, 0].clamp(0, self.tile_w)
        inter_y1 = boxes[:, 1].clamp(0, self.tile_h)
        inter_x2 = boxes[:, 2].clamp(0, self.tile_w)
        inter_y2 = boxes[:, 3].clamp(0, self.tile_h)
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # original box area
        area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

        # keep only if overlap fraction > threshold
        overlap_ratio = inter_area / (area + 1e-6)
        keep = overlap_ratio > 0.2   # e.g. require at least 20% of box inside tile

        boxes = boxes[keep]
        labels = labels[keep]

        # clip to tile boundaries
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, self.tile_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, self.tile_h)

        return {"image": tile, "boxes": boxes, "labels": labels, "image_id": torch.Tensor(idx).int()}
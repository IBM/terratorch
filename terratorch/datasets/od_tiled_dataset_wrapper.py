import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image

class TiledDataset(Dataset):
    def __init__(self, base_dataset: Dataset, tile_size=512, overlap=0):
        """
        Args:
            base_dataset: dataset returning (image, target) where
                image is a PIL.Image or tensor (H, W, C),
                target is a dict with 'boxes' in [x1,y1,x2,y2] format.
            tile_size: size of square tiles (default 512)
            overlap: overlap between tiles in pixels
        """
        self.base_dataset = base_dataset
        self.tile_size = tile_size
        self.overlap = overlap

        # Precompute mapping: (base_idx, tile_row, tile_col)
        self.tiles = []
        for idx in range(len(base_dataset)):
            img, target = base_dataset[idx]
            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
            else:  # PIL
                w, h = img.size

            step = tile_size - overlap
            for y0 in range(0, h, step):
                for x0 in range(0, w, step):
                    self.tiles.append((idx, x0, y0))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        base_idx, x0, y0 = self.tiles[idx]
        img, target = self.base_dataset[base_idx]

        if isinstance(img, torch.Tensor):
            c, h, w = img.shape
        else:
            w, h = img.size

        # crop tile
        tile = F.crop(img, top=y0, left=x0,
                      height=self.tile_size, width=self.tile_size)

        # adjust boxes
        boxes = target["boxes"].clone() if isinstance(target["boxes"], torch.Tensor) else torch.tensor(target["boxes"])
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]

        # shift relative to crop
        x1 = x1 - x0
        x2 = x2 - x0
        y1 = y1 - y0
        y2 = y2 - y0
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        # keep only boxes that intersect tile
        keep = (boxes[:,2] > 0) & (boxes[:,0] < self.tile_size) & \
               (boxes[:,3] > 0) & (boxes[:,1] < self.tile_size)
        boxes = boxes[keep]

        # clip to tile boundaries
        boxes[:,0::2] = boxes[:,0::2].clamp(0, self.tile_size)
        boxes[:,1::2] = boxes[:,1::2].clamp(0, self.tile_size)

        new_target = {**target, "boxes": boxes}

        return tile, new_target

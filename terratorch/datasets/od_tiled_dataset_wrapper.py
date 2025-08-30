import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image


class TiledDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        cache_dir: str = "./tile_cache",
        min_size=(1, 1),
        tile_size=(512, 512),
        overlap: int = 0,
        save_png: bool = False,   # optional for debugging
    ):
        """
        Args:
            base_dataset: dataset returning dict {"image","boxes","labels"}
            min_size: (min_h, min_w). Skip images smaller than this.
            tile_size: (tile_h, tile_w).
            overlap: overlap in pixels (applied both H and W).
            cache_dir: folder where tiles are cached to disk.
            save_png: whether to also save .png images for inspection.
        """
        self.base_dataset = base_dataset
        self.cache_dir = cache_dir
        self.min_h, self.min_w = min_size
        self.tile_h, self.tile_w = tile_size
        self.overlap = overlap
        self.save_png = save_png

        os.makedirs(cache_dir, exist_ok=True)

        # cache index file
        self.index_file = os.path.join(cache_dir, "index.pt")

        if os.path.exists(self.index_file):
            print(f"[TiledDataset] Loading cache index from {self.index_file}")
            self.tiles = torch.load(self.index_file)
        else:
            print("[TiledDataset] Building tiles...")
            self.tiles = []
            for idx in range(len(base_dataset)):
                print(' preprocessing image:', idx)
                img = base_dataset[idx]['image']
                if isinstance(img, torch.Tensor):
                    h, w = img.shape[-2:]
                else:
                    raise RuntimeError(f'Only torch.Tensor supported, got {type(img)}')

                if h < self.min_h or w < self.min_w:
                    print(' image too small, skipping:', idx, h, w)
                    continue

                step_h = self.tile_h - self.overlap
                step_w = self.tile_w - self.overlap

                for y0 in range(0, h - self.tile_h + 1, step_h):
                    for x0 in range(0, w - self.tile_w + 1, step_w):
                        self.tiles.append((idx, x0, y0))

            torch.save(self.tiles, self.index_file)
            print(f"[TiledDataset] Saved tile index with {len(self.tiles)} tiles")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        cache_path = os.path.join(self.cache_dir, f"tile_{idx}.pt")
        if os.path.exists(cache_path):
            return torch.load(cache_path)

        # not cached yet → generate & save
        base_idx, x0, y0 = self.tiles[idx]
        sample = self.base_dataset[base_idx]
        img, boxes, labels = sample['image'], sample['boxes'], sample['labels']

        if isinstance(img, torch.Tensor):
            c, h, w = img.shape
        else:
            raise RuntimeError("Only torch.Tensor supported")

        tile = F.crop(img, top=y0, left=x0,
                      height=self.tile_h, width=self.tile_w)
        tile = tile.detach().contiguous().clone()   # <-- makes its own storage (mitigtes memory leak)

        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes, dtype=torch.float32)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # shift relative to crop
        x1 = x1 - x0
        x2 = x2 - x0
        y1 = y1 - y0
        y2 = y2 - y0
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        # intersection
        inter_x1 = boxes[:, 0].clamp(0, self.tile_w)
        inter_y1 = boxes[:, 1].clamp(0, self.tile_h)
        inter_x2 = boxes[:, 2].clamp(0, self.tile_w)
        inter_y2 = boxes[:, 3].clamp(0, self.tile_h)
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

        overlap_ratio = inter_area / (area + 1e-6)
        keep = overlap_ratio > 0.2

        boxes = boxes[keep]
        labels = labels[keep]

        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, self.tile_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, self.tile_h)

        sample_out = {
            "image": tile,
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(idx, dtype=torch.int32),
        }

        tmp_path = cache_path + ".tmp"
        torch.save(sample_out, tmp_path)
        os.replace(tmp_path, cache_path)   # atomic rename on POSIX
        torch.save(sample_out, cache_path)

        if self.save_png:
            from torchvision.utils import save_image
            save_image(tile, cache_path.replace(".pt", ".png"))

        return sample_out

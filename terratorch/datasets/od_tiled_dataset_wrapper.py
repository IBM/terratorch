import os
import json
from glob import glob
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.utils import save_image
from PIL import Image


class TiledDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        cache_dir: str = "./tile_cache",
        min_size: Tuple[int, int] = (1, 1),
        tile_size: Tuple[int, int] = (512, 512),
        overlap: int = 0,
        rebuild: bool = False,  # force regeneration
    ):
        """
        Args:
            base_dataset: dataset returning dict {"image","boxes","labels"}
            cache_dir: folder where tiles are cached as PNG + JSON files.
            min_size: (min_h, min_w). Skip images smaller than this.
            tile_size: (tile_h, tile_w).
            overlap: overlap in pixels (applied both H and W).
            rebuild: force regeneration of tiles even if cache exists.
        """
        self.base_dataset = base_dataset
        self.cache_dir = cache_dir
        self.min_h, self.min_w = min_size
        self.tile_h, self.tile_w = tile_size
        self.overlap = overlap

        os.makedirs(cache_dir, exist_ok=True)

        # find cached tiles
        self.tiles: List[str] = sorted(glob(os.path.join(cache_dir, "*.png")))

        if not self.tiles or rebuild:
            print("[TiledDataset] Building tiles...")
            self.tiles = []
            step_h = self.tile_h - self.overlap
            step_w = self.tile_w - self.overlap

            for idx in range(len(base_dataset)):
                print(" preprocessing image:", idx)
                sample = base_dataset[idx]
                img, boxes, labels = sample["image"], sample["boxes"], sample["labels"]

                if not isinstance(img, torch.Tensor):
                    raise RuntimeError(f"Only torch.Tensor supported, got {type(img)}")

                c, h, w = img.shape
                if h < self.min_h or w < self.min_w:
                    print(" image too small, skipping:", idx, h, w)
                    continue

                for y0 in range(0, h - self.tile_h + 1, step_h):
                    for x0 in range(0, w - self.tile_w + 1, step_w):
                        # crop image
                        tile = F.crop(img, top=y0, left=x0,
                                      height=self.tile_h, width=self.tile_w).clone()

                        # shift boxes
                        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                        x1, x2 = x1 - x0, x2 - x0
                        y1, y2 = y1 - y0, y2 - y0
                        box_shifted = torch.stack([x1, y1, x2, y2], dim=1)

                        # clip + filter boxes
                        inter_x1 = box_shifted[:, 0].clamp(0, self.tile_w)
                        inter_y1 = box_shifted[:, 1].clamp(0, self.tile_h)
                        inter_x2 = box_shifted[:, 2].clamp(0, self.tile_w)
                        inter_y2 = box_shifted[:, 3].clamp(0, self.tile_h)
                        inter_w = (inter_x2 - inter_x1).clamp(min=0)
                        inter_h = (inter_y2 - inter_y1).clamp(min=0)
                        inter_area = inter_w * inter_h
                        area = (box_shifted[:, 2] - box_shifted[:, 0]).clamp(min=0) * \
                               (box_shifted[:, 3] - box_shifted[:, 1]).clamp(min=0)
                        overlap_ratio = inter_area / (area + 1e-6)
                        keep = overlap_ratio > 0.2

                        box_shifted = box_shifted[keep]
                        label_shifted = labels[keep]

                        # clamp final
                        box_shifted[:, 0::2] = box_shifted[:, 0::2].clamp(0, self.tile_w)
                        box_shifted[:, 1::2] = box_shifted[:, 1::2].clamp(0, self.tile_h)

                        # file names
                        fname = f"tile_{idx}_{x0}_{y0}"
                        f_img = os.path.join(cache_dir, fname + ".png")
                        f_json = os.path.join(cache_dir, fname + ".json")

                        # save image
                        save_image(tile, f_img)

                        # save metadata
                        meta = {
                            "boxes": box_shifted.tolist(),
                            "labels": label_shifted.tolist(),
                            "image_id": int(idx),
                        }
                        with open(f_json, "w") as f:
                            json.dump(meta, f)

                        self.tiles.append(f_img)

            print(f"[TiledDataset] Saved {len(self.tiles)} tiles to {cache_dir}")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        f_img = self.tiles[idx]
        f_json = f_img.replace(".png", ".json")

        img = Image.open(f_img).convert("RGB")
        img_tensor = F.to_tensor(img)

        with open(f_json, "r") as f:
            meta = json.load(f)

        sample_out = {
            "image": img_tensor,
            "boxes": torch.tensor(meta["boxes"], dtype=torch.float32),
            "labels": torch.tensor(meta["labels"], dtype=torch.long),
            "image_id": torch.tensor(meta["image_id"], dtype=torch.int32),
        }
        return sample_out

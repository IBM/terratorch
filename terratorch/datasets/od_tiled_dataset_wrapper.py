import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image


def atomic_write_image(tensor, path):
    """Write image to disk atomically (via temp + rename)."""
    tmp = path + ".tmp"
    img = F.to_pil_image(tensor.cpu())
    img.save(tmp, format="PNG")
    os.replace(tmp, path)  # atomic rename


def atomic_write_json(data, path):
    """Write JSON to disk atomically (via temp + rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)  # atomic rename


class TiledDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        cache_dir: str = "tile_cache",
        tile_size=(512, 512),
        overlap=0,
        min_size=(1, 1),
        rebuild=False,
    ):
        self.base_dataset = base_dataset
        self.cache_dir = cache_dir
        self.tile_h, self.tile_w = tile_size
        self.overlap = overlap
        self.min_h, self.min_w = min_size

        os.makedirs(cache_dir, exist_ok=True)

        self.tiles = []
        self._prepare_tiles(rebuild=rebuild)

    def _prepare_tiles(self, rebuild=False):
        print("[TiledDataset] Checking/building tiles...")
        step_h = self.tile_h - self.overlap
        step_w = self.tile_w - self.overlap

        for idx in range(len(self.base_dataset)):
            print(" preprocessing image:", idx)
            sample = self.base_dataset[idx]
            img, boxes, labels = sample["image"], sample["boxes"], sample["labels"]

            if not isinstance(img, torch.Tensor):
                raise RuntimeError(f"Only torch.Tensor supported, got {type(img)}")

            c, h, w = img.shape
            if h < self.min_h or w < self.min_w:
                print(" image too small, skipping:", idx, h, w)
                continue

            for y0 in range(0, h - self.tile_h + 1, step_h):
                for x0 in range(0, w - self.tile_w + 1, step_w):
                    fname = f"tile_{idx}_{x0}_{y0}"
                    f_img = os.path.join(self.cache_dir, fname + ".png")
                    f_json = os.path.join(self.cache_dir, fname + ".json")

                    if not rebuild and os.path.exists(f_img) and os.path.exists(f_json):
                        # already cached → reuse
                        self.tiles.append(f_img)
                        continue

                    # crop tile
                    tile = F.crop(
                        img, top=y0, left=x0,
                        height=self.tile_h, width=self.tile_w
                    ).clone()

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

                    if len(box_shifted) > 0:
                        box_shifted[:, 0::2] = box_shifted[:, 0::2].clamp(0, self.tile_w)
                        box_shifted[:, 1::2] = box_shifted[:, 1::2].clamp(0, self.tile_h)

                    # atomic saves
                    atomic_write_image(tile, f_img)
                    meta = {
                        "boxes": box_shifted.tolist(),
                        "labels": label_shifted.tolist(),
                        "image_id": int(idx),
                    }
                    atomic_write_json(meta, f_json)

                    self.tiles.append(f_img)

        print(f"[TiledDataset] Prepared {len(self.tiles)} tiles in {self.cache_dir}")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        f_img = self.tiles[idx]
        f_json = f_img.replace(".png", ".json")

        img = Image.open(f_img).convert("RGB")
        img = F.to_tensor(img)

        with open(f_json, "r") as f:
            meta = json.load(f)

        boxes = torch.tensor(meta["boxes"], dtype=torch.float32)
        labels = torch.tensor(meta["labels"], dtype=torch.int64)

        return {
            "image": img,
            "boxes": boxes,
            "labels": labels,
            "image_id": meta["image_id"],
        }

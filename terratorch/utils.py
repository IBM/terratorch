import math
from collections import Counter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_statistics(dataloader: DataLoader) -> dict[str, list[float]]:
    n_bands = dataloader.dataset[0]["image"].shape[0]
    n_data = torch.zeros([n_bands], dtype=torch.int64)
    sum_data = torch.zeros([n_bands], dtype=torch.float64)

    # First pass for mean
    for batch in tqdm(dataloader, desc="Compute mean"):
        imgs: torch.Tensor = batch["image"]
        # switch batch and band dimensions and flatten
        samples = imgs.transpose(0, 1).reshape(n_bands, -1).double()
        sum_data += samples.sum(dim=1)
        n_data += samples.shape[1]
    mean = sum_data / n_data

    sum_squared = torch.zeros(n_bands, dtype=torch.float64)
    for batch in tqdm(dataloader, desc="Compute variance"):
        imgs = batch["image"]
        samples = imgs.transpose(0, 1).reshape(n_bands, -1).double()
        sum_squared += ((samples - mean.unsqueeze(1)) ** 2).sum(dim=1)

    variance = sum_squared / n_data
    std = torch.sqrt(variance)
    return {"means": mean.numpy().tolist(), "stds": std.numpy().tolist()}


def compute_mask_statistics(dataloader: DataLoader) -> dict[int, dict[str, int | float]] | dict[str, float]:
    if torch.is_floating_point(dataloader.dataset[0]["mask"]):
        return compute_float_mask_statistics(dataloader)
    else:
        return compute_int_mask_statistics(dataloader)


def compute_int_mask_statistics(dataloader: DataLoader) -> dict[int, dict[str, int | float]]:
    counter = Counter()
    for batch in tqdm(dataloader, desc="Compute counts"):
        masks: torch.Tensor = batch["mask"]
        counter.update(masks.flatten().tolist())

    stats = {}
    for key, count in counter.items():
        stats[key] = {"count": count, "percentage": count / counter.total()}
    return stats


def compute_float_mask_statistics(dataloader: DataLoader) -> dict[str, float]:
    n_data = 0
    total = 0.0

    for batch in tqdm(dataloader, desc="Compute mask mean"):
        masks: torch.Tensor = batch["mask"]
        total += masks.sum().item()
        n_data += masks.numel()
    mean = total / n_data

    sum_squared = 0.0
    for batch in tqdm(dataloader, desc="Compute mask variance"):
        masks = batch["mask"]
        sum_squared += ((masks - mean) ** 2).sum().item()

    variance = sum_squared / n_data
    std = math.sqrt(variance)
    return {"mean": mean, "std": std}

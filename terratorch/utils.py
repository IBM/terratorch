import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_statistics(dataloader: DataLoader) -> tuple[list[float], list[float]]:
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

    return mean.numpy().tolist(), std.numpy().tolist()

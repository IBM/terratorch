import numpy as np
import torch


class MultimodalToTensor:
    def __init__(self, modalities):
        self.modalities = modalities
    def __call__(self, d):
        new_dict = {}
        for k, v in d.items():
            if not isinstance(v, np.ndarray):
                new_dict[k] = v
            else:
                # TODO: This code has hard assumptions on the data structure
                if k in self.modalities and len(v.shape) >= 3:  # Assuming raster modalities with 3+ dimensions
                    if len(v.shape) <= 4:
                        v = np.moveaxis(v, -1, 0)  # C, H, W or C, T, H, W
                    elif len(v.shape) == 5:
                        v = np.moveaxis(v, -1, 1)  # B, C, T, H, W
                    else:
                        raise ValueError(f"Unexpected shape for {k}: {v.shape}")
                new_dict[k] = torch.from_numpy(v)
        return new_dict

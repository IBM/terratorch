from collections.abc import Callable, Iterable

import albumentations as A
import torch


def wrap_in_compose_is_list(transform_list, image_modalities=None, non_image_modalities=None):
    additional_targets = {}
    if image_modalities:
        for modality in image_modalities:
            additional_targets[modality] = "image"
    if non_image_modalities:
        # Global label values are ignored and need to be processed separately
        for modality in non_image_modalities:
            additional_targets[modality] = "global_label"
    # set check shapes to false because of the multitemporal case
    return A.Compose(transform_list, is_check_shapes=False, additional_targets=additional_targets) \
        if isinstance(transform_list, Iterable) else transform_list

class MultimodalNormalize(Callable):
    def __init__(self, means, stds):
        super().__init__()
        self.means = means
        self.stds = stds

    def __call__(self, batch):
        for m in self.means.keys():
            if m not in batch["image"]:
                continue
            image = batch["image"][m]
            if len(image.shape) == 5:
                # B, C, T, H, W
                means = torch.tensor(self.means[m], device=image.device).view(1, -1, 1, 1, 1)
                stds = torch.tensor(self.stds[m], device=image.device).view(1, -1, 1, 1, 1)
            elif len(image.shape) == 4:
                # B, C, H, W
                means = torch.tensor(self.means[m], device=image.device).view(1, -1, 1, 1)
                stds = torch.tensor(self.stds[m], device=image.device).view(1, -1, 1, 1)
            elif len(self.means[m]) == 1:
                # B, (T,) H, W
                means = torch.tensor(self.means[m], device=image.device)
                stds = torch.tensor(self.stds[m], device=image.device)
            elif len(image.shape) == 3:  # No batch dim
                # C, H, W
                means = torch.tensor(self.means[m], device=image.device).view(-1, 1, 1)
                stds = torch.tensor(self.stds[m], device=image.device).view(-1, 1, 1)

            elif len(image.shape) == 2:
                means = torch.tensor(self.means[m], device=image.device)
                stds = torch.tensor(self.stds[m], device=image.device)

            elif len(image.shape) == 1:
                means = torch.tensor(self.means[m], device=image.device)
                stds = torch.tensor(self.stds[m], device=image.device)

            else:
                msg = (f"Expected batch with 5 or 4 dimensions (B, C, (T,) H, W), sample with 3 dimensions (C, H, W) "
                       f"or a single channel, but got {len(image.shape)}")
                raise Exception(msg)
            batch["image"][m] = (image - means) / stds
        return batch


import logging

from torch import nn, Tensor
import torch 

class DecoderNotFoundError(Exception):
    pass

def extract_prefix_keys(d: dict, prefix: str) -> dict:
    extracted_dict = {}
    remaining_dict = {}
    for k, v in d.items():
        if k.startswith(prefix):
            extracted_dict[k[len(prefix) :]] = v
        else:
            remaining_dict[k] = v

    return extracted_dict, remaining_dict


def pad_images(imgs: Tensor, patch_size: int | list, padding: str) -> Tensor:
    p_t = 1
    if isinstance(patch_size, int):
         p_h = p_w = patch_size
    elif len(patch_size) == 1:
        p_h = p_w = patch_size[0]
    elif len(patch_size) == 2:
        p_h, p_w = patch_size
    elif len(patch_size) == 3:
        p_t, p_h, p_w = patch_size
    else:
        raise ValueError(f'patch size {patch_size} not valid, must be int or list of ints with length 1, 2 or 3.')

    # Double the patch size to ensure the resulting number of patches is divisible by 2 (required for many decoders)
    p_h, p_w = p_h * 2, p_w * 2

    if p_t > 1 and len(imgs.shape) < 5:
        raise ValueError(f"Multi-temporal padding requested (p_t = {p_t}) "
                         f"but no multi-temporal data provided (data shape = {imgs.shape}).")

    h, w = imgs.shape[-2:]
    t = imgs.shape[-3] if len(imgs.shape) > 4 else 1
    t_pad, h_pad, w_pad = (p_t - t % p_t) % p_t, (p_h - h % p_h) % p_h, (p_w - w % p_w) % p_w
    if t_pad > 0:
        # Multi-temporal padding
        imgs = torch.stack([
            nn.functional.pad(img, (0, w_pad, 0, h_pad, 0, t_pad), mode=padding)
            for img in imgs  # Apply per image to avoid NotImplementedError from torch.nn.functional.pad
        ])
    elif h_pad > 0 or w_pad > 0:
        imgs = torch.stack([
            nn.functional.pad(img, (0, w_pad, 0, h_pad), mode=padding)
            for img in imgs  # Apply per image to avoid NotImplementedError from torch.nn.functional.pad
        ])
    return imgs

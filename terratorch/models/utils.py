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

def pad_images(imgs: Tensor,patch_size: int, padding:str) -> Tensor:
    p = patch_size
    # h, w = imgs.shape[3], imgs.shape[4]
    t, h, w = imgs.shape[-3:]
    h_pad, w_pad = (p - h % p) % p, (p - w % p) % p  # Ensure padding is within bounds
    if h_pad > 0 or w_pad > 0:
        imgs = torch.stack([
            nn.functional.pad(img, (0, w_pad, 0, h_pad), mode=padding)
            for img in imgs  # Apply per image to avoid NotImplementedError from torch.nn.functional.pad
        ])
    return imgs


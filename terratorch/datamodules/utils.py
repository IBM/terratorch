# Copyright contributors to the Terratorch project

import collections.abc
import re
from collections.abc import Iterable

import albumentations as A
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate

np_str_obj_array_pattern = re.compile(r"[SaUO]")

def wrap_in_compose_is_list(transform_list):
    # set check shapes to false because of the multitemporal case
    return A.Compose(transform_list, is_check_shapes=False) if isinstance(transform_list, Iterable) else transform_list

def pad_tensor(x, max_len, pad_value=0):
    pad_size = [0] * (2 * (x.dim() - 1)) + [0, max_len - x.size(0)]
    return F.pad(x, pad_size, value=pad_value)

def pad_collate(batch, pad_value=0):
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        max_len = max([e.size(0) for e in batch])
        batch = [pad_tensor(e, max_len, pad_value=pad_value) for e in batch]
        return torch.stack(batch, 0)
    elif isinstance(elem, np.ndarray):
        if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
            msg = f"Unsupported data type: {elem.dtype}"
            raise TypeError(msg)
        return pad_collate([torch.from_numpy(b) for b in batch], pad_value=pad_value)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch], pad_value=pad_value) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
        return elem_type(*(pad_collate(samples, pad_value=pad_value) for samples in zip(*batch, strict=False)))
    elif isinstance(elem, collections.abc.Sequence) and not isinstance(elem, str):
        transposed = zip(*batch, strict=False)
        return [pad_collate(samples, pad_value=pad_value) for samples in transposed]
    else:
        return default_collate(batch)

def check_dataset_stackability(dataset, batch_size) -> bool:

    shapes = np.array([item["image"].shape for item in dataset])

    if np.array_equal(shapes.max(0), shapes.min(0)):
        return batch_size
    else:
        print("The batch samples can't be stacked, since they don't have the same dimensions. Setting batch_size=1.")
        return 1



# Copyright contributors to the Terratorch project

import re
from collections.abc import Iterable

import albumentations as A
import numpy as np

np_str_obj_array_pattern = re.compile(r"[SaUO]")

def wrap_in_compose_is_list(transform_list):
    # set check shapes to false because of the multitemporal case
    return A.Compose(transform_list, is_check_shapes=False) if isinstance(transform_list, Iterable) else transform_list

def check_dataset_stackability(dataset, batch_size) -> bool:

    shapes = np.array([item["image"].shape for item in dataset])

    if np.array_equal(shapes.max(0), shapes.min(0)):
        return batch_size
    else:
        print("The batch samples can't be stacked, since they don't have the same dimensions. Setting batch_size=1.")
        return 1



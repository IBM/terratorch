# Copyright contributors to the Terratorch project

import os
from collections.abc import Iterator
from enum import Enum
from functools import partial
from typing import Any

import numpy as np
import torch


class HLSBands(Enum):
    COASTAL_AEROSOL = "COASTAL_AEROSOL"
    BLUE = "BLUE"
    GREEN = "GREEN"
    RED = "RED"
    RED_EDGE_1 = "RED_EDGE_1"
    RED_EDGE_2 = "RED_EDGE_2"
    RED_EDGE_3 = "RED_EDGE_3"
    NIR_BROAD = "NIR_BROAD"
    NIR_NARROW = "NIR_NARROW"
    SWIR_1 = "SWIR_1"
    SWIR_2 = "SWIR_2"
    WATER_VAPOR = "WATER_VAPOR"
    CIRRUS = "CIRRUS"
    THEMRAL_INFRARED_1 = "THEMRAL_INFRARED_1"
    THEMRAL_INFRARED_2 = "THEMRAL_INFRARED_2"

    @classmethod
    def try_convert_to_hls_bands_enum(cls, x: Any):
        try:
            return cls(x)
        except ValueError:
            return x

def default_transform(**batch):
    return to_tensor(batch)


def filter_valid_files(
    files, valid_files: Iterator[str] | None = None, ignore_extensions: bool = False, allow_substring: bool = True
):
    if valid_files is None:
        return sorted(files)
    valid_files = list(valid_files)
    if ignore_extensions:
        valid_files = [os.path.splitext(sub)[0] for sub in valid_files]
    filter_function = partial(
        _split_filter_function,
        valid_files=valid_files,
        ignore_extensions=ignore_extensions,
        allow_substring=allow_substring,
    )
    # TODO fix this
    filtered = filter(filter_function, files)

    return sorted(filtered)


def _split_filter_function(file_name, valid_files: list[str], ignore_extensions=False, allow_substring=True):
    base_name = os.path.basename(file_name)
    if ignore_extensions:
        base_name = os.path.splitext(base_name)[0]
    if not allow_substring:
        return base_name in valid_files

    for valid_file in valid_files:
        if valid_file in base_name:
            return True
    return False


def to_tensor(d, transpose=True):
    new_dict = {}
    for k, v in d.items():
        if not isinstance(v, np.ndarray):
            new_dict[k] = v
        else:
            if k == "image" and transpose:
                v = np.moveaxis(v, -1, 0)
            new_dict[k] = torch.from_numpy(v)
    return new_dict

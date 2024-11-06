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

def generate_bands_intervals(bands_intervals: list[int | str | HLSBands | tuple[int]] | None = None):
        if bands_intervals is None:
            return None
        bands = []
        for element in bands_intervals:
            # if its an interval
            if isinstance(element, tuple):
                if len(element) != 2:  # noqa: PLR2004
                    msg = "When defining an interval, a tuple of two integers should be passed,\
                        defining start and end indices inclusive"
                    raise Exception(msg)
                expanded_element = list(range(element[0], element[1] + 1))
                bands.extend(expanded_element)
            else:
                bands.append(element)
        return bands

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


def pad_numpy(x, target_length, pad_value=0):
    padlen = target_length - x.shape[0]
    if padlen <= 0:
        return x

    pad_width = [(padlen, 0)] + [(0, 0) for _ in range(len(x.shape) - 1)]

    return np.pad(x, pad_width=pad_width, mode="constant", constant_values=pad_value)


def pad_dates_numpy(dates, target_length, pad_value=-1):
    padlen = target_length - dates.shape[0]
    if padlen <= 0:
        return dates

    pad_width = [(padlen, 0)]

    return np.pad(dates, pad_width=pad_width, mode="constant", constant_values=pad_value)
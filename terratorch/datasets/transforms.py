# Copyright contributors to the Terratorch project

from albumentations import BasicTransform, Compose, ImageOnlyTransform
from einops import rearrange
from torch import Tensor
import albumentations as A

N_DIMS_FOR_TEMPORAL = 4
N_DIMS_FLATTENED_TEMPORAL = 3


def albumentations_to_callable_with_dict(albumentation: list[BasicTransform] | None = None):
    if albumentation is None:
        return lambda x: x
    albumentation = Compose(albumentation)

    def fn(data):
        return albumentation(**data)

    return fn


class FlattenTemporalIntoChannels(ImageOnlyTransform):
    """Flatten the temporal dimension into channels"""

    def __init__(self):
        super().__init__(True, 1)

    def apply(self, img, **params):
        if len(img.shape) != N_DIMS_FOR_TEMPORAL:
            msg = f"Expected input temporal sequence to have {N_DIMS_FOR_TEMPORAL} dimensions, but got {len(img.shape)}"
            raise Exception(msg)
        rearranged = rearrange(img, "time height width channels -> height width (time channels)")
        return rearranged

    def get_transform_init_args_names(self):
        return ()


class UnflattenTemporalFromChannels(ImageOnlyTransform):
    """Unflatten the temporal dimension from the channel dimension
    Assumes channels first (usually should be run after ToTensorV2)"""

    def __init__(self, n_timesteps: int | None = None, n_channels: int | None = None):
        super().__init__(True, 1)
        if n_timesteps is None and n_channels is None:
            msg = "One of n_timesteps or n_channels must be provided"
            raise Exception(msg)
        self.additional_info = {"channels": n_channels} if n_channels else {"time": n_timesteps}

    def apply(self, img, **params):
        if len(img.shape) != N_DIMS_FLATTENED_TEMPORAL:
            msg = f"Expected input temporal sequence to have {N_DIMS_FLATTENED_TEMPORAL} dimensions\
                , but got {len(img.shape)}"
            raise Exception(msg)

        rearranged = rearrange(
            img, "(time channels) height width -> channels time height width", **self.additional_info
        )
        return rearranged

    def get_transform_init_args_names(self):
        return ("n_timesteps", "n_channels")

class FlattenSamplesIntoChannels(ImageOnlyTransform):
    """Flatten the sample and optional temporal dimension into channels"""

    def __init__(self, time_dim: bool = True):
        super().__init__(True, 1)
        self.time_dim = time_dim

    def apply(self, img, **params):
        if self.time_dim:
            rearranged = rearrange(img,
                                   "samples time height width channels -> height width (samples time channels)")
        else:
            rearranged = rearrange(img, "samples height width channels -> height width (samples channels)")
        return rearranged

    def get_transform_init_args_names(self):
        return ()


class UnflattenSamplesFromChannels(ImageOnlyTransform):
    """Unflatten the sample and optional the temporal dimension from the channel dimension
    Assumes channels first (usually should be run after ToTensorV2)"""

    def __init__(
            self,
            time_dim: bool = True,
            n_samples: int | None = None,
            n_timesteps: int | None = None,
            n_channels: int | None = None
    ):
        super().__init__(True, 1)

        self.time_dim = time_dim
        if self.time_dim:
            if bool(n_channels) + bool(n_timesteps) + bool(n_samples) < 2:
                msg = "Two of n_channels, n_timesteps, and n_channels must be provided"
                raise Exception(msg)
            if n_timesteps and n_channels:
                self.additional_info = {"channels": n_channels, "time": n_timesteps}
            elif n_timesteps and n_samples:
                self.additional_info = {"time": n_timesteps, "samples": n_samples}
            else:
                self.additional_info = {"channels": n_channels, "samples": n_samples}
        else:
            if n_channels is None and n_samples is None:
                msg = "One of n_channels or n_samples must be provided"
                raise Exception(msg)
            self.additional_info = {"channels": n_channels} if n_channels else {"samples": n_samples}

    def apply(self, img, **params):
        if self.time_dim:
            rearranged = rearrange(
                img, "(samples time channels) height width -> samples channels time height width",
                **self.additional_info
            )
        else:
            rearranged = rearrange(
                img, "(samples channels) height width -> samples channels height width", **self.additional_info
            )
        return rearranged

    def get_transform_init_args_names(self):
        return ("n_timesteps", "n_channels")


class Rearrange(ImageOnlyTransform):
    """Flatten the temporal dimension into channels"""

    def __init__(
        self, rearrange: str, rearrange_args: dict[str, int] | None = None, always_apply: bool = True, p: float = 1
    ):
        super().__init__(always_apply, p)
        self.rearrange = rearrange
        self.vars = rearrange_args if rearrange_args else {}

    def apply(self, img, **params):
        return rearrange(img, self.rearrange, **self.vars)

    def get_transform_init_args_names(self):
        return "rearrange"


class SelectBands(ImageOnlyTransform):
    """Select a subset of the input bands"""

    def __init__(self, band_indices: list[int]):
        super().__init__(True, 1)
        self.band_indices = band_indices

    def apply(self, img, **params):
        return img[..., self.band_indices]

    def get_transform_init_args_names(self):
        return "band_indices"


class MultimodalTransforms:
    """Applies albumentations transforms to multiple images"""
    def __init__(self, transforms: dict | A.Compose, shared : bool = True):
        self.transforms = transforms
        self.shared = shared

    def __call__(self, data: dict):
        if self.shared:
            # albumentations requires a key 'image'
            image_modality = list(data.keys())[0]
            data['image'] = data.pop(image_modality)
            data = self.transforms(**data)
            data[image_modality] = data.pop('image')
        else:
            # Applies transformations for each modality separate
            for key, value in data.items():
                data[key] = self.transforms[key](image=value)['image']  # Only works with image modalities

        return data

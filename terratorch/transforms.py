from albumentations.core.transforms_interface import BasicTransform
import torch
import numpy as np


from albumentations.core.transforms_interface import BasicTransform

class AddStaticKeys(BasicTransform):
    def __init__(self, static_items, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.static_items = self._init_static_items(static_items)

    def _init_static_items(self, items):
        out = {}
        for k, v in items.items():
            if isinstance(v, dict) and "init" in v:
                init_type = v["init"]
                shape = v["shape"]
                if init_type == "zeros":
                    out[k] = torch.zeros(*shape)
                elif init_type == "randn":
                    out[k] = torch.randn(*shape)
                else:
                    raise ValueError(f"Unsupported init type: {init_type}")
            else:
                out[k] = v
        return out

    def apply(self, img, **params):
        return img  # Albumentations expects this to return the new image

    def update_params(self, params, **kwargs):
        # Add static keys to params, which Albumentations merges into the sample
        return {**params, **self.static_items}

    @property
    def targets(self):
        return {"image": self.apply}

    def get_transform_init_args_names(self):
        return ("static_items",)



class RemapKeys(BasicTransform):
    def __init__(self, key_map, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.key_map = key_map

    def __call__(self, force_apply=False, **kwargs):
        # kwargs is the full input dict, e.g., image=..., mask=...
        return {self.key_map.get(k, k): v for k, v in kwargs.items()}

    def get_transform_init_args_names(self):
        return ("key_map",)


class ToDict(BasicTransform):
    def __init__(self, key: str = "image", always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.key = key

    def apply(self, data, **params):
        return data

    def __call__(self, data, force_apply=False, **kwargs):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(data)}")

        return {self.key: data}

    @property
    def targets(self):
        return {self.key: self.apply}

    def get_transform_init_args_names(self):
        return ("key",)


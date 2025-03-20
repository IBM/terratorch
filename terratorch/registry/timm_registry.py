from collections.abc import Callable, KeysView, Mapping, Set

import timm
import torch
from torch import nn

from terratorch.registry import BACKBONE_REGISTRY
import pdb

class TimmBackboneWrapper(nn.Module):
    def __init__(self, timm_module: nn.Module) -> None:
        super().__init__()
        self._timm_module = timm_module
        # for backwards compatibility for times before necks
        self.prepare_features_for_image_model = getattr(self._timm_module, "prepare_features_for_image_model", lambda x: x)

    @property
    def out_channels(self):
        
        return self._timm_module.feature_info.channels()


    def forward(self, *args, **kwargs) -> list[torch.Tensor]:
        return self._timm_module(*args, **kwargs)


class TimmRegistry(Set):
    """Registry wrapper for timm"""

    def register(self, constructor: Callable | type) -> Callable:
        raise NotImplementedError()

    def build(self, name: str, features_only=True, *constructor_args, **constructor_kwargs) -> nn.Module:
        """Build and return the component.
        Use prefixes ending with _ to forward to a specific source
        """
        try:
            # pdb.set_trace()
            constructor_kwargs_new = constructor_kwargs.copy()
            # if 'bands' in constructor_kwargs_new.keys():
            #     del constructor_kwargs_new['bands']
            # elif 'model_bands' in constructor_kwargs_new.keys():
            #     del constructor_kwargs_new['model_bands']
            return TimmBackboneWrapper(
                timm.create_model(
                    name,
                    *constructor_args,
                    features_only=features_only,
                    **constructor_kwargs_new,
                )
            )
        except RuntimeError as e:
            if "Unknown model" in str(e):
                msg = f"Unknown model {name}"
                raise KeyError(msg) from e
            raise e

    def __iter__(self):
        return iter(timm.list_models())

    def __len__(self):
        return len(timm.list_models())

    def __contains__(self, key):
        return key in timm.list_models()

    # def __getitem__(self, name):
    #     return timm.model_entrypoint(name)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return f"timm registry with {len(self)} registered backbones"



TIMM_BACKBONE_REGISTRY = TimmRegistry()
BACKBONE_REGISTRY.register_source("timm", TIMM_BACKBONE_REGISTRY)

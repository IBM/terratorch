from collections.abc import Callable, Mapping

import timm
from torch import nn

from terratorch.registry import BACKBONE_REGISTRY


class TimmRegistry(Mapping):
    """Registry wrapper for timm"""

    def register(self, constructor: Callable | type) -> Callable:
        raise NotImplementedError()

    def build(self, name: str, *constructor_args, **constructor_kwargs) -> nn.Module:
        """Build and return the component.
        Use prefixes ending with _ to forward to a specific source
        """
        return timm.create_model(
            name,
            *constructor_args,
            features_only=True,
            **constructor_kwargs,
        )

    def __iter__(self):
        return iter(timm.list_models())

    def __len__(self):
        return len(timm.list_models())

    def __contains__(self, key):
        return key in timm.list_models()

    def __getitem__(self, name):
        return timm.model_entrypoint(name)

    def __repr__(self):
        return repr(timm.list_models())

    def __str__(self):
        return f"timm registry with {len(self)} registered backbones"

TIMM_BACKBONE_REGISTRY = TimmRegistry()
BACKBONE_REGISTRY.register_source("timm", TIMM_BACKBONE_REGISTRY)

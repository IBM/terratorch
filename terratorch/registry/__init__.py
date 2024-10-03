from terratorch.registry.registry import (  # noqa: I001 dont sort, import order matters
    MultiSourceRegistry,
    Registry,
    BACKBONE_REGISTRY,
    DECODER_REGISTRY,
    POST_BACKBONE_OPS_REGISTRY,
    TERRATORCH_BACKBONE_REGISTRY,
    TERRATORCH_DECODER_REGISTRY,
)
import terratorch.registry.smp_registry  # register smp registry
import terratorch.registry.timm_registry  # register timm registry  # noqa: F401

__all__ = [
    "MultiSourceRegistry",
    "Registry",
    "BACKBONE_REGISTRY",
    "DECODER_REGISTRY",
    "POST_BACKBONE_OPS_REGISTRY",
    "TERRATORCH_BACKBONE_REGISTRY",
    "TERRATORCH_DECODER_REGISTRY",
]

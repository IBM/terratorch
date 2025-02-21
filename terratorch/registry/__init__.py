from terratorch.registry.registry import (  # noqa: I001 dont sort, import order matters
    MultiSourceRegistry,
    Registry,
    BACKBONE_REGISTRY,
    DECODER_REGISTRY,
    NECK_REGISTRY,
    FULL_MODEL_REGISTRY,
    TERRATORCH_BACKBONE_REGISTRY,
    TERRATORCH_DECODER_REGISTRY,
    TERRATORCH_NECK_REGISTRY,
    TERRATORCH_FULL_MODEL_REGISTRY,
    MODEL_FACTORY_REGISTRY,
)
import terratorch.registry.smp_registry  # register smp registry
import terratorch.registry.timm_registry  # register timm registry  # noqa: F401
import terratorch.registry.mmseg_registry
import terratorch.registry.custom_registry

__all__ = [
    "MultiSourceRegistry",
    "Registry",
    "BACKBONE_REGISTRY",
    "DECODER_REGISTRY",
    "NECK_REGISTRY",
    "FULL_MODEL_REGISTRY",
    "TERRATORCH_BACKBONE_REGISTRY",
    "TERRATORCH_DECODER_REGISTRY",
    "TERRATORCH_NECK_REGISTRY",
    "TERRATORCH_FULL_MODEL_REGISTRY",
    "MODEL_FACTORY_REGISTRY"
]

# Copyright contributors to the Terratorch project

import terratorch.models  # noqa: F401
from terratorch.models.backbones import *  # register models in registries # noqa: F403
from terratorch.registry import BACKBONE_REGISTRY, DECODER_REGISTRY, MODEL_FACTORY_REGISTRY, FULL_MODEL_REGISTRY  # noqa: F401

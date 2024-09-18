# Copyright contributors to the Terratorch project
import importlib

from terratorch.models.generic_unet_model_factory import GenericUnetModelFactory
from terratorch.models.prithvi_model_factory import PrithviModelFactory
from terratorch.models.satmae_model_factory import SatMAEModelFactory
from terratorch.models.scalemae_model_factory import ScaleMAEModelFactory
from terratorch.models.smp_model_factory import SMPModelFactory
from terratorch.models.timm_model_factory import TimmModelFactory

__all__ = (
    "PrithviModelFactory",
    "ClayModelFactory",
    "SatMAEModelFactory",
    "ScaleMAEModelFactory",
    "SMPModelFactory",
    "GenericUnetModelFactory",
    "TimmModelFactory",
    "SatlasModelFactory",
    "AuxiliaryHead",
    "AuxiliaryHeadWithDecoderWithoutInstantiatedHead",
)

if importlib.util.find_spec("satlaspretrain_models"):
    from terratorch.models.satlas_model_factory import SatlasModelFactory
    __all__ = (*__all__, "SatlasModelFactory")

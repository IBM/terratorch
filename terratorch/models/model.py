# Copyright contributors to the Terratorch project

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn

FACTORY_REGISTRY = {}


@dataclass
class ModelOutput:
    output: Tensor
    auxiliary_heads: dict[str, Tensor] = None


class Model(ABC, nn.Module):
    @abstractmethod
    def freeze_encoder(self):
        pass

    @abstractmethod
    def freeze_decoder(self):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> ModelOutput:
        pass


class ModelFactory(ABC):
    @abstractmethod
    def build_model(self, *args, **kwargs) -> Model:
        pass


def get_factory(factory_name: str) -> ModelFactory:
    if factory_name not in FACTORY_REGISTRY:
        msg = f"Factory with name {factory_name} does not exist. Choose one of {list(FACTORY_REGISTRY.keys())}"
        raise Exception(msg)
    return FACTORY_REGISTRY[factory_name]()


def register_factory(factory_class: type) -> None:
    FACTORY_REGISTRY[factory_class.__name__] = factory_class
    return factory_class


@dataclass
class AuxiliaryHead:
    """Class containing all information to create auxiliary heads.

    Args:
        name (str): Name of the head. Should match the name given to the auxiliary loss.
        decoder (str): Name of the decoder class to be used.
        decoder_args (dict | None): parameters to be passed to the decoder constructor.
            Parameters for the decoder should be prefixed with `decoder_`.
            Parameters for the head should be prefixed with `head_`.
    """

    name: str
    decoder: str
    decoder_args: dict | None


@dataclass
class AuxiliaryHeadWithDecoderWithoutInstantiatedHead:
    """Intermediary class containing the instantiated decoder without the instantiated head.

    Args:
        name (str): Name of the head. Should match the name given to the auxiliary loss.
        decoder (nn.Module): Instantiated decoder.
        head_args (dict | None): parameters to be passed to the head constructor.
    """

    name: str
    decoder: nn.Module
    head_args: dict | None

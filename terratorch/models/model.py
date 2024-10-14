# Copyright contributors to the Terratorch project

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class ModelOutput:
    output: Tensor
    auxiliary_heads: dict[str, Tensor] = None


class Model(ABC, nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    @abstractmethod
    def freeze_encoder(self):
        pass

    @abstractmethod
    def freeze_decoder(self):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> ModelOutput:
        pass


class ModelFactory(typing.Protocol):
    def build_model(self, *args, **kwargs) -> Model:...

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
        decoder_includes_head (bool): Whether the decoder already includes a head
    """

    name: str
    decoder: nn.Module
    head_args: dict | None
    decoder_includes_head: bool = False

import itertools
import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Protocol

import timm
from torch import nn


class RegistryMapping(Protocol, Mapping[str, Callable]):
    def build(self, name: str, *args, **kwargs) -> nn.Module:
        ...

class MultiSourceRegistry(RegistryMapping):
    def __init__(self) -> None:
        self._sources: OrderedDict[str, RegistryMapping] = OrderedDict()

    def _parse_prefix(self, name) -> tuple[str, str] | None:
        split = name.split("_")
        if len(split) > 1 and split[0] in self._sources:
            prefix = self._sources[split[0]]
            name_without_prefix = "_".join(split[1:])
            return prefix, name_without_prefix
        return None

    def build(self, name: str, *constructor_args, **constructor_kwargs) -> nn.Module:
        parsed_prefix = self._parse_prefix(name)
        if parsed_prefix:
            prefix, name_without_prefix = parsed_prefix
            return self._sources[prefix].build(name_without_prefix, *constructor_args, **constructor_kwargs)

        # if no prefix
        for source in self._sources.values():
            try:
                return source.build(name, *constructor_args, **constructor_kwargs)
            except Exception as e:
                logging.debug(e)

        msg = f"Could not instantiate Model {name} not from any source."
        raise Exception(msg)

    def register_source(self, prefix: str, registry) -> None:
        """Register a source in the registry

        """
        if prefix in self._sources:
            msg = f"Source for prefix {prefix} already exists."
            raise KeyError(msg)
        self._sources[prefix] = registry

    def __iter__(self):
        return itertools.chain(*[iter(source) for source in self._sources])

    def __len__(self):
        return sum(len(source) for source in self.sources)

    def __getitem__(self, name):
        parsed_prefix = self._parse_prefix(name)
        if parsed_prefix:
            prefix, name_without_prefix = parsed_prefix
            return self._sources[prefix][name_without_prefix]

        # if no prefix is given, go through all sources in order
        for source in self._sources.values():
            try:
                return source[name]
            except Exception as e:
                logging.debug(e)

        msg = f"Could not find Model {name} not from any source."
        raise Exception(msg)

    def __contains__(self, name):
        parsed_prefix = self._parse_prefix(name)
        if parsed_prefix:
            prefix, name_without_prefix = parsed_prefix
            return name_without_prefix in self._sources[prefix]
        return any(name in source for source in self._sources)

class Registry(RegistryMapping):
    """Registry holding model constructors and multiple additional sources.

    This registry behaves as a dictionary from strings, which are model names,
    to model classes or functions which instantiate model classes.

    In addition, it supports the addition of other sources from which it can instantiate models. These sources require
    a prefix and a function that instantiates models given a name and *args and **kwargs.

    Add constructors to the registry by annotating them with @registry.register.

    Add sources by annotating them with @registry.register_source(prefix)

    Build models with registry.build(name, *args, **kwargs). If the name has a prefix separated with _, that
    will be taken as the source to instantiate from.
    If not, sources will be tried in the order they were added, starting with the internal registry.

    >>> registry = Registry()
    >>> @registry.register
    ... def model(*args, **kwargs):
    ...     return object()
    >>> "model" in registry
    True
    >>> model_instance = registry.build("model")
    """
    def __init__(self):
        self._registry: dict[str, Callable] = {}

    def register(self, constructor: Callable | type) -> Callable:
        """Register a component in the registry. Used as a decorator.

        Args:
            backbone_constructor (Callable | type): Function or class to be decorated with @register.
        """
        if not callable(constructor):
            msg = f"Invalid argument. Decorate a function or class with @{self.__class__.__name__}.register"
            raise TypeError(msg)
        self._registry[constructor.__name__] = constructor
        return constructor

    def build(self, name: str, *constructor_args, **constructor_kwargs) -> nn.Module:
        """Build and return the component.
        Use prefixes ending with _ to forward to a specific source
        """
        return self._registry[name](*constructor_args, **constructor_kwargs)

    def __iter__(self):
        return iter(self._registry)

    def __getitem__(self, key):
        return self._registry[key]

    def __len__(self):
        return len(self._registry)

    def __contains__(self, key):
        return key in self._registry

class TimmRegistry(RegistryMapping):
    """Registry wrapper for timm
    """
    def register(self, constructor: Callable | type) -> Callable:
        raise NotImplementedError()

    def build(self, name: str, *constructor_args, **constructor_kwargs):
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

TERRATORCH_BACKBONE_REGISTRY = Registry()
TIMM_BACKBONE_REGISTRY = TimmRegistry()
BACKBONE_REGISTRY = MultiSourceRegistry()
BACKBONE_REGISTRY.register_source("terratorch", TERRATORCH_BACKBONE_REGISTRY)
BACKBONE_REGISTRY.register_source("timm", TIMM_BACKBONE_REGISTRY)


TERRATORCH_DECODER_REGISTRY = Registry()
DECODER_REGISTRY = MultiSourceRegistry()
DECODER_REGISTRY.register_source("terratorch", TERRATORCH_DECODER_REGISTRY)
#TODO: add smp decoders
# @DECODER_REGISTRY.register_source("")
# def _build_smp_decoder(decoder_identifier: str, *args, **kwargs) -> nn.Module:
#     return _get_smp_decoder(
#                 decoder_identifier,
#                 backbone_kwargs,
#                 decoder_kwargs,
#                 out_channels,
#                 in_channels,
#                 num_classes,
#                 output_stride,
#             )

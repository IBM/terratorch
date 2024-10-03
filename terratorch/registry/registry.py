import itertools
import logging
import typing
from collections import OrderedDict
from collections.abc import Callable, Mapping

V = typing.TypeVar("V")


class BuildableRegistry(typing.Protocol):
    def __getitem__(self, name: str): ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __contains__(self, name: str) -> bool: ...
    def build(self, name: str, *args, **kwargs): ...


class MultiSourceRegistry(Mapping):
    def __init__(self) -> None:
        self._sources: OrderedDict[str, BuildableRegistry] = OrderedDict()

    def _parse_prefix(self, name) -> tuple[str, str] | None:
        split = name.split("_")
        if len(split) > 1 and split[0] in self._sources:
            prefix = split[0]
            name_without_prefix = "_".join(split[1:])
            return prefix, name_without_prefix
        return None

    def build(self, name: str, *constructor_args, **constructor_kwargs):
        parsed_prefix = self._parse_prefix(name)
        if parsed_prefix:
            prefix, name_without_prefix = parsed_prefix
            registry = self._sources[prefix]
            return registry.build(name_without_prefix, *constructor_args, **constructor_kwargs)

        # if no prefix
        for source in self._sources.values():
            try:
                return source.build(name, *constructor_args, **constructor_kwargs)
            except Exception as e:
                logging.debug(e)

        msg = f"Could not instantiate Model {name} not from any source."
        raise Exception(msg)

    def register_source(self, prefix: str, registry: BuildableRegistry) -> None:
        """Register a source in the registry"""
        if prefix in self._sources:
            msg = f"Source for prefix {prefix} already exists."
            raise KeyError(msg)
        self._sources[prefix] = registry

    def __iter__(self):
        return itertools.chain(*[iter(source) for source in self._sources])

    def __len__(self):
        return sum(len(source) for source in self._sources)

    def __getitem__(self, name):
        parsed_prefix = self._parse_prefix(name)
        if parsed_prefix:
            prefix, name_without_prefix = parsed_prefix
            registry = self._sources[prefix]
            return registry[name_without_prefix]

        # if no prefix is given, go through all sources in order
        for source in self._sources.values():
            try:
                return source[name]
            except Exception as e:
                logging.debug(e)

        msg = f"Could not find Model {name} not from any source."
        raise KeyError(msg)

    def __contains__(self, name):
        parsed_prefix = self._parse_prefix(name)
        if parsed_prefix:
            prefix, name_without_prefix = parsed_prefix
            return name_without_prefix in self._sources[prefix]
        return any(name in source for source in self._sources.values())

    def __repr__(self):
        repr_dict = {prefix: repr(source) for prefix, source in self._sources.items()}
        return repr(repr_dict)

    def __str__(self):
        sources_str = str(" | ".join([f"{prefix}: {source!s}" for prefix, source in self._sources.items()]))
        return f"Multi source registry with {len(self)} items: {sources_str}"


class Registry(Mapping):
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

    def __init__(self) -> None:
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

    def build(self, name: str, *constructor_args, **constructor_kwargs):
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

    def __repr__(self):
        return repr(self._registry)

    def __str__(self):
        return f"Registry with {len(self)} registered items"

## Declare library level registries below

### Backbone Registry
TERRATORCH_BACKBONE_REGISTRY = Registry()
BACKBONE_REGISTRY = MultiSourceRegistry()
BACKBONE_REGISTRY.register_source("terratorch", TERRATORCH_BACKBONE_REGISTRY)

### Decoder Registry
TERRATORCH_DECODER_REGISTRY = Registry()
DECODER_REGISTRY = MultiSourceRegistry()
DECODER_REGISTRY.register_source("terratorch", TERRATORCH_DECODER_REGISTRY)

### Post Backbone Ops Registry
POST_BACKBONE_OPS_REGISTRY = Registry()

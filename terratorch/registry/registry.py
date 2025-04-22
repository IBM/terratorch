import typing
from collections import OrderedDict
from collections.abc import Callable, Mapping, Set
from contextlib import suppress
from reprlib import recursive_repr as _recursive_repr
import logging

logger = logging.getLogger(__name__)

class BuildableRegistry(typing.Protocol):
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __contains__(self, name: str) -> bool: ...
    def build(self, name: str, *args, **kwargs):
        """Should raise KeyError if model not found.
        """


class DecoderRegistry(BuildableRegistry, typing.Protocol):
    includes_head: bool

T = typing.TypeVar("T", bound=BuildableRegistry)
class MultiSourceRegistry(Mapping[str, T], typing.Generic[T]):
    """Registry that searches in multiple sources

        Correct functioning of this class depends on registries raising a KeyError when the model is not found.
    """
    def __init__(self, **sources) -> None:
        self._sources: OrderedDict[str, T] = OrderedDict(sources)

    def _parse_prefix(self, name) -> tuple[str, str] | None:
        split = name.split("_")
        if len(split) > 1 and split[0] in self._sources:
            prefix = split[0]
            name_without_prefix = "_".join(split[1:])
            return prefix, name_without_prefix
        return None

    def find_registry(self, name: str) -> T:
        parsed_prefix = self._parse_prefix(name)
        if parsed_prefix:
            prefix, name_without_prefix = parsed_prefix
            registry = self._sources[prefix]
            return registry

        # if no prefix is given, go through all sources in order
        for registry in self._sources.values():
            if name in registry:
                return registry
        msg = f"Model {name} not found in any registry"
        raise KeyError(msg)

    def find_class(self, name: str) -> type:
        parsed_prefix = self._parse_prefix(name)
        registry = self.find_registry(name)
        if parsed_prefix:
            prefix, name_without_prefix = parsed_prefix
            return registry[name_without_prefix]
        return registry[name]

    def build(self, name: str, *constructor_args, **constructor_kwargs):
        parsed_prefix = self._parse_prefix(name)
        if parsed_prefix:
            prefix, name_without_prefix = parsed_prefix
            registry = self._sources[prefix]
            return registry.build(name_without_prefix, *constructor_args, **constructor_kwargs)

        # if no prefix, try to build in order
        for source in self._sources.values():
            with suppress(KeyError):
                return source.build(name, *constructor_args, **constructor_kwargs)

        msg = f"Could not instantiate model {name} not from any source."
        raise KeyError(msg)

    def register_source(self, prefix: str, registry: T) -> None:
        """Register a source in the registry"""
        if prefix in self._sources:
            msg = f"Source for prefix {prefix} already exists."
            raise KeyError(msg)
        self._sources[prefix] = registry

    def __iter__(self):
        for prefix in self._sources:
            for element in self._sources[prefix]:
                yield prefix + "_" + element

    def __len__(self):
        return sum(len(source) for source in self._sources.values())

    def __getitem__(self, name):
        return self._sources[name]

    def __contains__(self, name):
        parsed_prefix = self._parse_prefix(name)
        if parsed_prefix:
            prefix, name_without_prefix = parsed_prefix
            return name_without_prefix in self._sources[prefix]
        return any(name in source for source in self._sources.values())

    @_recursive_repr()
    def __repr__(self):
        args = [f"{name}={source!r}" for name, source in self._sources.items()]
        return f'{self.__class__.__name__}({", ".join(args)})'

    def __str__(self):
        sources_str = str(" | ".join([f"{prefix}: {source!s}" for prefix, source in self._sources.items()]))
        return f"Multi source registry with {len(self)} items: {sources_str}"

    def keys(self):
        return self._sources.keys()


class Registry(Set):
    """Registry holding model constructors and multiple additional sources.

    This registry behaves as a set of strings, which are model names,
    to model classes or functions which instantiate model classes.

    In addition, it can instantiate models with the build method.

    Add constructors to the registry by annotating them with @registry.register.
    ```
    >>> registry = Registry()
    >>> @registry.register
    ... def model(*args, **kwargs):
    ...     return object()
    >>> "model" in registry
    True
    >>> model_instance = registry.build("model")
    ```
    """

    def __init__(self, **elements) -> None:
        self._registry: dict[str, Callable] = dict(elements)

    def register(self, constructor: Callable | type) -> Callable:
        """Register a component in the registry. Used as a decorator.

        Args:
            constructor (Callable | type): Function or class to be decorated with @register.
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
        return f"{self.__class__.__name__}({self._registry!r})"

    def __str__(self):
        return f"Registry with {len(self)} registered items"

## Declare library level registries below

### Backbone Registry
TERRATORCH_BACKBONE_REGISTRY = Registry()
BACKBONE_REGISTRY: MultiSourceRegistry[BuildableRegistry] = MultiSourceRegistry()
BACKBONE_REGISTRY.register_source("terratorch", TERRATORCH_BACKBONE_REGISTRY)

### Neck Registry
NECK_REGISTRY: MultiSourceRegistry[BuildableRegistry] = MultiSourceRegistry()
TERRATORCH_NECK_REGISTRY = Registry()
NECK_REGISTRY.register_source("terratorch", TERRATORCH_NECK_REGISTRY)

### Decoder Registry
TERRATORCH_DECODER_REGISTRY = typing.cast(DecoderRegistry, Registry())
TERRATORCH_DECODER_REGISTRY.includes_head = False
DECODER_REGISTRY: MultiSourceRegistry[DecoderRegistry] = MultiSourceRegistry()
DECODER_REGISTRY.register_source("terratorch", TERRATORCH_DECODER_REGISTRY)

# Full model registry
TERRATORCH_FULL_MODEL_REGISTRY = Registry()
FULL_MODEL_REGISTRY: MultiSourceRegistry[BuildableRegistry] = MultiSourceRegistry()
FULL_MODEL_REGISTRY.register_source("terratorch", TERRATORCH_FULL_MODEL_REGISTRY)

### Model Factory Registry
MODEL_FACTORY_REGISTRY = Registry()

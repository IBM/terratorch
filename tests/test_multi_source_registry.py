from collections.abc import Callable, Set

import pytest

from terratorch.registry import MultiSourceRegistry


# Define a simple BuildableRegistry mock
class SimpleRegistry(Set):
    def __init__(self):
        self._registry = {}

    def register(self, name: str, obj: Callable):
        self._registry[name] = obj

    def build(self, name: str, *args, **kwargs):
        return self._registry[name](*args, **kwargs)


    def __iter__(self):
        return iter(self._registry)

    def __len__(self):
        return len(self._registry)

    def __contains__(self, name) -> bool:
        return name in self._registry


# Fixtures for the tests
@pytest.fixture
def multi_source_registry():
    return MultiSourceRegistry()


@pytest.fixture
def simple_registry():
    return SimpleRegistry()


def test_register_source(multi_source_registry, simple_registry):
    multi_source_registry.register_source("simple", simple_registry)
    assert "simple" in multi_source_registry._sources


def test_build_from_registered_source(multi_source_registry, simple_registry):
    # Register a function in the source registry
    simple_registry.register("my_model", lambda: "constructed_model")

    # Register the source in the MultiSourceRegistry
    multi_source_registry.register_source("simple", simple_registry)

    # Build the model
    result = multi_source_registry.build("simple_my_model")

    assert result == "constructed_model"


def test_build_without_prefix(multi_source_registry, simple_registry):
    # Register a function in the source registry
    simple_registry.register("my_model", lambda: "constructed_model")

    # Register the source
    multi_source_registry.register_source("simple", simple_registry)

    # Build the model (without prefix)
    result = multi_source_registry.build("my_model")

    assert result == "constructed_model"


def test_build_fails_if_not_found(multi_source_registry):
    with pytest.raises(KeyError) as excinfo:
        multi_source_registry.build("nonexistent_model")
    assert "Could not instantiate model" in str(excinfo.value)


def test_register_duplicate_source(multi_source_registry, simple_registry):
    multi_source_registry.register_source("simple", simple_registry)
    with pytest.raises(KeyError) as excinfo:
        multi_source_registry.register_source("simple", simple_registry)
    assert "Source for prefix simple already exists." in str(excinfo.value)


def test_contains_method(multi_source_registry, simple_registry):
    # Register a function in the source registry
    simple_registry.register("my_model", lambda: "constructed_model")

    # Register the source
    multi_source_registry.register_source("simple", simple_registry)

    # Check for prefixed name
    assert "simple_my_model" in multi_source_registry

    # Check without prefix
    assert "my_model" in multi_source_registry


def test_getitem_method(multi_source_registry, simple_registry):
    # Register a function in the source registry
    def model_constructor():
        return "constructed_model"
    simple_registry.register("my_model", model_constructor)

    # Register the source
    multi_source_registry.register_source("simple", simple_registry)

    # Get item by prefixed name
    assert multi_source_registry["simple"] is simple_registry

def test_find_registry_method(multi_source_registry, simple_registry):
    def model_constructor():
        return "constructed_model"
    simple_registry.register("my_model", model_constructor)

    multi_source_registry.register_source("simple", simple_registry)

    assert multi_source_registry.find_registry("my_model") is simple_registry

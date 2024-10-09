
import pytest

from terratorch.registry import Registry


class DummyModel:
    def __init__(self, param):
        self.param = param


@pytest.fixture
def registry():
    return Registry()


def test_register_function(registry):
    """Test if a function can be registered and retrieved."""

    @registry.register
    def dummy_function():
        return "dummy"

    assert "dummy_function" in registry

def test_register_class(registry):
    """Test if a class can be registered and instantiated."""

    @registry.register
    class DummyClass:
        def __init__(self, value):
            self.value = value

    assert "DummyClass" in registry
    instance = registry.build("DummyClass", value=5)
    assert isinstance(instance, DummyClass)
    assert instance.value == 5


def test_build_model(registry):
    """Test building a model with arguments."""

    @registry.register
    def build_dummy_model(param):
        return DummyModel(param)

    model = registry.build("build_dummy_model", param=42)
    assert isinstance(model, DummyModel)
    assert model.param == 42


def test_build_nonexistent_model(registry):
    """Test that an error is raised when trying to build an unregistered model."""
    with pytest.raises(KeyError, match="build_nonexistent_model"):
        registry.build("build_nonexistent_model")


def test_len_and_iter(registry):
    """Test the length and iteration over the registry."""

    @registry.register
    def model_a():
        pass

    @registry.register
    def model_b():
        pass

    assert len(registry) == 2
    assert set(registry) == {"model_a", "model_b"}


def test_invalid_registration(registry):
    """Test that registering a non-callable object raises a TypeError."""
    with pytest.raises(TypeError):
        registry.register("not_a_function_or_class")


def test_contains_method(registry):
    """Test if the __contains__ method works as expected."""

    @registry.register
    def model_x():
        pass

    assert "model_x" in registry
    assert "nonexistent_model" not in registry

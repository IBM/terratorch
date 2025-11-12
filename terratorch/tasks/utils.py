import importlib
from torch import nn
from typing import Any, Dict

def get_module_and_class(path: str) -> (str, str):
    """Splits a dotted path string into module and class name."""
    class_name = path.split(".")[-1]
    module_name = path.replace("." + class_name, "")
    return module_name, class_name

def _instantiate_from_path(dotted_path: str, **kwargs: Any) -> Any:
    """
    Instantiates a class from a dotted path string (e.g., "my_module.MyClass").

    Args:
        dotted_path: The complete dotted path to the class.
        **kwargs: Keyword arguments to pass to the class constructor.

    Returns:
        An instance of the specified class.
    """
    module_name, class_name = get_module_and_class(dotted_path)
    
    # Import the module
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_name}' from path '{dotted_path}'") from e
        
    # Get the class from the module
    try:
        class_ = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(f"Module '{module_name}' has no class named '{class_name}'") from e
        
    # Instantiate and return the class
    return class_(**kwargs)
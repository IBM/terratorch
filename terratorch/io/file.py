import os
import torch
import importlib 
from torch import nn
import numpy as np

def open_generic_torch_model(model: type | str = None,
                             model_kwargs: dict = None,
                             model_weights_path: str = None):

    if isinstance(model, type):

        assert model_kwargs, "model_kwargs must be defined"
        model: nn.Module = model(**model_kwargs)

    else:
        model_module = importlib.import_module(model)

        model: nn.Module = model_module(**model_kwargs)

    filename = os.path.basename(model_weights_path)
    dirname = os.path.dirname(model_weights_path)

    return load_torch_weights(model=model, save_dir=dirname, name=filename) 


def load_torch_weights(model:nn.Module=None, save_dir: str = None, name: str = None, device: str = None) -> None:

    print(f"Trying to load for {device}")

    try: # If 'model' was instantiated outside this function, the dictionary of weights will be loaded.
        if device != None:
            model.load_state_dict(
                torch.load(
                    os.path.join(save_dir, name),
                    map_location=torch.device(device),
                    weights_only=True,
                )
            )
        else:
            model.load_state_dict(torch.load(os.path.join(save_dir, name), map_location='cpu', weights_only=True))

    except Exception:
        print(
            f"It was not possible to load from {os.path.join(save_dir, name)}"
        )

    return model


def load_from_file_or_attribute(value: list[float]|str):

    if isinstance(value, list):
        return value
    elif isinstance(value, str):  # It can be the path for a file
        if os.path.isfile(value):
            try:
                content = np.genfromtxt(value).tolist()
            except:
                raise Exception(f"File must be txt, but received {value}")
        else:
            raise Exception(f"The input {value} does not exist or is not a file.")

        return content



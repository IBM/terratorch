import os
import importlib 
from torch import nn

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
                    os.path.join(save_dir, name + ".pth"),
                    map_location=torch.device(device),
                )
            )
        else:
            #try:
            #    path = os.path.join(save_dir, name)
            #    checkpoint = torch.load(path, map_location='cpu')
            #    model = checkpoint['model']
            #    state_dict = model.state_dict()
            #    msg = model.load_state_dict(model, strict=False)

            #except Exception:         

            model.load_state_dict(torch.load(os.path.join(save_dir, name)))

    except Exception:
        print(
            f"It was not possible to load from {os.path.join(save_dir, name + '.pth')}"
        )

    return model

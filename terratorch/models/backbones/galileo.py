import logging

import huggingface_hub
import torch
from torch import nn

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

try:
    from galileo.galileo import GalileoWrapper
except:
    logging.getLogger("terratorch").debug(
        "The package `galileo` is not installed. If you want to use it, install it using"
        "`pip install git+https://github.com/Joao-L-S-Almeida/terratorch-galileo.git`"
    )


def load_weights(model: nn.Module, ckpt_data: dict, **kwargs) -> nn.Module:
    print("Loading weights")
    if ckpt_data is not None:
        ckpt_data = huggingface_hub.hf_hub_download(**ckpt_data)

        checkpoint_model = torch.load(ckpt_data, map_location="cpu", weights_only=True)
        state_dict = model.model.state_dict()

        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logging.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        model.model.load_state_dict(checkpoint_model, strict=False)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def galileo_tiny_encoder(pretrained: bool = None, kind: str = "s1", ckpt_data: str | None = None, **kwargs):
    remote_checkpoint_path = {"repo_id": "nasaharvest/galileo", "subfolder": "models/tiny", "filename": "encoder.pt"}

    if not ckpt_data:
        ckpt_data = remote_checkpoint_path

    model = Galileo(kind=kind, **kwargs)

    if pretrained:
        model = load_weights(model, ckpt_data)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def galileo_base_encoder(pretrained: bool = None, kind: str = "s1", ckpt_data: str | None = None, **kwargs):
    remote_checkpoint_path = {"repo_id": "nasaharvest/galileo", "subfolder": "models/base", "filename": "encoder.pt"}

    if not ckpt_data:
        ckpt_data = remote_checkpoint_path

    model = Galileo(kind=kind, **kwargs)

    if pretrained:
        model = load_weights(model, ckpt_data)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def galileo_nano_encoder(pretrained: bool = None, kind: str = "s1", ckpt_data: str | None = None, **kwargs):
    remote_checkpoint_path = {"repo_id": "nasaharvest/galileo", "subfolder": "models/nano", "filename": "encoder.pt"}

    if not ckpt_data:
        ckpt_data = remote_checkpoint_path

    model = Galileo(kind=kind, **kwargs)

    if pretrained:
        model = load_weights(model, ckpt_data)

    return model


# Built-in wrapper for Galileo FM
class Galileo(nn.Module):
    def __init__(self, kind: str = "s1", **kwargs):
        super().__init__()

        self.kind = kind
        self.model = GalileoWrapper(**kwargs)

    def forward(self, x):
        return self.model(**{f"{self.kind}": x})

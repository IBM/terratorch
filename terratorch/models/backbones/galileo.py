import logging

import huggingface_hub
import torch
from torch import nn

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY


def load_weights(model: nn.Module, ckpt_data: dict, **kwargs) -> nn.Module:
    logging.getLogger("terratorch").info("Loading weights")

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
def galileo_tiny_encoder(
    pretrained: bool = None,
    kind: str = "s1",
    ckpt_data: str | None = None,
    bands: list = None,
    model_bands: list = None,
    **kwargs,
):
    remote_checkpoint_path = {"repo_id": "nasaharvest/galileo", "subfolder": "models/tiny", "filename": "encoder.pt"}

    if not ckpt_data:
        ckpt_data = remote_checkpoint_path

    model = Galileo(kind=kind, bands=bands, model_bands=model_bands, **kwargs)

    if pretrained:
        model = load_weights(model, ckpt_data)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def galileo_base_encoder(
    pretrained: bool = None,
    kind: str = "s1",
    ckpt_data: str | None = None,
    bands: list = None,
    model_bands: list = None,
    **kwargs,
):
    remote_checkpoint_path = {"repo_id": "nasaharvest/galileo", "subfolder": "models/base", "filename": "encoder.pt"}

    if not ckpt_data:
        ckpt_data = remote_checkpoint_path

    model = Galileo(kind=kind, bands=bands, model_bands=model_bands, **kwargs)

    if pretrained:
        model = load_weights(model, ckpt_data)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def galileo_nano_encoder(
    pretrained: bool = None,
    kind: str = "s1",
    ckpt_data: str | None = None,
    bands: list = None,
    model_bands: list = None,
    **kwargs,
):
    remote_checkpoint_path = {"repo_id": "nasaharvest/galileo", "subfolder": "models/nano", "filename": "encoder.pt"}

    if not ckpt_data:
        ckpt_data = remote_checkpoint_path

    model = Galileo(kind=kind, bands=bands, model_bands=model_bands, **kwargs)

    if pretrained:
        model = load_weights(model, ckpt_data)

    return model


# Built-in wrapper for Galileo FM
class Galileo(nn.Module):
    def __init__(
        self, kind: str = "s1", transpose: bool = False, bands: list = None, model_bands: list = None, **kwargs
    ):
        super().__init__()

        try:
            from galileo.data.earthengine.s1 import S1_BANDS
            from galileo.data.earthengine.s2 import ALL_S2_BANDS
            from galileo.galileo import GalileoWrapper
        except ModuleNotFoundError:
            raise Exception(
                "The package `galileo` is not installed. If you want to use it, install it using"
                "`pip install git+https://github.com/Joao-L-S-Almeida/terratorch-galileo.git`"
            )

        self.kind = kind

        if transpose:
            self.modify_batch = self.modify_batch_
        else:
            self.modify_batch = lambda x: x

        if self.kind == "s1":
            self.model_bands_default = S1_BANDS
        else:
            self.model_bands_default = ALL_S2_BANDS

        self.model_bands = model_bands or self.model_bands_default
        self.bands = bands or self.model_bands_default

        self.model = GalileoWrapper(**kwargs)

        self.out_channels = [self.model.encoder.embedding_size]

    def modify_batch_(self, x):
        dims = x.shape

        if len(dims) == 5:
            return x.permute(0, 2, 3, 4, 1)
        elif len(dims) == 4:
            return x.permute(0, 2, 3, 1)

    def prepare_input_tensor(self, x):
        x = self.modify_batch(x)

        dims = x.shape[:-1] + (len(self.model_bands_default),)
        x_ext = torch.zeros(*dims).to(x.device)

        for j, band in enumerate(self.model_bands):
            i = self.bands.index(band)

            x_ext[..., i] = x[..., j]

        return x_ext

    def forward(self, x):
        x = self.prepare_input_tensor(x)

        output = self.model(**{f"{self.kind}": x})

        return [output]

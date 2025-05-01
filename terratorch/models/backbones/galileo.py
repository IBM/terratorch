import logging
import torch
from torch import nn
import huggingface_hub
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY


def load_weights(model: nn.Module, ckpt_data: dict, **kwargs) -> nn.Module:

    print(f"Loading weights for {model}")
    if ckpt_data is not None:

        ckpt_data = huggingface_hub.hf_hub_download(**ckpt_data)

        checkpoint_model = torch.load(ckpt_data, map_location="cpu", weights_only=True)
        state_dict = model.state_dict()

        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                logging.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        model.load_state_dict(checkpoint_model, strict=False)

    return model

@TERRATORCH_BACKBONE_REGISTRY.register
def galileo_tiny_encoder(pretrained:bool=None, ckpt_data: str | None = None, **kwargs):

    remote_checkpoint_path = {
        "repo_id":
        "nasaharvest/galileo",
        "subfolder": "models/tiny",
        "filename": "encoder.pt"
    }

    if not ckpt_data:
      ckpt_data = remote_checkpoint_path

    model = Galileo(**kwargs)

    if pretrained:
      model = load_weights(model, ckpt_data)

    return model 

@TERRATORCH_BACKBONE_REGISTRY.register
def galileo_base_encoder(pretrained:bool=None, ckpt_data: str | None = None, **kwargs):

    remote_checkpoint_path = {
        "repo_id":
        "nasaharvest/galileo",
        "subfolder": "models/base",
        "filename": "encoder.pt"
    }

    if not ckpt_data:
      ckpt_data = remote_checkpoint_path

    model = Galileo(**kwargs)

    if pretrained:
      model = load_weights(model, ckpt_data)

    return model 

@TERRATORCH_BACKBONE_REGISTRY.register
def galileo_nano_encoder(pretrained:bool=None, ckpt_data: str | None = None, **kwargs):

    remote_checkpoint_path = {
        "repo_id":
        "nasaharvest/galileo",
        "subfolder": "models/nano",
        "filename": "encoder.pt"
    }

    if not ckpt_data:
      ckpt_data = remote_checkpoint_path

    model = Galileo(**kwargs)

    if pretrained:
      model = load_weights(model, ckpt_data)

    return model 

class Galileo(torch.nn.Module):

  def __init__(self,
              max_patch_size: int = 8,
              embedding_size: int = 128,
              depth=2,
              mlp_ratio=2,
              num_heads=8,
              max_sequence_length=24,
              freeze_projections: bool = False,
              drop_path: float = 0.0,
            ):

      super(Galileo, self).__init__()

      # Checking if the package galileo is installed.
      try:
          from galileo.galileo import Encoder
      except ModuleNotFoundError:
          raise Exception("It's necessary to install the package `galileo` to access these models: `pip install terratorch[galileo]`")

      self.encoder = Encoder(max_patch_size = max_patch_size,
                          embedding_size = embedding_size,
                          depth=depth,
                          mlp_ratio=mlp_ratio,
                          num_heads=num_heads,
                          max_sequence_length=max_sequence_length,
                          freeze_projections=freeze_projections,
                          drop_path = drop_path,
                  )

  def forward(self, **kwargs):

    return self.encoder(**kwargs)

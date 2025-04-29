import torch
from single_file_galileo import Encoder
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

@TERRATORCH_BACKBONE_REGISTRY.register
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

      self.encoder = Encoder(max_patch_size = max_patch_size,
                          embedding_size = embedding_size,
                          depth=depth,
                          mlp_ratio=mlp_ratio,
                          num_heads=num_heads,
                          max_sequence_length=max_sequence_length,
                          freeze_projections=freeze_projections,
                          drop_path = drop_path,
                  )

  def forward(self, x):

    return self.encoder(x)

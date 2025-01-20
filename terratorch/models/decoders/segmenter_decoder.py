import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
import warnings

from terratorch.registry import TERRATORCH_DECODER_REGISTRY
from terratorch.models.backbones.multimae.multimae_utils import Attention, Mlp, DropPath

from .utils import ConvModule

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        hidden_features = int(dim*mlp_dim) 
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_features, drop=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# Adapted from MMSegmentation
@TERRATORCH_DECODER_REGISTRY.register
class SegmenterDecoder(nn.Module):

    def __init__(
        self,
        embed_dim:  int = 768,
        n_classes = 2,
        patch_size = 16,
        encoder_dim = 768,
        channels = 6,
        n_layers = 4,
        n_heads = 4,
        mlp_dim = 4,
        drop_path_rate = 0.5,
        dropout = 0.5,):

        super().__init__() 
        self.embed_dim = embed_dim[0]
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.channels = channels
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim
        self.drop_path_rate = drop_path_rate
        self.dropout = dropout


        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.n_layers)]

        self.blocks = nn.ModuleList(
                    [Block(self.embed_dim, self.n_heads, self.mlp_dim, self.dropout, dpr[i]) for i in range(self.n_layers)]
                )

        self.cls_emb = nn.Parameter(torch.randn(1, self.n_cls, self.encoder_dim))
        self.proj_dec = nn.Linear(self.encoder_dim, self.encoder_dim)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(self.encoder_dim, self.encoder_dim))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(self.encoder_dim, self.encoder_dim))

        self.decoder_norm = nn.LayerNorm(self.encoder_dim)
        self.mask_norm = nn.LayerNorm(self.n_classes)

    @property
    def out_channels(self):
        return self.channels

    def forward(self, x):
        print(x[0].shape)

from terratorch.io.file import open_generic_torch_model
from terratorch.models.backbones.prithvi_mae import PrithviMAE
from torch import nn

# Path for a downloaded model
model_weights_path = "./pretrain-vit-base-e199.pth"
model_template = PrithviMAE

model_kwargs = {
    'img_size': 224,
    'patch_size': 16,
    'in_chans': 3,
    'embed_dim': 1024,
    'depth': 24,
    'num_heads': 16,
    'decoder_embed_dim': 512,
    'decoder_depth': 8,
    'decoder_num_heads': 16,
    'mlp_ratio': 4,
    'norm_layer': nn.LayerNorm,
    'norm_pix_loss': False,
}

model = open_generic_torch_model(model=model_template, model_kwargs=model_kwargs, model_weights_path=model_weights_path)

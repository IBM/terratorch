import torch
import numpy as np

from models_mae import MaskedAutoencoderViT

kwargs = {
    "img_size": 224,
    "patch_size": 16,
    "in_chans": 3,
    "embed_dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "decoder_embed_dim": 512,
    "decoder_depth": 8,
    "decoder_num_heads": 16,
    "mlp_ratio": 4.0,
}

vit_mae = MaskedAutoencoderViT(**kwargs)

mask_ratio = 0.75
data = torch.from_numpy(np.random.rand(4, 3, 224, 224).astype("float32"))
latent, _, ids_restore = vit_mae.forward_encoder(data, mask_ratio)
reconstructed = vit_mae.forward_decoder(latent, ids_restore)


print(f"Output shape: {latent.shape}")
print("Done.")

_, reconstructed, _ = vit_mae.forward(data, mask_ratio)

print(f"Output shape: {reconstructed.shape}")
print("Done.")

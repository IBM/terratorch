import logging
import os
from functools import partial

import torch
import wget
from torch import nn

from terratorch.models.backbones.SatMAE.models_mae import MaskedAutoencoderViT
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY


def load_weights(model: nn.Module, ckpt_data: dict, out_dir: str = "/tmp", **kwargs) -> nn.Module:
    logging.getLogger("terratorch").info("Loading weights.")

    if ckpt_data is not None:
        filename = ckpt_data.split("/")[-1]
        ckpt_data_file = os.path.join(out_dir, filename)

        if not os.path.isfile(ckpt_data_file):
            wget.download(ckpt_data, out=out_dir)

        checkpoint_model = torch.load(ckpt_data_file, map_location="cpu", weights_only=True)

        # load pre-trained model
        model.load_state_dict(checkpoint_model, strict=False)
        logging.getLogger("terratorch").info("Weights loaded.")
    else:
        print("There is no available checkpoint.")

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def satmae_mae_vit_base_patch16(pretrained=False, **kwargs):
    ckpt_data = None

    model = MaskedAutoencoderViT(
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    if pretrained:
        model = load_weights(model, ckpt_data)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def satmae_mae_vit_large_patch16(pretrained=False, **kwargs):
    ckpt_data = "https://zenodo.org/record/7369797/files/fmow_pretrain.pth"

    model = MaskedAutoencoderViT(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    if pretrained:
        model = load_weights(model, ckpt_data)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def satmae_mae_vit_huge_patch14(pretrained: bool = False, **kwargs):
    ckpt_data = None

    model = MaskedAutoencoderViT(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    if pretrained:
        model = load_weights(model, ckpt_data)

    return model

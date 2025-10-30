import logging

import huggingface_hub
import torch
from terratorch_surya.models.helio_spectformer import HelioSpectFormer
from torch import nn

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY


def load_weights(model: nn.Module, ckpt_data: dict, **kwargs) -> nn.Module:
    logging.getLogger("terratorch").info("Loading weights.")

    if ckpt_data is not None:
        ckpt_data = huggingface_hub.hf_hub_download(**ckpt_data)

        checkpoint_model = torch.load(ckpt_data, map_location="cpu", weights_only=True)

        # load pre-trained model
        model.load_state_dict(checkpoint_model, strict=False)
        logging.getLogger("terratorch").info("Weights loaded.")

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def heliofm_backbone_surya(
    img_size: int = 4096,
    patch_size: int = 16,
    in_chans: int = 13,
    embed_dim: int = 1280,
    time_embedding: dict = {"type": "linear", "n_queries": None, "time_dim": 2},
    depth: int = 10,
    n_spectral_blocks: int = 2,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.0,
    window_size: int = 2,
    dp_rank: int = 8,
    learned_flow: bool = False,
    use_latitude_in_learned_flow: bool = False,
    init_weights: bool = False,
    checkpoint_layers: list = [],
    rpe: bool = False,
    ensemble: int = None,
    finetune: bool = False,
    ckpt_data: str = None,
    pretrained: bool = False,
    dtype="bfloat16",
):
    remote_checkpoint_path = {"repo_id": "nasa-ibm-ai4science/Surya-1.0", "filename": "surya.366m.v1.pt"}

    if not ckpt_data:
        ckpt_data = remote_checkpoint_path

    model = HelioSpectFormer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        time_embedding=time_embedding,
        depth=depth,
        n_spectral_blocks=n_spectral_blocks,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        window_size=window_size,
        dp_rank=dp_rank,
        learned_flow=learned_flow,
        use_latitude_in_learned_flow=use_latitude_in_learned_flow,
        init_weights=init_weights,
        checkpoint_layers=checkpoint_layers,
        rpe=rpe,
        ensemble=ensemble,
        finetune=finetune,
        dtype=dtype,
    )

    if pretrained:
        model = load_weights(model, ckpt_data)

    return model

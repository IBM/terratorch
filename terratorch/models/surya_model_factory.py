import importlib
import logging
import os
import typing

import torch
from terratorch_surya.downstream_examples.ar_segmentation.models import HelioSpectformer2D
from torch import nn

from terratorch.models.backbones.unet import UNet
from terratorch.models.model import (
    Model,
    ModelFactory,
    ModelOutput,
)
from terratorch.registry import MODEL_FACTORY_REGISTRY


@MODEL_FACTORY_REGISTRY.register
class SuryaModelFactory(ModelFactory):
    def build_model(
        self,
        backbone: str | nn.Module,
        backbone_kwargs: dict,
        aux_decoders: str,
        checkpoint_path: str = None,
        backbone_weights: str = None,
        **kwargs,
    ) -> Model:
        if backbone_kwargs["model"]["model_type"] == "spectformer_lora":
            print0("Initializing spectformer with LoRA.")
            model = HelioSpectformer2D(
                img_size=backbone_kwargs["model"]["img_size"],
                patch_size=backbone_kwargs["model"]["patch_size"],
                in_chans=backbone_kwargs["model"]["in_channels"],
                embed_dim=backbone_kwargs["model"]["embed_dim"],
                time_embedding=backbone_kwargs["model"]["time_embedding"],
                depth=backbone_kwargs["model"]["depth"],
                n_spectral_blocks=backbone_kwargs["model"]["spectral_blocks"],
                num_heads=backbone_kwargs["model"]["num_heads"],
                mlp_ratio=backbone_kwargs["model"]["mlp_ratio"],
                drop_rate=backbone_kwargs["model"]["drop_rate"],
                dtype=backbone_kwargs["dtype"],
                window_size=backbone_kwargs["model"]["window_size"],
                dp_rank=backbone_kwargs["model"]["dp_rank"],
                learned_flow=backbone_kwargs["model"]["learned_flow"],
                use_latitude_in_learned_flow=backbone_kwargs["use_latitude_in_learned_flow"],
                init_weights=backbone_kwargs["model"]["init_weights"],
                checkpoint_layers=backbone_kwargs["model"]["checkpoint_layers"],
                rpe=backbone_kwargs["model"]["rpe"],
                finetune=backbone_kwargs["model"]["finetune"],
                backbone_kwargs=backbone_kwargs,
            )
        elif backbone_kwargs["model"]["model_type"] == "unet":
            print0("Initializing UNet.")
            model = UNet(
                in_chans=backbone_kwargs["model"]["in_channels"],
                embed_dim=backbone_kwargs["model"]["unet_embed_dim"],
                out_chans=1,
                n_blocks=backbone_kwargs["model"]["unet_blocks"],
            )
        else:
            raise ValueError(f"Unknown model type {backbone_kwargs['model']['model_type']}.")

        model.load_state_dict(checkpoint_path)

        return model

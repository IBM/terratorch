# Copyright contributors to the Terratorch project

import gc

import pytest
import torch

from terratorch.tasks.ar_segmentation_task import ArSegmentationTask


def test_ar_segmentation_task():
    B = 8
    C = 6
    T = 3
    H = 256
    W = 256

    model_args = {
        "img_size": 256,
        "in_chans": C,
        "embed_dim": 64,
        "num_heads": 8,
        "time_embedding": {"type": "linear", "n_queries": None, "time_dim": 3},
        "depth": 2,
        "n_spectral_blocks": 0,
        "dp_rank": 2,
    }

    peft_config = {
        "method": "LORA",
        "peft_config_kwargs": {
            "r": 32,  # LoRA rank
            "lora_alpha": 64,  # LoRA alpha parameter
            "target_modules": [
                "q_proj",
                "v_proj",
                "k_proj",
                "out_proj",
                "fc1",
                "fc2",
            ],
        },  # Target modules for LoRA
        "lora_dropout": 0.1,
        "bias": "none",
    }

    task = ArSegmentationTask(
        model="heliofm_backbone_surya_ar_segmentation", model_args=model_args, peft_config=peft_config
    )

    data = {"ts": torch.rand(B, C, T, H, W), "time_delta_input": torch.rand(B, T)}

    assert task.model(data).shape == (B, 1, H, W)

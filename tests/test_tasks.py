# Copyright contributors to the Terratorch project
import sys
import types

import pytest
import torch

from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from terratorch.tasks import ClassificationTask, PixelwiseRegressionTask, SemanticSegmentationTask, ReconstructionTask, ScalarRegressionTask

import gc

NUM_CHANNELS = 6
NUM_CLASSES = 2
EXPECTED_SEGMENTATION_OUTPUT_SHAPE = (1, NUM_CLASSES, 224, 224)
VIT_UPERNET_NECK = [
    {"name": "SelectIndices", "indices": [1, 2, 3, 4]},
    {"name": "ReshapeTokensToImage"},
    {"name": "LearnedInterpolateToPyramidal"},
]


@pytest.fixture(scope="session")
def model_factory() -> str:
    return "EncoderDecoderFactory"

@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("decoder", ["IdentityDecoder"])
@pytest.mark.parametrize("loss", ["mse"])
@pytest.mark.parametrize("lr_overrides", [{"encoder": 0.01}, None])
@pytest.mark.parametrize("num_outputs", [1, 3])
def test_create_scalar_regression_task_encoder_decoder(backbone, decoder, loss, model_factory: str, lr_overrides, num_outputs):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "num_outputs": num_outputs,
    }

    task = ScalarRegressionTask(
        model_args,
        model_factory,
        loss=loss,
        lr_overrides=lr_overrides,
        num_outputs=num_outputs
    )
    
    task.train_step(sample)

    gc.collect()
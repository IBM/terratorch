# Copyright contributors to the Terratorch project

import pytest
import torch

from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from terratorch.tasks import ClassificationTask, PixelwiseRegressionTask, SemanticSegmentationTask

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


@pytest.fixture(scope="session")
def model_input() -> torch.Tensor:
    return torch.ones((1, NUM_CHANNELS, 224, 224))


@pytest.mark.parametrize("backbone", ["prithvi_vit_100", "prithvi_vit_300", "prithvi_swin_B"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder"])
@pytest.mark.parametrize("loss", ["ce", "jaccard", "focal", "dice"])
def test_create_segmentation_task(backbone, decoder, loss, model_factory: str):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "num_classes": NUM_CLASSES,
    }

    if decoder == "UperNetDecoder" and backbone.startswith("prithvi_vit"):
        model_args["necks"] = VIT_UPERNET_NECK
    SemanticSegmentationTask(
        model_args,
        model_factory,
        loss=loss,
    )

    gc.collect()

@pytest.mark.parametrize("backbone", ["prithvi_vit_100", "prithvi_vit_300", "prithvi_swin_B"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder"])
@pytest.mark.parametrize("loss", ["mae", "rmse", "huber"])
def test_create_regression_task(backbone, decoder, loss, model_factory: str):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
    }

    if decoder == "UperNetDecoder" and backbone.startswith("prithvi_vit"):
        model_args["necks"] = VIT_UPERNET_NECK

    PixelwiseRegressionTask(
        model_args,
        model_factory,
        loss=loss,
    )

    gc.collect()

@pytest.mark.parametrize("backbone", ["prithvi_vit_100", "prithvi_vit_300", "prithvi_swin_B"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder"])
@pytest.mark.parametrize("loss", ["ce", "bce", "jaccard", "focal"])
def test_create_classification_task(backbone, decoder, loss, model_factory: str):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "num_classes": NUM_CLASSES,
    }

    if decoder == "UperNetDecoder" and backbone.startswith("prithvi_vit"):
        model_args["necks"] = VIT_UPERNET_NECK

    ClassificationTask(
        model_args,
        model_factory,
        loss=loss,
    )

    gc.collect()

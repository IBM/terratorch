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


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100", "prithvi_eo_v2_300", "prithvi_swin_B"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder", "UNetDecoder"])
@pytest.mark.parametrize("loss", ["ce", "jaccard", "focal", "dice"])
@pytest.mark.parametrize("lr_overrides", [{"encoder": 0.01}, None])
def test_create_segmentation_task(backbone, decoder, loss, model_factory: str, lr_overrides):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "num_classes": NUM_CLASSES,
    }

    if decoder in ["UperNetDecoder", "UNetDecoder"] and backbone.startswith("prithvi_eo"):
        model_args["necks"] = VIT_UPERNET_NECK
    if decoder == "UNetDecoder":
        model_args["decoder_channels"] = [256, 128, 64, 32]
    SemanticSegmentationTask(
        model_args,
        model_factory,
        loss=loss,
        lr_overrides=lr_overrides,
    )

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100", "prithvi_eo_v2_300", "prithvi_swin_B"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder", "UNetDecoder"])
@pytest.mark.parametrize("loss", ["mae", "rmse", "huber"])
@pytest.mark.parametrize("lr_overrides", [{"encoder": 0.01}, None])
def test_create_regression_task(backbone, decoder, loss, model_factory: str, lr_overrides):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
    }

    if decoder in ["UperNetDecoder", "UNetDecoder"] and backbone.startswith("prithvi_eo"):
        model_args["necks"] = VIT_UPERNET_NECK
    if decoder == "UNetDecoder":
        model_args["decoder_channels"] = [256, 128, 64, 32]

    PixelwiseRegressionTask(
        model_args,
        model_factory,
        loss=loss,
        lr_overrides=lr_overrides,
    )

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100", "prithvi_eo_v2_300", "prithvi_swin_B"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder", "UNetDecoder"])
@pytest.mark.parametrize("loss", ["ce", "bce", "jaccard", "focal"])
@pytest.mark.parametrize("lr_overrides", [{"encoder": 0.01}, None])
def test_create_classification_task(backbone, decoder, loss, model_factory: str, lr_overrides):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "num_classes": NUM_CLASSES,
    }

    if decoder in ["UperNetDecoder", "UNetDecoder"] and backbone.startswith("prithvi_eo"):
        model_args["necks"] = VIT_UPERNET_NECK
    if decoder == "UNetDecoder":
        model_args["decoder_channels"] = [256, 128, 64, 32]

    ClassificationTask(
        model_args,
        model_factory,
        loss=loss,
        lr_overrides=lr_overrides,
    )

    gc.collect()

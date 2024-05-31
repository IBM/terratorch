import pytest
import torch

from terratorch.models import PrithviModelFactory
from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from terratorch.tasks import ClassificationTask, PixelwiseRegressionTask, SemanticSegmentationTask

NUM_CHANNELS = 6
NUM_CLASSES = 2
EXPECTED_SEGMENTATION_OUTPUT_SHAPE = (1, NUM_CLASSES, 224, 224)


@pytest.fixture(scope="session")
def model_factory() -> str:
    return "PrithviModelFactory"


@pytest.fixture(scope="session")
def model_input() -> torch.Tensor:
    return torch.ones((1, NUM_CHANNELS, 224, 224))


@pytest.mark.parametrize("backbone", ["prithvi_vit_100", "prithvi_vit_300"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder"])
@pytest.mark.parametrize("loss", ["ce", "jaccard", "focal", "dice"])
def test_create_segmentation_task(backbone, decoder, loss, model_factory: PrithviModelFactory):
    SemanticSegmentationTask(
        {
            "backbone": backbone,
            "decoder": decoder,
            "in_channels": NUM_CHANNELS,
            "bands": PRETRAINED_BANDS,
            "pretrained": False,
            "num_classes": NUM_CLASSES,
        },
        model_factory,
        loss=loss
    )


@pytest.mark.parametrize("backbone", ["prithvi_vit_100", "prithvi_vit_300"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder"])
@pytest.mark.parametrize("loss", ["mae", "rmse", "huber"])
def test_create_regression_task(backbone, decoder, loss, model_factory: PrithviModelFactory):
    PixelwiseRegressionTask(
        {
            "backbone": backbone,
            "decoder": decoder,
            "in_channels": NUM_CHANNELS,
            "bands": PRETRAINED_BANDS,
            "pretrained": False,
        },
        model_factory,
        loss=loss
    )


@pytest.mark.parametrize("backbone", ["prithvi_vit_100", "prithvi_vit_300"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder"])
@pytest.mark.parametrize("loss", ["ce", "bce", "jaccard", "focal"])
def test_create_classification_task(backbone, decoder, loss, model_factory: PrithviModelFactory):
    ClassificationTask(
        {
            "backbone": backbone,
            "decoder": decoder,
            "in_channels": NUM_CHANNELS,
            "bands": PRETRAINED_BANDS,
            "pretrained": False,
            "num_classes": NUM_CLASSES,
        },
        model_factory,
        loss=loss
    )

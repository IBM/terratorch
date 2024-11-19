# Copyright contributors to the Terratorch project

import pytest
import torch

from terratorch.models import SMPModelFactory
from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS

import gc 

NUM_CHANNELS = 6
NUM_CLASSES = 2
EXPECTED_SEGMENTATION_OUTPUT_SHAPE = (1, NUM_CLASSES, 224, 224)
EXPECTED_REGRESSION_OUTPUT_SHAPE = (1, 224, 224)
EXPECTED_CLASSIFICATION_OUTPUT_SHAPE = (1, NUM_CLASSES)


@pytest.fixture(scope="session")
def model_factory() -> SMPModelFactory:
    return SMPModelFactory()


@pytest.fixture(scope="session")
def model_input() -> torch.Tensor:
    return torch.ones((1, NUM_CHANNELS, 224, 224))


@pytest.mark.parametrize("backbone", ["timm-regnetx_002"])
@pytest.mark.parametrize("model", ["Unet", "DeepLabV3"])
def test_create_segmentation_model(backbone, model, model_factory: SMPModelFactory, model_input):
    model = model_factory.build_model(
        "segmentation",
        backbone=backbone,
        model=model,
        in_channels=NUM_CHANNELS,
        bands=PRETRAINED_BANDS,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_SEGMENTATION_OUTPUT_SHAPE

    gc.collect()

@pytest.mark.parametrize("backbone", ["timm-regnetx_002"])
@pytest.mark.parametrize("model", ["Unet", "DeepLabV3"])
def test_create_segmentation_model_no_in_channels(backbone, model, model_factory: SMPModelFactory, model_input):
    model = model_factory.build_model(
        "segmentation",
        backbone=backbone,
        model=model,
        bands=PRETRAINED_BANDS,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_SEGMENTATION_OUTPUT_SHAPE

    gc.collect()

@pytest.mark.parametrize("backbone", ["timm-regnetx_002"])
@pytest.mark.parametrize("model", ["Unet", "DeepLabV3"])
def test_create_model_with_extra_bands(backbone, model, model_factory: SMPModelFactory):
    model = model_factory.build_model(
        "segmentation",
        backbone=backbone,
        model=model,
        in_channels=NUM_CHANNELS + 1,
        bands=[*PRETRAINED_BANDS, 7],  # add an extra band
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.eval()
    model_input = torch.ones((1, NUM_CHANNELS + 1, 224, 224))
    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_SEGMENTATION_OUTPUT_SHAPE

    gc.collect()

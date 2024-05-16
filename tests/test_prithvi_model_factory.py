import os

import pytest
import torch

from terratorch.models import PrithviModelFactory
from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS

#from terratorch.models.backbones.prithvi_vit import default_cfgs as vit_default_cfgs
from terratorch.models.model import AuxiliaryHead

NUM_CHANNELS = 6
NUM_CLASSES = 2
EXPECTED_SEGMENTATION_OUTPUT_SHAPE = (1, NUM_CLASSES, 224, 224)
EXPECTED_REGRESSION_OUTPUT_SHAPE = (1, 224, 224)


@pytest.fixture(scope="session")
def model_factory() -> PrithviModelFactory:
    return PrithviModelFactory()


@pytest.fixture(scope="session")
def model_input() -> torch.Tensor:
    return torch.ones((1, NUM_CHANNELS, 224, 224))

@pytest.mark.parametrize("backbone", ["prithvi_vit_100", "prithvi_vit_300"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder"])
def test_create_segmentation_model(backbone, decoder, model_factory: PrithviModelFactory, model_input):
    model = model_factory.build_model(
        "segmentation",
        backbone=backbone,
        decoder=decoder,
        in_channels=NUM_CHANNELS,
        bands=PRETRAINED_BANDS,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_SEGMENTATION_OUTPUT_SHAPE


@pytest.mark.parametrize("backbone", ["prithvi_vit_100", "prithvi_vit_300"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder"])
def test_create_segmentation_model_with_aux_heads(backbone, decoder, model_factory: PrithviModelFactory, model_input):
    aux_heads_name = ["first_aux", "second_aux"]
    model = model_factory.build_model(
        "segmentation",
        backbone=backbone,
        decoder=decoder,
        in_channels=NUM_CHANNELS,
        bands=PRETRAINED_BANDS,
        pretrained=False,
        num_classes=NUM_CLASSES,
        aux_decoders=[AuxiliaryHead(name, "FCNDecoder", None) for name in aux_heads_name],
    )
    model.eval()

    with torch.no_grad():
        model_output = model(model_input)
        assert model_output.output.shape == EXPECTED_SEGMENTATION_OUTPUT_SHAPE

        assert len(model_output.auxiliary_heads.keys() & aux_heads_name) == len(aux_heads_name)
        for _, output in model_output.auxiliary_heads.items():
            assert output.shape == EXPECTED_SEGMENTATION_OUTPUT_SHAPE


@pytest.mark.parametrize("backbone", ["prithvi_vit_100", "prithvi_vit_300"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder"])
def test_create_regression_model(backbone, decoder, model_factory: PrithviModelFactory, model_input):
    model = model_factory.build_model(
        "regression",
        backbone=backbone,
        decoder=decoder,
        in_channels=NUM_CHANNELS,
        bands=PRETRAINED_BANDS,
        pretrained=False,
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_REGRESSION_OUTPUT_SHAPE


@pytest.mark.parametrize("backbone", ["prithvi_vit_100", "prithvi_vit_300"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder"])
def test_create_regression_model_with_aux_heads(backbone, decoder, model_factory: PrithviModelFactory, model_input):
    aux_heads_name = ["first_aux", "second_aux"]
    model = model_factory.build_model(
        "regression",
        backbone=backbone,
        decoder=decoder,
        in_channels=NUM_CHANNELS,
        bands=PRETRAINED_BANDS,
        pretrained=False,
        aux_decoders=[AuxiliaryHead(name, "FCNDecoder", None) for name in aux_heads_name],
    )
    model.eval()

    with torch.no_grad():
        model_output = model(model_input)
        assert model_output.output.shape == EXPECTED_REGRESSION_OUTPUT_SHAPE

        assert len(model_output.auxiliary_heads.keys() & aux_heads_name) == len(aux_heads_name)
        for _, output in model_output.auxiliary_heads.items():
            assert output.shape == EXPECTED_REGRESSION_OUTPUT_SHAPE


@pytest.mark.parametrize("backbone", ["prithvi_vit_100", "prithvi_vit_300"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder"])
def test_create_model_with_extra_bands(backbone, decoder, model_factory: PrithviModelFactory):
    model = model_factory.build_model(
        "segmentation",
        backbone=backbone,
        decoder=decoder,
        in_channels=NUM_CHANNELS,
        bands=[*PRETRAINED_BANDS, 7],  # add an extra band
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.eval()
    model_input = torch.ones((1, NUM_CHANNELS + 1, 224, 224))
    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_SEGMENTATION_OUTPUT_SHAPE

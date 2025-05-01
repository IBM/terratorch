# Copyright contributors to the Terratorch project

import importlib

import pytest
import torch

from terratorch.models import EncoderDecoderFactory
from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from terratorch.models.model import AuxiliaryHead
from terratorch.models.backbones import torchgeo_resnet as torchgeo_resnet

NUM_CHANNELS = 6
NUM_CLASSES = 10
EXPECTED_SEGMENTATION_OUTPUT_SHAPE = (1, NUM_CLASSES, 224, 224)
EXPECTED_REGRESSION_OUTPUT_SHAPE = (1, 224, 224)
EXPECTED_CLASSIFICATION_OUTPUT_SHAPE = (1, NUM_CLASSES)

PIXELWISE_TASK_EXPECTED_OUTPUT = [
    ("regression", EXPECTED_REGRESSION_OUTPUT_SHAPE),
    ("segmentation", EXPECTED_SEGMENTATION_OUTPUT_SHAPE),
]

VIT_UPERNET_NECK = [
    {"name": "SelectIndices", "indices": [0, 1, 2, 3]},
    {"name": "ReshapeTokensToImage"},
    {"name": "LearnedInterpolateToPyramidal"},
]

PRETRAINED_BANDS = ["RED", "GREEN", "BLUE", "NIR_NARROW", "SWIR_1", "SWIR_2"]

@pytest.fixture(scope="session")
def model_factory() -> EncoderDecoderFactory:
    return EncoderDecoderFactory()


@pytest.fixture(scope="session")
def model_input() -> torch.Tensor:
    return torch.ones((1, NUM_CHANNELS, 224, 224))

def torchgeo_resnet_backbones():
    return [i for i in dir(torchgeo_resnet) if "_resnet" in i and i != "load_resnet_weights"]

backbones = torchgeo_resnet_backbones()
pretrained = [False]
@pytest.mark.parametrize("backbone", backbones)
@pytest.mark.parametrize("backbone_pretrained", pretrained)
def test_create_classification_model_resnet(backbone, model_factory: EncoderDecoderFactory, model_input, backbone_pretrained):
    model = model_factory.build_model(
        "classification",
        backbone=backbone,
        decoder="IdentityDecoder",
        backbone_model_bands=PRETRAINED_BANDS,
        backbone_pretrained=backbone_pretrained,
        num_classes=NUM_CLASSES,
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_CLASSIFICATION_OUTPUT_SHAPE

backbones = ["ssl4eos12_resnet50_sentinel2_all_decur"]
pretrained = [True]
@pytest.mark.parametrize("backbone", backbones)
@pytest.mark.parametrize("backbone_pretrained", pretrained)
def test_create_classification_model_resnet_pretrained(backbone, model_factory: EncoderDecoderFactory, model_input, backbone_pretrained):
    model = model_factory.build_model(
        "classification",
        backbone=backbone,
        decoder="IdentityDecoder",
        backbone_model_bands=PRETRAINED_BANDS,
        backbone_pretrained=backbone_pretrained,
        num_classes=NUM_CLASSES,
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_CLASSIFICATION_OUTPUT_SHAPE

backbones = ["dofa_large_patch16_224"]
@pytest.mark.parametrize("backbone", backbones)
@pytest.mark.parametrize("backbone_pretrained", pretrained)
def test_create_classification_model_dofa(backbone, model_factory: EncoderDecoderFactory, model_input, backbone_pretrained):
    model = model_factory.build_model(
        "classification",
        backbone=backbone,
        decoder="IdentityDecoder",
        backbone_model_bands=PRETRAINED_BANDS,
        backbone_pretrained=backbone_pretrained,
        num_classes=NUM_CLASSES,
        necks = [{"name": "PermuteDims", "new_order": [0, 2, 1]}]
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_CLASSIFICATION_OUTPUT_SHAPE

backbones = ["satlas_swin_b_sentinel2_si_ms"]
@pytest.mark.parametrize("backbone", backbones)
@pytest.mark.parametrize("backbone_pretrained", pretrained)
def test_create_classification_model_swin(backbone, model_factory: EncoderDecoderFactory, model_input, backbone_pretrained):
    model = model_factory.build_model(
        "classification",
        backbone=backbone,
        decoder="IdentityDecoder",
        backbone_model_bands=PRETRAINED_BANDS,
        backbone_pretrained=backbone_pretrained,
        num_classes=NUM_CLASSES,
        necks = [{"name": "PermuteDims", "new_order": [0, 3, 1, 2]}]
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_CLASSIFICATION_OUTPUT_SHAPE

@pytest.mark.parametrize("backbone", ["ssl4eos12_resnet50_sentinel2_all_decur"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["FCNDecoder", "IdentityDecoder", "smp_Unet"])
@pytest.mark.parametrize("backbone_pretrained", pretrained)
def test_create_pixelwise_model_resnet(backbone, task, expected, decoder, model_factory: EncoderDecoderFactory, model_input, backbone_pretrained):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "backbone_model_bands": PRETRAINED_BANDS,
        "backbone_pretrained": backbone_pretrained,
        "backbone_out_indices": [0, 1, 2, 3, 4], 
        
    }
        
    if decoder == "smp_Unet":
        model_args["decoder_decoder_channels"] = [512, 256, 128, 64]
    
    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected



@pytest.mark.parametrize("backbone", ["dofa_large_patch16_224"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder"])
@pytest.mark.parametrize("backbone_pretrained", pretrained)
def test_create_pixelwise_model_dofa(backbone, task, expected, decoder, model_factory: EncoderDecoderFactory, model_input, backbone_pretrained):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "backbone_model_bands": PRETRAINED_BANDS,
        "backbone_pretrained": backbone_pretrained,
        "backbone_out_indices": [5, 11, 17, 23]
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES
        
    model_args["necks"] = [{"name": "ReshapeTokensToImage"}]

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected


@pytest.mark.parametrize("backbone", ["satlas_swin_b_sentinel2_si_ms"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["UperNetDecoder", "IdentityDecoder"])
@pytest.mark.parametrize("backbone_pretrained", pretrained)
def test_create_pixelwise_model_swin(backbone, task, expected, decoder, model_factory: EncoderDecoderFactory, model_input, backbone_pretrained):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "backbone_model_bands": PRETRAINED_BANDS,
        "backbone_pretrained": backbone_pretrained,
        "backbone_out_indices": [1, 3, 5, 7]
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES
        
    model_args["necks"] = [{"name": "PermuteDims", "new_order": [0, 3, 1, 2]}]

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected



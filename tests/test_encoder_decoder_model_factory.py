# Copyright contributors to the Terratorch project

import importlib

import pytest
import torch

from terratorch.models import EncoderDecoderFactory
from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from terratorch.models.model import AuxiliaryHead
import gc

NUM_CHANNELS = 6
NUM_CLASSES = 2
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


@pytest.fixture(scope="session")
def model_factory() -> EncoderDecoderFactory:
    return EncoderDecoderFactory()


@pytest.fixture(scope="session")
def model_input() -> torch.Tensor:
    return torch.ones((1, NUM_CHANNELS, 224, 224))


def test_unused_args_raise_exception(model_factory: EncoderDecoderFactory):
    with pytest.raises(ValueError) as excinfo:
        model_factory.build_model(
            "classification",
            backbone="prithvi_eo_v1_100",
            decoder="IdentityDecoder",
            backbone_bands=PRETRAINED_BANDS,
            backbone_pretrained=False,
            num_classes=NUM_CLASSES,
            unused_argument="unused_argument",
        )
    assert "unused_argument" in str(excinfo.value)

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100", "prithvi_eo_v2_300"])
def test_create_classification_model(backbone, model_factory: EncoderDecoderFactory, model_input):
    model = model_factory.build_model(
        "classification",
        backbone=backbone,
        decoder="IdentityDecoder",
        backbone_bands=PRETRAINED_BANDS,
        backbone_pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_CLASSIFICATION_OUTPUT_SHAPE

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100", "prithvi_eo_v2_300"])
def test_create_classification_model_no_in_channels(backbone, model_factory: EncoderDecoderFactory, model_input):
    model = model_factory.build_model(
        "classification",
        backbone=backbone,
        decoder="IdentityDecoder",
        backbone_bands=PRETRAINED_BANDS,
        backbone_pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_CLASSIFICATION_OUTPUT_SHAPE

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder", "UNetDecoder"])
def test_create_pixelwise_model(backbone, task, expected, decoder, model_factory: EncoderDecoderFactory, model_input):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES
    if decoder in ["UperNetDecoder", "UNetDecoder"] and backbone.startswith("prithvi_eo"):
        model_args["necks"] = VIT_UPERNET_NECK
    if decoder == "UNetDecoder":
        model_args["decoder_channels"] = [256, 128, 64, 32]

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
def test_create_model_with_smp_fpn_decoder(backbone, task, expected, model_factory: EncoderDecoderFactory, model_input):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": "smp_FPN",
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "necks": VIT_UPERNET_NECK,
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
def test_create_model_with_smp_unet_decoder(
    backbone, task, expected, model_factory: EncoderDecoderFactory, model_input
):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": "smp_Unet",
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "decoder_decoder_channels": [256, 128, 64],
        "necks": VIT_UPERNET_NECK,
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()


@pytest.mark.skip(reason="Failing without clear reason.")
@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
def test_create_model_with_smp_deeplabv3plus_decoder(
    backbone, task, expected, model_factory: EncoderDecoderFactory, model_input
):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": "smp_DeepLabV3Plus",
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "necks": VIT_UPERNET_NECK + [{"name": "AddBottleneckLayer"}],
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == tuple(expected)

    gc.collect()


@pytest.mark.skipif(not importlib.util.find_spec("mmseg"), reason="mmsegmentation not installed")
@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
def test_create_model_with_mmseg_fcn_decoder(
    backbone, task, expected, model_factory: EncoderDecoderFactory, model_input
):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": "mmseg_FCNHead",
        "decoder_channels": 128,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "necks": [
            {"name": "SelectIndices", "indices": [-1]},
            {"name": "ReshapeTokensToImage"},
        ],
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES
    else:
        model_args["num_classes"] = 1

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()


@pytest.mark.skipif(not importlib.util.find_spec("mmseg"), reason="mmsegmentation not installed")
@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
def test_create_model_with_mmseg_uperhead_decoder(
    backbone, task, expected, model_factory: EncoderDecoderFactory, model_input
):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": "mmseg_UPerHead",
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "decoder_channels": 256,
        "decoder_in_index": [0, 1, 2, 3],
        "necks": [
            {"name": "SelectIndices", "indices": [0, 1, 2, 3]},
            {"name": "ReshapeTokensToImage"},
        ],
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES
    else:
        model_args["num_classes"] = 1

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder", "UNetDecoder"])
def test_create_pixelwise_model_no_in_channels(
    backbone, task, expected, decoder, model_factory: EncoderDecoderFactory, model_input
):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES
    if decoder in ["UperNetDecoder", "UNetDecoder"] and backbone.startswith("prithvi_eo"):
        model_args["necks"] = VIT_UPERNET_NECK
    if decoder == "UNetDecoder":
        model_args["decoder_channels"] = [256, 128, 64, 32]

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder", "UNetDecoder"])
def test_create_pixelwise_model_with_aux_heads(
    backbone, task, expected, decoder, model_factory: EncoderDecoderFactory, model_input
):
    aux_heads_name = ["first_aux", "second_aux"]
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "backbone_in_chans": NUM_CHANNELS,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "aux_decoders": [AuxiliaryHead(name, "FCNDecoder", None) for name in aux_heads_name],
    }
    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES

    if decoder in ["UperNetDecoder", "UNetDecoder"] and backbone.startswith("prithvi_eo"):
        model_args["necks"] = VIT_UPERNET_NECK
    if decoder == "UNetDecoder":
        model_args["decoder_channels"] = [256, 128, 64, 32]

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        model_output = model(model_input)
        assert model_output.output.shape == expected

        assert len(model_output.auxiliary_heads.keys() & aux_heads_name) == len(aux_heads_name)
        for _, output in model_output.auxiliary_heads.items():
            assert output.shape == expected

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder", "UNetDecoder"])
def test_create_pixelwise_model_with_extra_bands(
    backbone, task, expected, decoder, model_factory: EncoderDecoderFactory
):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "backbone_in_chans": NUM_CHANNELS + 1,
        "backbone_bands": [*PRETRAINED_BANDS, 7],
        "backbone_pretrained": False,
    }
    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES

    if decoder in ["UperNetDecoder", "UNetDecoder"] and backbone.startswith("prithvi_eo"):
        model_args["necks"] = VIT_UPERNET_NECK
    if decoder == "UNetDecoder":
        model_args["decoder_channels"] = [256, 128, 64, 32]
    model = model_factory.build_model(**model_args)
    model.eval()
    model_input = torch.ones((1, NUM_CHANNELS + 1, 224, 224))
    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()

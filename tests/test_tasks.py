# Copyright contributors to the Terratorch project
import sys
import types

import pytest
import torch

from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from terratorch.tasks import ClassificationTask, PixelwiseRegressionTask, SemanticSegmentationTask, ReconstructionTask, ScalarRegressionTask
from terratorch.models.model import ModelOutput

import gc

@pytest.fixture(scope="session")
def model_factory() -> str:
    return "EncoderDecoderFactory"

@pytest.fixture(scope="session")
def model_input(num_outputs: int = 3) -> dict[str, torch.Tensor]:
    image = torch.ones((1, 6, 224, 224))
    mask = torch.zeros((1, num_outputs))
    return {"image": image, "mask": mask}

@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("decoder", ["IdentityDecoder"])
@pytest.mark.parametrize("loss", ["mse"])
@pytest.mark.parametrize("lr_overrides", [{"encoder": 0.01}, None])
@pytest.mark.parametrize("num_classes", [1, 3])
def test_create_scalar_regression_task_encoder_decoder(backbone, decoder, loss, model_factory: str, lr_overrides, num_classes):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "num_classes": num_classes,
    }

    task = ScalarRegressionTask(
        model_args,
        model_factory,
        loss=loss,
        lr_overrides=lr_overrides,
        num_classes=num_classes
    )
    
    batch = {
        "image": torch.ones((1, 6, 224, 224)),
        "mask": torch.zeros((1, num_classes))
    }
    
    output = task.training_step(batch, batch_idx=0)
    assert isinstance(output, torch.Tensor), "Model output is not a tensor"
    assert output.shape[1] == num_classes, f"Expected {num_classes} outputs, got {output.shape[1]}"

    # Dummy ground truth to compute loss
    target = torch.zeros_like(output)
    model_output = ModelOutput(output=output, auxiliary_heads=None)
    loss_dict = task.train_loss_handler.compute_loss(model_output, target, task.criterion, aux_loss_weights=None)
    assert "loss" in loss_dict, "LossHandler did not return 'loss'"
    assert torch.isfinite(loss_dict["loss"]).all(), "Loss contains non-finite values"

    # Metrics sanity check
    task.train_metrics.update(output, target)
    computed_metrics = task.train_metrics.compute()
    assert isinstance(computed_metrics, dict), "Metrics did not return a dict"
    print(f"Loss: {loss_dict['loss'].item()}")
    print(f"Metrics: {computed_metrics}")
    
    gc.collect()
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

@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("decoder", ["IdentityDecoder"])
@pytest.mark.parametrize("loss", ["mse"])
@pytest.mark.parametrize("lr_overrides", [{"encoder": 0.01}, None])
@pytest.mark.parametrize("num_classes", [3])
#@pytest.mark.parametrize("class_weights", [[0.2, 0.3, 0.5]])
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
        num_classes=num_classes,
        #class_weights=class_weights
    )
    
    batch = {
        "image": torch.ones((1, 6, 224, 224)),
        "mask": torch.randn((1, num_classes))
    }
    
    model_output : ModelOutput = task(batch["image"])
    y_hat = model_output.output
    y = batch["mask"]
    
    loss = task.training_step(batch, batch_idx=0)

    # Metrics sanity check
    print(f"y_hat: {y_hat.shape}; y: {y.shape}")
    task.train_metrics.update(y_hat, y)
    computed_metrics = task.train_metrics.compute()
    assert isinstance(computed_metrics, dict), "Metrics did not return a dict"
    print(f"\nMetrics: {computed_metrics}")
  
    gc.collect()
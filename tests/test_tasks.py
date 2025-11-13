# Copyright contributors to the Terratorch project
import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from terratorch.tasks import ScalarRegressionTask, PixelwiseRegressionTask, ClassificationTask
from terratorch.models.model import ModelOutput

import gc

class CustomMSELoss(nn.Module):
    """This is a custom MSELoss from the Hyperview challenge."""
    
    def __init__(self, baseline_outputs=None):
        super().__init__()
        if baseline_outputs is None:
            baseline_outputs = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.register_buffer("baseline_outputs", baseline_outputs)

    def forward(self, y_pred, y_true):
        baseline_outputs = self.baseline_outputs.to(y_true.device)
        mse_model = F.mse_loss(y_pred, y_true, reduction="none").mean(dim=0)
        baseline_tensor = baseline_outputs.unsqueeze(0).expand_as(y_true)
        mse_baseline = F.mse_loss(baseline_tensor, y_true, reduction="none").mean(dim=0)
        normalized_mse = mse_model / mse_baseline
        return normalized_mse.mean()
    
BATCH_SIZE = 2
@pytest.fixture(scope="session")
def model_factory() -> str:
    return "EncoderDecoderFactory"

@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("decoder", ["IdentityDecoder"])
@pytest.mark.parametrize("lr_overrides", [{"encoder": 0.01}, None])
@pytest.mark.parametrize(("num_outputs", "var_weights", "var_names", "loss"), [(1, None, None, "mse"),
                                                                       (1, None, ["P", "K", "Mg", "pH"], "mse"),
                                                                       (4, [0.2, 0.3, 0.4, 0.1], None, "mse"),
                                                                       (4, [0.2, 0.3, 0.4, 0.1], ["P", "K", "Mg", "pH"], CustomMSELoss())])
def test_create_scalar_regression_task_encoder_decoder(backbone, decoder, loss, model_factory: str, 
                                                       lr_overrides, num_outputs, var_weights, var_names):
    
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
        num_outputs=num_outputs,
        var_weights=var_weights,
        var_names=var_names,
    )
    
    batch = {
        "image": torch.ones((BATCH_SIZE, 6, 224, 224)),
        "label": torch.randn((BATCH_SIZE, num_outputs))
    }
    
    model_output : ModelOutput = task(batch["image"])
    y_hat = model_output.output
    y = batch["label"]
    assert y_hat.shape[1] == num_outputs, f"Expected {num_outputs} predicted variables, got {y_hat.shape[1]}"
    assert y_hat.shape == y.shape, "Model output shape and label shape don't match."
    
    loss = task.training_step(batch, batch_idx=0)
    assert loss.ndim == 0, "Expected loss to be a scalar for backprop."
    assert isinstance(loss, torch.Tensor), "Loss is not a Tensor"
    
    # Metrics sanity check
    task.train_metrics.update(y_hat, y)
    computed_metrics = task.train_metrics.compute()
    assert isinstance(computed_metrics, dict), "Metrics did not return a dict"
      
    gc.collect()
    
    
@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("decoder", ["IdentityDecoder"])
@pytest.mark.parametrize("loss", ["mse"])
@pytest.mark.parametrize("lr_overrides", [{"encoder": 0.01}, None])
def test_create_pixel_wise_regression_task_encoder_decoder(backbone, decoder, loss, model_factory: str, lr_overrides):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
    }

    task = PixelwiseRegressionTask(
        model_args,
        model_factory,
        loss=loss,
        lr_overrides=lr_overrides,
    )
    
    batch = {
        "image": torch.ones((BATCH_SIZE, 6, 224, 224)),
        "mask": torch.randn((BATCH_SIZE, 224, 224))
    }
    
    model_output : ModelOutput = task(batch["image"])
    y_hat = model_output.output
    y = batch["mask"]
    assert y_hat.shape == y.shape, "Model output shape and label shape don't match."
    
    loss = task.training_step(batch, batch_idx=0)
    assert loss.ndim == 0, "Expected loss to be a scalar for backprop."
    assert isinstance(loss, torch.Tensor), "Loss is not a Tensor"
    
    # Metrics sanity check
    task.train_metrics.update(y_hat, y)
    computed_metrics = task.train_metrics.compute()
    assert isinstance(computed_metrics, dict), "Metrics did not return a dict"
      
    gc.collect()
    

@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("decoder", ["IdentityDecoder"])
@pytest.mark.parametrize("loss", ["ce"])
@pytest.mark.parametrize("lr_overrides", [{"encoder": 0.01}, None])
@pytest.mark.parametrize("num_classes", [2, 4, 10])
def test_create_classification_task_encoder_decoder(backbone, decoder, loss, model_factory: str, lr_overrides, num_classes):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "num_classes": num_classes,
    }

    task = ClassificationTask(
        model_args,
        model_factory,
        loss=loss,
        lr_overrides=lr_overrides,
    )
    
    batch = {
        "image": torch.ones((BATCH_SIZE, 6, 224, 224)),
        "label": torch.randint(0, num_classes, (BATCH_SIZE,))
    }
    
    model_output : ModelOutput = task(batch["image"])
    y_hat = model_output.output
    y = batch["label"]
  
    assert y_hat.ndim == 2, "Model output should have shape (batch_size, num_classes)"
    assert y_hat.shape[0] == y.shape[0], "Batch size mismatch between model output and labels"
    
    loss = task.training_step(batch, batch_idx=0)
    assert loss.ndim == 0, "Expected loss to be a scalar for backprop."
    assert isinstance(loss, torch.Tensor), "Loss is not a Tensor"
    
    # Metrics sanity check
    task.train_metrics.update(y_hat, y)
    computed_metrics = task.train_metrics.compute()
    assert isinstance(computed_metrics, dict), "Metrics did not return a dict"
      
    gc.collect()
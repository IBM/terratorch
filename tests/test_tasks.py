# Copyright contributors to the Terratorch project
import sys
import types

import pytest
import torch

from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from terratorch.tasks import ScalarRegressionTask
from terratorch.models.model import ModelOutput

import gc

BATCH_SIZE = 2
@pytest.fixture(scope="session")
def model_factory() -> str:
    return "EncoderDecoderFactory"

@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("decoder", ["IdentityDecoder"])
@pytest.mark.parametrize("loss", ["mse"])
@pytest.mark.parametrize("lr_overrides", [{"encoder": 0.01}, None])
@pytest.mark.parametrize(("num_outputs", "class_weights"), [(1, None),
                                                            (3, None), 
                                                            (3, [0.2, 0.3, 0.5])])
def test_create_scalar_regression_task_encoder_decoder(backbone, decoder, loss, model_factory: str, lr_overrides, num_outputs, class_weights):
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
        class_weights=class_weights
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
import os
import logging
from collections.abc import Iterable
from typing import Any, Dict
import numpy as np
import torch 
import pandas as pd
import lightning
from lightning.pytorch.callbacks import Callback
from torchgeo.trainers import BaseTask

from terratorch.models.model import Model
from terratorch.tasks.optimizer_factory import optimizer_factory
from terratorch.tasks.tiled_inference import tiled_inference
from terratorch.models.model import ModelOutput

BATCH_IDX_FOR_VALIDATION_PLOTTING = 10
logger = logging.getLogger("terratorch")


class TerraTorchTask(BaseTask):
    """
    Parent used to share common methods among all the
    tasks implemented in terratorch

    NOTE: This task now expects input batches as dictionaries.
    Standard keys include:
    - "image": The primary input tensor(s).
    - "mask" / "label": The target tensor(s).
    - "coordinates": Tensor [B, 2] of coordinates (optional).
    - "wavelengths": Tensor [Num_Bands] of wavelengths (optional).
    - "time_features": Dict[str, Tensor] of time features (optional).
    Models should accept this batch dictionary in their forward pass.
    """

    def __init__(self, tiled_inference_on_testing: bool = False, path_to_record_metrics: str | None = None):

        self.tiled_inference_on_testing = tiled_inference_on_testing
        self.path_to_record_metrics = path_to_record_metrics

        super().__init__()

    # overwrite early stopping
    def configure_callbacks(self) -> list[Callback]:
        return []

    def configure_models(self) -> None:
        if not hasattr(self, "model_factory"):
            if self.hparams.get("freeze_backbone", False) or self.hparams.get("freeze_decoder", False):
                logger.warning("freeze_backbone and freeze_decoder are ignored if a custom model is provided.")
            # Skipping model factory because custom model is provided
            if hasattr(self, "model") and self.model is not None:
                 logger.info("Using pre-configured custom model. Ensure its forward method accepts a batch dictionary.")
            else:
                 logger.warning("No model_factory found and no custom model provided.")
            return

        task_type = self.hparams.get("task", None) 
        if task_type is None:
             logger.warning("Task type not found in hparams, model factory might fail if it requires it.")
        
        aux_heads = self.hparams.get("aux_heads", None)
        model_args = self.hparams.get("model_args", {})

        self.model: Model = self.model_factory.build_model(
            task_type, aux_decoders=aux_heads, **model_args
        )
        logger.info("Model built. Ensure its forward method accepts a batch dictionary and handles internal metadata processing.")

        if self.hparams.get("freeze_backbone", False):
            if self.hparams.get("peft_config", None) is not None:
                msg = "PEFT should be run with freeze_backbone = False"
                raise ValueError(msg)
            self.model.freeze_encoder()

        if self.hparams.get("freeze_decoder", False):
            self.model.freeze_decoder()

        if self.hparams.get("freeze_head", False):
            self.model.freeze_head()
            
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor | ModelOutput]:
        """Forward pass of the model.
        
        Args:
            batch: A dictionary containing the input batch data, potentially including keys 
                   like "image", "coordinates", "wavelengths", "time_features".

        Returns:
            A dictionary containing the model output, typically under the key "prediction".
            For models returning ModelOutput structure, it might be nested.
        """
        model_output: ModelOutput | torch.Tensor = self.model(batch)

        if isinstance(model_output, ModelOutput):
             return {"prediction": model_output} 
        elif isinstance(model_output, torch.Tensor):
             return {"prediction": ModelOutput(output=model_output)}
        elif isinstance(model_output, dict):
             if "prediction" not in model_output:
                  if "output" in model_output:
                       model_output["prediction"] = model_output.pop("output")
                  else:
                       logger.warning("Model output dict does not contain 'prediction' key.")
             if "prediction" in model_output and isinstance(model_output["prediction"], torch.Tensor):
                  model_output["prediction"] = ModelOutput(output=model_output["prediction"])
             return model_output
        else:
             raise TypeError(f"Unexpected model output type: {type(model_output)}")

    def handle_full_or_tiled_inference(self, batch: Dict[str, Any], num_categories:int=None) -> ModelOutput:

        def model_forward_dict(batch_dict: Dict[str, Any]) -> torch.Tensor:
             output_dict = self(batch_dict)
             prediction = output_dict["prediction"]
             if isinstance(prediction, ModelOutput):
                  return prediction.output
             else:
                  logger.warning("Tiled inference expected ModelOutput, got Tensor directly.")
                  return prediction

        if torch.cuda.is_available():
            device = "GPU Memory"
        else:
            device = "RAM"

        if not self.tiled_inference_on_testing:
            try:
                output_dict = self(batch) 
                model_output: ModelOutput = output_dict["prediction"]
            except (torch.OutOfMemoryError, MemoryError) as e:
                logger.exception(f"Inference on testing failed due to insufficient {device}. Try setting `tiled_inference_on_testing` to `True` in your config.")
                raise e
            except Exception as e:
                 logger.exception("An error occurred during the forward pass in testing.")
                 raise e

        else:
            logger.info("Running tiled inference.")
            logger.info("Notice that the tiled inference WON'T produce the exactly same result as the full inference.")
            
            if "image" not in batch:
                raise KeyError("Tiled inference requires the input batch dictionary to have an 'image' key.")
            x = batch["image"]

            tiled_inference_parameters = self.hparams.get("tiled_inference_parameters", None)

            if tiled_inference_parameters:
                try:
                    y_hat: torch.Tensor = tiled_inference(
                        model_forward_dict,
                        x,
                        num_categories, 
                        tiled_inference_parameters,
                        batch_context=batch,
                    )
                    model_output = ModelOutput(output=y_hat)
                except (torch.OutOfMemoryError, MemoryError) as e:
                     logger.exception("Tiled inference failed due to insufficient memory. Try reducing tile sizes in 'tiled_inference_parameters'.")
                     raise e
                except Exception as e:
                     logger.exception("An error occurred during tiled inference.")
                     raise e
            else:
                raise ValueError("Tiled inference requested ('tiled_inference_on_testing: True') but no 'tiled_inference_parameters' found in hyperparameters.")

        return model_output

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        optimizer_name = self.hparams.get("optimizer", "Adam")

        parameters: Iterable
        learning_rate = self.hparams.get("lr", 1e-3)
        lr_overrides = self.hparams.get("lr_overrides", {})

        if lr_overrides:
            parameters = []
            overridden_params = set()
            for param_name, custom_lr in lr_overrides.items():
                group = [p for n, p in self.named_parameters() if param_name in n and p.requires_grad]
                if group:
                     parameters.append({"params": group, "lr": custom_lr})
                     overridden_params.update(n for n, p in self.named_parameters() if param_name in n)
                else:
                     logger.warning(f"Parameter name '{param_name}' in lr_overrides did not match any trainable parameters.")
            
            rest_p = [
                p
                for n, p in self.named_parameters()
                if n not in overridden_params and p.requires_grad
            ]
            if rest_p:
                parameters.append({"params": rest_p, "lr": learning_rate})
            
            if not parameters:
                 logger.warning("No trainable parameters found after applying lr_overrides.")
                 parameters = [p for p in self.parameters() if p.requires_grad]

        else:
            parameters = [p for p in self.parameters() if p.requires_grad]

        optimizer_hparams = self.hparams.get("optimizer_hparams", None)
        scheduler = self.hparams.get("scheduler", None)
        monitor = getattr(self, "monitor", "val/loss")
        scheduler_hparams = self.hparams.get("scheduler_hparams", None)

        return optimizer_factory(
            optimizer_name,
            learning_rate,
            parameters,
            optimizer_hparams,
            scheduler,
            monitor,
            scheduler_hparams,
        )

    def on_train_epoch_end(self) -> None:
        if hasattr(self, "train_metrics"):
             self.log_dict(self.train_metrics.compute(), sync_dist=True)
             self.train_metrics.reset()
        else:
             logger.warning("train_metrics not found, skipping logging.")

    def on_validation_epoch_end(self) -> None:
        if hasattr(self, "val_metrics"):
             self.log_dict(self.val_metrics.compute(), sync_dist=True)
             self.val_metrics.reset()
        else:
             logger.warning("val_metrics not found, skipping logging.")

    def on_test_epoch_end(self) -> None:
        if hasattr(self, "test_metrics") and isinstance(self.test_metrics, list):
            for metrics in self.test_metrics:
                try:
                     self.log_dict(metrics.compute(), sync_dist=True)
                     metrics.reset()
                except Exception as e:
                     logger.error(f"Error computing or logging test metrics: {e}")
        elif hasattr(self, "test_metrics"):
            try:
                self.log_dict(self.test_metrics.compute(), sync_dist=True)
                self.test_metrics.reset()
            except Exception as e:
                logger.error(f"Error computing or logging test metrics: {e}")
        else:
             logger.warning("test_metrics not found or not a list, skipping logging.")

    def _do_plot_samples(self, batch_index):
        plot_on_val = getattr(self, "plot_on_val", 0)
        if not plot_on_val:
            return False

        can_log = (
            hasattr(self.trainer, "datamodule")
            and self.logger is not None
            and hasattr(self.logger, "experiment")
            and (hasattr(self.logger.experiment, "add_figure") or hasattr(self.logger.experiment, "log_figure"))
        )

        return (
            batch_index < BATCH_IDX_FOR_VALIDATION_PLOTTING
            and can_log
            and not self.current_epoch % plot_on_val
        )

    def record_metrics(self, dataloader_idx, y_hat_hard, y):

        if self.path_to_record_metrics:
            if not hasattr(self, "test_metrics") or not isinstance(self.test_metrics, list) or dataloader_idx >= len(self.test_metrics):
                 logger.error(f"Cannot record metrics: test_metrics not available for dataloader_idx {dataloader_idx}.")
                 return
                 
            try:
                metrics_instance = self.test_metrics[dataloader_idx]
                metrics_instance.update(y_hat_hard, y) 
                metrics_record_ = metrics_instance.compute()

                metrics_record_dict = {k: float(np.array(v.detach().cpu())) for k, v in metrics_record_.items()} 
                metrics_record_list = [{"Metric": k, "Value": v} for k, v in metrics_record_dict.items()]
                metrics_record = pd.DataFrame(data=metrics_record_list)

                filename = os.path.join(self.path_to_record_metrics, f"metrics_dl_{dataloader_idx}.csv")
                metrics_record.to_csv(filename, index=False)
                logger.info(f"Metrics for dataloader {dataloader_idx} saved to {filename}")
            except Exception as e:
                 logger.error(f"Error recording metrics for dataloader {dataloader_idx}: {e}")


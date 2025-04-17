import os
import logging
from collections.abc import Iterable
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
    """

    def __init__(self, task: str | None = None, tiled_inference_on_testing: bool = False, path_to_record_metrics: str = False):

        self.task = task
        self.tiled_inference_on_testing = tiled_inference_on_testing
        self.path_to_record_metrics = path_to_record_metrics

        super().__init__()

    # overwrite early stopping
    def configure_callbacks(self) -> list[Callback]:
        return []

    def configure_models(self) -> None:
        if not hasattr(self, "model_factory"):
            if self.hparams["freeze_backbone"] or self.hparams["freeze_decoder"]:
                logger.warning("freeze_backbone and freeze_decoder are ignored if a custom model is provided.")
            # Skipping model factory because custom model is provided
            return

        self.model: Model = self.model_factory.build_model(
            self.task, aux_decoders=self.aux_heads, **self.hparams["model_args"]
        )

        if self.hparams["freeze_backbone"]:
            if self.hparams.get("peft_config", None) is not None:
                msg = "PEFT should be run with freeze_backbone = False"
                raise ValueError(msg)
            self.model.freeze_encoder()

        if self.hparams["freeze_decoder"]:
            self.model.freeze_decoder()

        if self.hparams["freeze_head"]:
            self.model.freeze_head()

    def handle_full_or_tiled_inference(self, x, num_categories:int=None, **rest):

        # When the input sample cannot be fit on memory for some reason
        # the tiled inference is automatically invoked.
        def model_forward(x,  **kwargs):
            return self(x, **kwargs).output

        # The kind of memory device we are considering 
        if torch.cuda.is_available():
            device = "GPU Memory"
        else:
            device = "RAM"

        if not self.tiled_inference_on_testing:
            # When the user don't set the variable `tiled_inference_on_testing`
            # as `True` in the config, we will try to use full inference.
            try:
                model_output: ModelOutput = self(x, **rest)
            except (torch.OutOfMemoryError, MemoryError)  as e:
                raise Exception(f"Inference on testing failed due to insufficient {device}. Try to pass `tiled_inference_on_testing` as `True`, to use tiled inference for it.")

        else:
            logger.info("Running tiled inference.")
            logger.info("Notice that the tiled inference WON'T produce the exactly same result as the full inference.")
            if self.tiled_inference_parameters:
                # Even when tiled inference is chosen and we have a config
                # defined for it, we can have a memory issue when this
                # config isn't suitable. A bad choice for the tile sizes is
                # usually the cause for that.
                try:
                    y_hat: Tensor = tiled_inference(
                        model_forward,
                        x,
                        num_categories, 
                        self.tiled_inference_parameters,
                        **rest,
                    )
                    model_output = ModelOutput(output=y_hat)
                except (torch.OutOfMemoryError, MemoryError) as e:
                    raise Exception("It seems your tiled inference configuration is insufficient. Try to reduce the tile sizes.")
            else:
                raise Exception("You need to define a configuration for the tiled inference.")

        return model_output

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        optimizer = self.hparams["optimizer"]
        if optimizer is None:
            optimizer = "Adam"

        parameters: Iterable
        if self.hparams.get("lr_overrides", None) is not None and len(self.hparams["lr_overrides"]) > 0:
            parameters = []
            for param_name, custom_lr in self.hparams["lr_overrides"].items():
                p = [p for n, p in self.named_parameters() if param_name in n]
                parameters.append({"params": p, "lr": custom_lr})
            rest_p = [
                p
                for n, p in self.named_parameters()
                if all(param_name not in n for param_name in self.hparams["lr_overrides"])
            ]
            parameters.append({"params": rest_p})
        else:
            parameters = self.parameters()

        return optimizer_factory(
            optimizer,
            self.hparams["lr"],
            parameters,
            self.hparams["optimizer_hparams"],
            self.hparams["scheduler"],
            self.monitor,
            self.hparams["scheduler_hparams"],
        )

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        for metrics in self.test_metrics:
            self.log_dict(metrics.compute(), sync_dist=True)
            metrics.reset()

    def _do_plot_samples(self, batch_index):
        if not self.plot_on_val:  # dont plot if self.plot_on_val is 0
            return False

        return (
            batch_index < BATCH_IDX_FOR_VALIDATION_PLOTTING
            and hasattr(self.trainer, "datamodule")
            and self.logger
            and not self.current_epoch % self.plot_on_val  # will be True every self.plot_on_val epochs
            and hasattr(self.logger, "experiment")
            and (hasattr(self.logger.experiment, "add_figure") or hasattr(self.logger.experiment, "log_figure"))
        )

    def record_metrics(self, dataloader_idx, y_hat_hard, y):

        if self.path_to_record_metrics:
            # Recording the metrics
            metrics_record_ = self.test_metrics[dataloader_idx](y_hat_hard, y)
            metrics_record_dict = {k: float(np.array(v.detach().cpu())) for k,v in metrics_record_.items()} 
            metrics_record_list = [{"Metric": k, "Value": v} for k, v in metrics_record_dict.items()]
            metrics_record = pd.DataFrame(data=metrics_record_list)

            filename = os.path.join(self.path_to_record_metrics, "metrics.csv")
            metrics_record.to_csv(filename)


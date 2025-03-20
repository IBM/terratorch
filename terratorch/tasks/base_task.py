import logging
from collections.abc import Iterable

import lightning
from lightning.pytorch.callbacks import Callback
from torchgeo.trainers import BaseTask

from terratorch.models.model import Model
from terratorch.tasks.optimizer_factory import optimizer_factory

BATCH_IDX_FOR_VALIDATION_PLOTTING = 10
logger = logging.getLogger("terratorch")


class TerraTorchTask(BaseTask):
    """
    Parent used to share common methods among all the
    tasks implemented in terratorch
    """

    def __init__(self, task: str | None = None):
        self.task = task

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

"""This module contains the regression task and its auxiliary classes."""

from collections.abc import Sequence
from typing import Any

import lightning
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torch import Tensor, nn
from torchgeo.datasets.utils import unbind_samples
from torchgeo.trainers import BaseTask
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.wrappers.abstract import WrapperMetric

from terratorch.models.model import AuxiliaryHead, Model, ModelOutput
from terratorch.registry.registry import MODEL_FACTORY_REGISTRY
from terratorch.tasks.loss_handler import LossHandler
from terratorch.tasks.optimizer_factory import optimizer_factory
from terratorch.tasks.tiled_inference import TiledInferenceParameters, tiled_inference

BATCH_IDX_FOR_VALIDATION_PLOTTING = 10


class PreTrainingTask(BaseTask):

    """
    Pretaining task.
    """

    def __init__(
        self,
        model_args: dict,
        model_factory: str,
        loss: str = "mse",
        aux_heads: list[AuxiliaryHead] | None = None,
        aux_loss: dict[str, float] | None = None,
        class_weights: list[float] | None = None,
        ignore_index: int | None = None,
        lr: float = 0.001,
        # the following are optional so CLI doesnt need to pass them
        optimizer: str | None = None,
        optimizer_hparams: dict | None = None,
        scheduler: str | None = None,
        scheduler_hparams: dict | None = None,
        #
        freeze_backbone: bool = False,  # noqa: FBT001, FBT002
        freeze_decoder: bool = False,  # noqa: FBT001, FBT002
        plot_on_val: bool | int = 10,
        tiled_inference_parameters: TiledInferenceParameters | None = None,
    ) -> None:
        """Constructor

        Args:
            model_args (Dict): Arguments passed to the model factory.
            model_factory (str): Name of ModelFactory class to be used to instantiate the model.
            loss (str, optional): Loss to be used. Currently, supports 'mse', 'rmse', 'mae' or 'huber' loss.
                Defaults to "mse".
            aux_loss (dict[str, float] | None, optional): Auxiliary loss weights.
                Should be a dictionary where the key is the name given to the loss
                and the value is the weight to be applied to that loss.
                The name of the loss should match the key in the dictionary output by the model's forward
                method containing that output. Defaults to None.
            class_weights (list[float] | None, optional): List of class weights to be applied to the loss.
                Defaults to None.
            ignore_index (int | None, optional): Label to ignore in the loss computation. Defaults to None.
            lr (float, optional): Learning rate to be used. Defaults to 0.001.
            optimizer (str | None, optional): Name of optimizer class from torch.optim to be used.
                If None, will use Adam. Defaults to None. Overriden by config / cli specification through LightningCLI.
            optimizer_hparams (dict | None): Parameters to be passed for instantiation of the optimizer.
                Overriden by config / cli specification through LightningCLI.
            scheduler (str, optional): Name of Torch scheduler class from torch.optim.lr_scheduler
                to be used (e.g. ReduceLROnPlateau). Defaults to None.
                Overriden by config / cli specification through LightningCLI.
            scheduler_hparams (dict | None): Parameters to be passed for instantiation of the scheduler.
                Overriden by config / cli specification through LightningCLI.
            freeze_backbone (bool, optional): Whether to freeze the backbone. Defaults to False.
            freeze_decoder (bool, optional): Whether to freeze the decoder and segmentation head. Defaults to False.
            plot_on_val (bool | int, optional): Whether to plot visualizations on validation.
                If true, log every epoch. Defaults to 10. If int, will plot every plot_on_val epochs.
            tiled_inference_parameters (TiledInferenceParameters | None, optional): Inference parameters
                used to determine if inference is done on the whole image or through tiling.
        """
        self.tiled_inference_parameters = tiled_inference_parameters
        self.aux_loss = aux_loss
        self.aux_heads = aux_heads
        self.model_factory = MODEL_FACTORY_REGISTRY.build(model_factory)
        super().__init__()
        self.train_loss_handler = LossHandler(self.train_metrics.prefix)
        self.test_loss_handler = LossHandler(self.test_metrics.prefix)
        self.val_loss_handler = LossHandler(self.val_metrics.prefix)
        self.monitor = f"{self.val_metrics.prefix}loss"
        self.plot_on_val = int(plot_on_val)

    # overwrite early stopping
    def configure_callbacks(self) -> list[Callback]:
        return []

    def configure_models(self) -> None:
        self.model: Model = self.model_factory.build_model(
            "regression", aux_decoders=self.aux_heads, **self.hparams["model_args"]
        )
        if self.hparams["freeze_backbone"]:
            if self.hparams.get("peft_config", None) is not None:
                msg = "PEFT should be run with freeze_backbone = False"
                raise ValueError(msg)
            self.model.freeze_encoder()
        if self.hparams["freeze_decoder"]:
            self.model.freeze_decoder()

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        optimizer = self.hparams["optimizer"]
        if optimizer is None:
            optimizer = "Adam"
        return optimizer_factory(
            optimizer,
            self.hparams["lr"],
            self.parameters(),
            self.hparams["optimizer_hparams"],
            self.hparams["scheduler"],
            self.monitor,
            self.hparams["scheduler_hparams"],
        )

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"].lower()
        if loss == "mse":
            self.criterion: nn.Module = IgnoreIndexLossWrapper(
                nn.MSELoss(reduction="none"), self.hparams["ignore_index"]
            )
        elif loss == "mae":
            self.criterion = IgnoreIndexLossWrapper(nn.L1Loss(reduction="none"), self.hparams["ignore_index"])
        elif loss == "rmse":
            # IMPORTANT! Root is done only after ignore index! Otherwise the mean taken is incorrect
            self.criterion = RootLossWrapper(
                IgnoreIndexLossWrapper(nn.MSELoss(reduction="none"), self.hparams["ignore_index"]), reduction=None
            )
        elif loss == "huber":
            self.criterion = IgnoreIndexLossWrapper(nn.HuberLoss(reduction="none"), self.hparams["ignore_index"])
        else:
            exception_message = f"Loss type '{loss}' is not valid. Currently, supports 'mse', 'rmse' or 'mae' loss."
            raise ValueError(exception_message)

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""

        def instantiate_metrics():
            return {
                "RMSE": MeanSquaredError(squared=False),
                "MSE": MeanSquaredError(squared=True),
                "MAE": MeanAbsoluteError(),
            }

        def wrap_metrics_with_ignore_index(metrics):
            return {
                name: IgnoreIndexMetricWrapper(metric, ignore_index=self.hparams["ignore_index"])
                for name, metric in metrics.items()
            }

        self.train_metrics = MetricCollection(wrap_metrics_with_ignore_index(instantiate_metrics()), prefix="train/")
        self.val_metrics = MetricCollection(wrap_metrics_with_ignore_index(instantiate_metrics()), prefix="val/")
        self.test_metrics = MetricCollection(wrap_metrics_with_ignore_index(instantiate_metrics()), prefix="test/")

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute the train loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]

        model_output: ModelOutput = self(x)
        loss = self.train_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        x_hat = model_output.output
        self.train_metrics.update(x_hat, x)

        return loss["loss"]

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.train_metrics.reset()
        return super().on_train_epoch_end()

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

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        model_output: ModelOutput = self(x)
        loss = self.val_loss_handler.compute_loss(model_output, x, self.criterion, self.aux_loss)
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        x_hat = model_output.output
        self.val_metrics.update(x_hat, x)

        if self._do_plot_samples(batch_idx):
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = x_hat
                if isinstance(batch["image"], dict):
                    # Multimodal input
                    batch["image"] = batch["image"][self.trainer.datamodule.rgb_modality]
                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.val_dataset.plot(sample)
                if fig:
                    summary_writer = self.logger.experiment
                    if hasattr(summary_writer, "add_figure"):
                        summary_writer.add_figure(f"image/{batch_idx}", fig, global_step=self.global_step)
                    elif hasattr(summary_writer, "log_figure"):
                        summary_writer.log_figure(
                            self.logger.run_id, fig, f"epoch_{self.current_epoch}_{batch_idx}.png"
                        )
            except ValueError:
                pass
            finally:
                plt.close()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()
        return super().on_validation_epoch_end()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]

        model_output: ModelOutput = self(x)
        loss = self.test_loss_handler.compute_loss(model_output, x, self.criterion, self.aux_loss)
        self.test_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        x_hat = model_output.output
        self.test_metrics.update(x_hat, x)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()
        return super().on_test_epoch_end()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        file_names = batch["filename"]
        model_output: ModelOutput = self(x)

        def model_forward(x):
            return self(x).output

        # Tiled inference for autoencoder ? Is it making sense ?
        if self.tiled_inference_parameters:
            x_hat: Tensor = tiled_inference(model_forward, x, 1, self.tiled_inference_parameters)
        else:
            x_hat: Tensor = self(x).output
        return y_hat, file_names

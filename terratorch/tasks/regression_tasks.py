"""This module contains the regression task and its auxiliary classes."""

from collections.abc import Sequence
from typing import Any, Optional

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


class RootLossWrapper(nn.Module):
    def __init__(self, loss_function: nn.Module, reduction: None | str = "mean") -> None:
        super().__init__()
        self.loss_function = loss_function
        self.reduction = reduction

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        loss = torch.sqrt(self.loss_function.forward(output, target))
        if self.reduction is None:
            return loss

        if self.reduction == "mean":
            return loss.mean()

        msg = "Only 'mean' and None reduction supported"
        raise Exception(msg)


class IgnoreIndexLossWrapper(nn.Module):
    """Wrapper over loss that will ignore certain values"""

    def __init__(self, loss_function: nn.Module, ignore_index: int, reduction: None | str = "mean") -> None:
        super().__init__()
        self.loss_function = loss_function
        self.ignore_index = ignore_index
        self.reduction = reduction

    def _mask(self, loss: Tensor, target: Tensor):
        mask = torch.ones_like(target)
        mask[target == self.ignore_index] = 0

        # avoid ZeroDivisionError
        eps = torch.finfo(torch.float32).eps
        return loss * mask.float(), mask.sum() + eps

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        loss: Tensor = self.loss_function(output, target)
        if self.ignore_index is not None:
            loss, norm_value = self._mask(loss, target)
        else:
            norm_value = loss.numel()

        if self.reduction is None:
            return loss

        if self.reduction == "mean":
            return loss.sum() / norm_value

        msg = "Only 'mean' and None reduction supported"
        raise Exception(msg)


class IgnoreIndexMetricWrapper(WrapperMetric):
    """Wrapper over other metric that will ignore certain values.

    This class implements ignore_index by removing values where the target matches the ignored value.
    This will only work for metrics where the shape can be flattened.

    WARNING: This is trickier than it seems. Implementation inspired by https://github.com/Lightning-AI/torchmetrics/blob/99d6d9d6ac4eb1b3398241df558604e70521e6b0/src/torchmetrics/wrappers/classwise.py#L27-L211

    The wrapper needs to change the inputs on the forward directly, so that it affects the update of the wrapped metric
    """

    def __init__(self, wrapped_metric: Metric, ignore_index: int) -> None:
        super().__init__()
        self.metric = wrapped_metric
        self.ignore_index = ignore_index

    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.ignore_index is not None:
            items_to_take = target != self.ignore_index
            target = target[items_to_take]
            preds = preds[items_to_take]
        return self.metric.update(preds, target)

    def forward(self, preds: Tensor, target: Tensor, *args, **kwargs) -> Any:
        if self.ignore_index is not None:
            items_to_take = target != self.ignore_index
            target = target[items_to_take]
            preds = preds[items_to_take]
        return self.metric.forward(preds, target, *args, **kwargs)

    def compute(self) -> Tensor:
        return self.metric.compute()

    def plot(self, val: Tensor | Sequence[Tensor] | None = None, ax=None):
        return self.metric.plot(val, ax)

    def reset(self) -> None:
        self.metric.reset()

class PixelwiseRegressionTask(BaseTask):
    """Pixelwise Regression Task that accepts models from a range of sources.

    This class is analog in functionality to
    [PixelwiseRegressionTask]
    (https://torchgeo.readthedocs.io/en/stable/api/trainers.html#torchgeo.trainers.PixelwiseRegressionTask)
    defined by torchgeo.
    However, it has some important differences:
        - Accepts the specification of a model factory
        - Logs metrics per class
        - Does not have any callbacks by default (TorchGeo tasks do early stopping by default)
        - Allows the setting of optimizers in the constructor"""

    def __init__(
        self,
        model_args: dict,
        model_factory: str,
        model: Optional[torch.nn.Module]=None,
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

        if model_factory:  
            self.model_factory = MODEL_FACTORY_REGISTRY.build(model_factory)
            self.model_builder = self._build
        elif model:
            self.model_builder = self._bypass_build
            self._model_module = model
        else:
            raise Exception("Or a model_factory or a torch.nn.Module object must be provided.")

        super().__init__()

        self.train_loss_handler = LossHandler(self.train_metrics.prefix)
        self.test_loss_handler = LossHandler(self.test_metrics.prefix)
        self.val_loss_handler = LossHandler(self.val_metrics.prefix)
        self.monitor = f"{self.val_metrics.prefix}loss"
        self.plot_on_val = int(plot_on_val)

    @property
    def model_module(self):
        return self._model_module

    def _bypass_build(self):
        return self.model_module

    def _build(self):

        return self.model_factory.build_model(
            "regression", aux_decoders=self.aux_heads, **self.hparams["model_args"]
        )

    # overwrite early stopping
    def configure_callbacks(self) -> list[Callback]:
        return []

    def configure_models(self) -> None:

        self.model: Model = self.model_builder()

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
        y = batch["mask"]
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k:batch[k] for k in other_keys}

        model_output: ModelOutput = self(x, **rest)
        loss = self.train_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        y_hat = model_output.output
        self.train_metrics.update(y_hat, y)

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
        y = batch["mask"]
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k:batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)
        loss = self.val_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat = model_output.output
        self.val_metrics.update(y_hat, y)

        if self._do_plot_samples(batch_idx):
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat
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
        y = batch["mask"]
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k:batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)
        loss = self.test_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.test_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        y_hat = model_output.output
        self.test_metrics.update(y_hat, y)

    @property
    def model_module(self):
        return self._model_module

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
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k:batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)

        def model_forward(x):
            return self(x).output

        if self.tiled_inference_parameters:
            y_hat: Tensor = tiled_inference(model_forward, x, 1, self.tiled_inference_parameters)
        else:
            y_hat: Tensor = self(x, **rest).output
        return y_hat, file_names

class ScalarRegressionTask(PixelwiseRegressionTask):

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

        super().__init__(model_args=model_args,
                     model_factory=model_factory,
                     loss=loss, 
                     aux_heads=aux_heads,
                     aux_loss=aux_loss,
                     class_weights=class_weights,
                     ignore_index=ignore_index,
                     lr=lr,
                     optimizer=optimizer,
                     optimizer_hparams=optimizer_hparams,
                     scheduler=scheduler,
                     scheduler_hparams=scheduler_hparams,
                     freeze_backbone=freeze_backbone,
                     freeze_decoder=freeze_decoder,
                     plot_on_val=plot_on_val,
                     tiled_inference_parameters=tiled_inference_parameters,)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k:batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)
        loss = self.val_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        y_hat = model_output.output
        self.val_metrics.update(y_hat, y)

        ##############
        # Custom plots
        # ############



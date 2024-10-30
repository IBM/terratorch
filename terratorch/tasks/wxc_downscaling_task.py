import torch
from torch import Tensor, nn
from torchgeo.trainers import BaseTask
from typing import Any, Mapping

from terratorch.models.model import Model#, get_factory
from terratorch.registry import MODEL_FACTORY_REGISTRY
from terratorch.tasks.loss_handler import LossHandler
from terratorch.tasks.optimizer_factory import optimizer_factory
from terratorch.tasks.regression_tasks import RootLossWrapper
from granitewxc.utils.config import ExperimentConfig
from torchmetrics import MetricCollection
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection


class WxCDownscalingTask(BaseTask):
    def __init__(
        self,
        model_args: dict,
        model_factory: str,
        model_config: ExperimentConfig,
        loss: str = "mse",
        lr: float = 0.001,
        optimizer: str | None = None,
        optimizer_hparams: dict | None = None,
        scheduler: str | None = None,
        scheduler_hparams: dict | None = None,
        freeze_backbone: bool = False,  # noqa: FBT001, FBT002
        freeze_decoder: bool = False,  # noqa: FBT001, FBT002
        plot_on_val: bool | int = 10,
    ) -> None:
        
        # Special case for mask_init_size, which must be read as tuple:
        model_unit_size = tuple(model_args.pop["model_unit_size"])
        model_config["model_unit_size"] = model_unit_size

        self.model_factory = MODEL_FACTORY_REGISTRY.build(model_factory)
        self.model_config = model_config
        # TODO Unify it with self.hparams
        self.extended_hparams = self.model_config.to_dict()
        super().__init__()
       
        self.train_loss_handler = LossHandler(self.train_metrics.prefix)
        self.test_loss_handler = LossHandler(self.test_metrics.prefix)
        self.val_loss_handler = LossHandler(self.val_metrics.prefix)
        self.monitor = f"{self.val_metrics.prefix}loss"
        self.plot_on_val = int(plot_on_val)

    def configure_models(self) -> None:
        self.model: Model = self.model_factory.build_model(
            "regression", aux_decoders=None, model_config=self.model_config, **self.hparams["model_args"]
        )
        if self.extended_hparams["freeze_backbone"]:
            self.model.freeze_encoder()
        if self.extended_hparams["freeze_decoder"]:
            self.model.freeze_decoder()

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        optimizer = self.hparams["optimizer"]
        if optimizer is None:
            optimizer = "Adam"
        return optimizer_factory(
            self.hparams["optimizer"],
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
        #TODO 'reduction' should be chosen using the config and 
        # a similar class as IgnoreIndex should be defined for this class

        loss: str = self.hparams["loss"].lower()
        if loss == "mse":
            self.criterion: nn.Module = nn.MSELoss(reduction="mean")
        elif loss == "mae":
            self.criterion = nn.L1Loss(reduction="mean")
        elif loss == "rmse":
            # IMPORTANT! Root is done only after ignore index! Otherwise the mean taken is incorrect
            self.criterion = RootLossWrapper(nn.MSELoss(reduction="none"), reduction="none")
        elif loss == "huber":
            self.criterion = nn.HuberLoss(reduction="none")
        else:
            exception_message = f"Loss type '{loss}' is not valid. Currently, supports 'mse', 'rmse' or 'mae' loss."
            raise ValueError(exception_message)

    def configure_metrics(self) -> None:

        def instantiate_metrics():
            return {
                "RMSE": MeanSquaredError(squared=False),
                "MSE": MeanSquaredError(squared=True),
                "MAE": MeanAbsoluteError(),
            }

        self.train_metrics = MetricCollection(instantiate_metrics(), prefix="train/")
        self.val_metrics = MetricCollection(instantiate_metrics(), prefix="val/")
        self.test_metrics = MetricCollection(instantiate_metrics(), prefix="test/")

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:

        x = batch["image"]
        mask = batch["mask"]
        model_output: ModelOutput = self(x)
        y = mask
        loss = self.train_loss_handler.compute_loss(model_output, y, self.criterion, None)
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat = model_output.output
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, on_epoch=True)

        return loss["loss"]

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the train loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        model_output: ModelOutput = self(x)

        loss = self.train_loss_handler.compute_loss(model_output, y, self.criterion, None)
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat = model_output.output
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, on_epoch=True)

        return loss["loss"]

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        raise NotImplementedError("This function is not yet implemented.")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        x = batch["image"]
        file_names = batch["filename"]

        y_hat: Tensor = self(x).output
        return y_hat, file_names

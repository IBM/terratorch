from typing import Any
import logging
import lightning
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback
from segmentation_models_pytorch.losses import FocalLoss, JaccardLoss
from torch import Tensor, nn
from torchgeo.datasets.utils import unbind_samples
from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassFBetaScore, MulticlassJaccardIndex

from terratorch.models.model import AuxiliaryHead, Model, ModelOutput
from terratorch.registry.registry import MODEL_FACTORY_REGISTRY
from terratorch.tasks.loss_handler import LossHandler
from terratorch.tasks.optimizer_factory import optimizer_factory
from terratorch.tasks.base_task import TerraTorchTask

logger = logging.getLogger('terratorch')

def to_class_prediction(y: ModelOutput) -> Tensor:
    y_hat = y.output
    return y_hat.argmax(dim=1)


class ClassificationTask(TerraTorchTask):
    """Classification Task that accepts models from a range of sources.

    This class is analog in functionality to class:ClassificationTask defined by torchgeo.
    However, it has some important differences:
        - Accepts the specification of a model factory
        - Logs metrics per class
        - Does not have any callbacks by default (TorchGeo tasks do early stopping by default)
        - Allows the setting of optimizers in the constructor
        - It provides mIoU with both Micro and Macro averaging

    .. note::
           * 'Micro' averaging suits overall performance evaluation but may not reflect
             minority class accuracy.
           * 'Macro' averaging gives equal weight to each class, useful
             for balanced performance assessment across imbalanced classes.
    """

    def __init__(
        self,
        model_args: dict,
        model_factory: str | None = None,
        model: torch.nn.Module | None = None,
        loss: str = "ce",
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
        #
        freeze_backbone: bool = False,  # noqa: FBT001, FBT002
        freeze_decoder: bool = False,  # noqa: FBT002, FBT001
        class_names: list[str] | None = None,
        reduce_lr: list[tuple[str, float]] | None = None,
    ) -> None:
        """Constructor

        Args:

            Defaults to None.
            model_args (Dict): Arguments passed to the model factory.
            model_factory (str, optional): ModelFactory class to be used to instantiate the model.
                Is ignored when model is provided.
            model (torch.nn.Module, optional): Custom model.
            loss (str, optional): Loss to be used. Currently, supports 'ce', 'jaccard' or 'focal' loss.
                Defaults to "ce".
            aux_loss (dict[str, float] | None, optional): Auxiliary loss weights.
                Should be a dictionary where the key is the name given to the loss
                and the value is the weight to be applied to that loss.
                The name of the loss should match the key in the dictionary output by the model's forward
                method containing that output. Defaults to None.
            class_weights (Union[list[float], None], optional): List of class weights to be applied to the loss.
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
            class_names (list[str] | None, optional): List of class names passed to metrics for better naming.
                Defaults to numeric ordering.
            reduce_lr (list[tuple[str, float]] | None, optional): List of tuples with a substring of the parameter names
                to reduce the learning rate and the factor to reduce it by. Defaults to None.
        """
        self.aux_loss = aux_loss
        self.aux_heads = aux_heads

        if model is not None and model_factory is not None:
            logger.warning("A model_factory and a model was provided. The model_factory is ignored.")
        if model is None and model_factory is None:
            raise ValueError("A model_factory or a model (torch.nn.Module) must be provided.")

        if model_factory and model is None:
            self.model_factory = MODEL_FACTORY_REGISTRY.build(model_factory)

        super().__init__(task="classification")

        if model:
            # Custom model
            self.model = model

        self.train_loss_handler = LossHandler(self.train_metrics.prefix)
        self.test_loss_handler = LossHandler(self.test_metrics.prefix)
        self.val_loss_handler = LossHandler(self.val_metrics.prefix)
        self.monitor = f"{self.val_metrics.prefix}loss"

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"]
        ignore_index = self.hparams["ignore_index"]

        class_weights = (
                    torch.Tensor(self.hparams["class_weights"]) if self.hparams["class_weights"] is not None else None
                )
        if loss == "ce":
            ignore_value = -100 if ignore_index is None else ignore_index
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_value, weight=class_weights)
        elif loss == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss == "jaccard":
            self.criterion = JaccardLoss(mode="multiclass")
        elif loss == "focal":
            self.criterion = FocalLoss(mode="multiclass", normalized=True)
        else:
            msg = f"Loss type '{loss}' is not valid."
            raise ValueError(msg)

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["model_args"]["num_classes"]
        ignore_index: int = self.hparams["ignore_index"]
        class_names = self.hparams["class_names"]
        metrics = MetricCollection(
            {
                "Overall_Accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                "Average_Accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Multiclass_Accuracy_Class": ClasswiseWrapper(
                    MulticlassAccuracy(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    labels=class_names,
                ),
                "Multiclass_Jaccard_Index": MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index),
                "Multiclass_Jaccard_Index_Class": ClasswiseWrapper(
                    MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average=None),
                    labels=class_names,
                ),
                # why FBetaScore
                "Multiclass_F1_Score": MulticlassFBetaScore(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    beta=1.0,
                    average="micro",
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute the train loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["label"]
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k:batch[k] for k in other_keys}

        model_output: ModelOutput = self(x, **rest)
        loss = self.train_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        y_hat_hard = to_class_prediction(model_output)
        self.train_metrics.update(y_hat_hard, y)

        return loss["loss"]

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["label"]
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k:batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)
        loss = self.val_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        y_hat_hard = to_class_prediction(model_output)
        self.val_metrics.update(y_hat_hard, y)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["label"]
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k:batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)
        loss = self.test_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.test_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        y_hat_hard = to_class_prediction(model_output)
        self.test_metrics.update(y_hat_hard, y)

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
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k:batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)

        y_hat = self(x).output
        y_hat = y_hat.argmax(dim=1)
        return y_hat, file_names

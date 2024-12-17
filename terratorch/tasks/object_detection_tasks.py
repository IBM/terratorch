from typing import Any
import lightning
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback
from segmentation_models_pytorch.losses import FocalLoss, JaccardLoss
from torch import Tensor, nn
from torchgeo.datasets.utils import unbind_samples
from torchgeo.trainers import BaseTask
from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassFBetaScore, MulticlassJaccardIndex

from terratorch.models.model import AuxiliaryHead, Model, ModelOutput
from terratorch.registry.registry import MODEL_FACTORY_REGISTRY
from terratorch.tasks.loss_handler import LossHandler
from terratorch.tasks.optimizer_factory import optimizer_factory


def to_class_prediction(y: ModelOutput) -> Tensor:
    y_hat = y.output
    return y_hat.argmax(dim=1)


class ObjectDetectionTask(BaseTask):
    """ObjectDetection Task that accepts models from a range of sources.

    This class is analog in functionality to class:ObjectDetectionTask defined by torchgeo.
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
        model_factory: str,
        model_args: dict,
        loss: str = "ce",
        aux_heads: list[AuxiliaryHead] | None = None,
        aux_loss: dict[str, float] | None = None,
        class_weights: list[float] | None = None,  # Only for loss ce
        ignore_index: int | None = None,
        lr: float = 0.001,

        optimizer: str | None = None,
        optimizer_hparams: dict | None = None,
        scheduler: str | None = None,
        scheduler_hparams: dict | None = None,

        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        class_names: list[str] | None = None,
    ) -> None:
        """Constructor

        Args:

            Defaults to None.
            model_args (Dict): Arguments passed to the model factory.
            model_factory (str): ModelFactory class to be used to instantiate the model.
            loss (str, optional): Loss to be used. Currently, supports 'ce', 'jaccard' or 'focal' loss.
                Defaults to "ce".
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
            class_names (list[str] | None, optional): List of class names passed to metrics for better naming.
                Defaults to numeric ordering.
        """
        self.aux_loss = aux_loss
        self.aux_heads = aux_heads
        self.model_factory = MODEL_FACTORY_REGISTRY.build(model_factory)
        super().__init__()
        self.train_loss_handler = LossHandler(self.train_metrics.prefix)
        self.test_loss_handler = LossHandler(self.test_metrics.prefix)
        self.val_loss_handler = LossHandler(self.val_metrics.prefix)
        self.monitor = f"{self.val_metrics.prefix}loss"

    # overwrite early stopping
    def configure_callbacks(self) -> list[Callback]:
        return []

    def configure_models(self) -> None:
        self.model: Model = self.model_factory.build_model(
            "object_detection", aux_decoders=self.aux_heads, **self.hparams["model_args"]
        )
        if self.hparams["freeze_backbone"]:
            self.model.freeze_encoder()
        if self.hparams["freeze_decoder"]:
            self.model.freeze_decoder()

    def configure_optimizers(  # FIXME
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

    def configure_losses(self) -> None:  # FIXME
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"]
        if loss == "ce":
            self.criterion: nn.Module = nn.CrossEntropyLoss(weight=self.hparams["class_weights"])
        elif loss == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss == "jaccard":
            self.criterion = JaccardLoss(mode="multiclass")
        elif loss == "focal":
            self.criterion = FocalLoss(mode="multiclass", normalized=True)
        else:
            msg = f"Loss type '{loss}' is not valid."
            raise ValueError(msg)

    def configure_metrics(self) -> None:  # FIXME
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
        print(f'XXX ObjectDetectionTask.training_step() {batch_idx=} {dataloader_idx=}')
        x = batch["image"]
        batch_size = len(x)  # Set batch size (number of images)
        y = [
            {"boxes": batch["boxes"][i], "labels": batch["labels"][i]}
            for i in range(batch_size)
        ] # Extract bounding box and label information for each image
        torchgeo_way = True
        if torchgeo_way:
            model_output = self(x, y)  # Loss
            loss_dict = model_output.output
            print(f'XXX: training_step:')
            print(f'XXX: {len(x)=}')
            print(f'XXX: {x[0].size()}')
            print(f'XXX: {len(y)=}')
            print(f"XXX: {len(y[0]['labels'])=}")
            print(f"XXX: {y[0]['labels'][0].size()=}")
            print(f"XXX: {y[0]['labels'][0]=}")
            print(f"XXX: {y[0]['boxes'][0].size()=}")
            print(f'XXX: {type(loss_dict)=}')
            print(f'XXX: {loss_dict.keys()=}')
            print(f"XXX: {loss_dict['loss_classifier']=}")
            print(f"XXX: {loss_dict['loss_box_reg']=}")
            print(f"XXX: {loss_dict['loss_objectness']=}")
            print(f"XXX: {loss_dict['loss_rpn_box_reg']=}")
            train_loss: Tensor = sum(loss_dict.values())  # Training loss (sum of loss values)
            self.log_dict(loss_dict)  # Record loss values
            return train_loss  # Return training loss
        model_output: ModelOutput = self(x, y)
        loss = self.train_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        y_hat_hard = to_class_prediction(model_output)
        self.train_metrics.update(y_hat_hard, y)

        return loss["loss"]

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.train_metrics.reset()
        return super().on_train_epoch_end()

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        print(f'XXX ObjectDetectionTask.validation_step() {batch_idx=} {dataloader_idx=}')
        x = batch["image"]
        batch_size = len(x)  # Set batch size (number of images)
        y = [
            {"boxes": batch["boxes"][i], "labels": batch["labels"][i]}
            for i in range(batch_size)
        ]
        torchgeo_way = True
        if torchgeo_way:
            y_hat = self(x).output
            metrics = self.val_metrics(y_hat, y)

            # https://github.com/Lightning-AI/torchmetrics/pull/1832#issuecomment-1623890714
            metrics.pop('val_classes', None)

            self.log_dict(metrics, batch_size=batch_size)

            if (
                batch_idx < 10
                and hasattr(self.trainer, 'datamodule')
                and hasattr(self.trainer.datamodule, 'plot')
                and self.logger
                and hasattr(self.logger, 'experiment')
                and hasattr(self.logger.experiment, 'add_figure')
            ):
                datamodule = self.trainer.datamodule
                batch['prediction_boxes'] = [b['boxes'].cpu() for b in y_hat]
                batch['prediction_labels'] = [b['labels'].cpu() for b in y_hat]
                batch['prediction_scores'] = [b['scores'].cpu() for b in y_hat]
                batch['image'] = batch['image'].cpu()
                sample = unbind_samples(batch)[0]
                # Convert image to uint8 for plotting
                if torch.is_floating_point(sample['image']):
                    sample['image'] *= 255
                    sample['image'] = sample['image'].to(torch.uint8)

                fig: Figure | None = None
                try:
                    fig = datamodule.plot(sample)
                except RGBBandsMissingError:
                    pass

                if fig:
                    summary_writer = self.logger.experiment
                    summary_writer.add_figure(
                        f'image/{batch_idx}', fig, global_step=self.global_step
                    )
                    plt.close()
            model_output = self(x, y)  # Loss  # torchgeo way
            loss_dict = model_output.output
            # model_output.auxiliary_heads  # Is usually None
            print(f'XXX validation_step:')
            print(f'XXX {len(x)=}')
            print(f'XXX {x[0].size()}')
            print(f'XXX {len(y)=}')
            print(f"XXX {len(y[0]['labels'])=}")
            print(f"XXX {y[0]['labels'][0].size()=}")
            print(f"XXX {y[0]['labels'][0]=}")
            print(f"XXX {y[0]['boxes'][0].size()=}")
            print(f"XXX")
            print(f'XXX {type(loss_dict)=}')
            print(f'XXX {len(loss_dict)=}')
            print(f'XXX {type(loss_dict[0])=}')
            print(f'XXX {loss_dict[0].keys()=}')
            print(f'XXX {loss_dict[0]=}')
            print(f'XXX {model_output.auxiliary_heads=}')
            train_loss: Tensor = sum(loss_dict.values())  # Training loss (sum of loss values)  # torchgeo way
            self.log_dict(loss_dict)  # Record loss values
            return train_loss  # Return training loss
        model_output: ModelOutput = self(x, y)
        loss = self.val_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        # model_output.auxiliary_heads  # Is usually None
        y_hat_hard = to_class_prediction(model_output)
        self.val_metrics.update(y_hat_hard, y)

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
        batch_size = len(x)  # Set batch size (number of images)
        y = [
            {"boxes": batch["boxes"][i], "labels": batch["labels"][i]}
            for i in range(batch_size)
        ] # Extract bounding box and label information for each image
        model_output: ModelOutput = self(x, y)
        loss = self.test_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.test_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        y_hat_hard = to_class_prediction(model_output)
        self.test_metrics.update(y_hat_hard, y)

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
        batch_size = len(x)  # Set batch size (number of images)
        y = [
            {"boxes": batch["boxes"][i], "labels": batch["labels"][i]}
            for i in range(batch_size)
        ] # Extract bounding box and label information for each image
        model_output: ModelOutput = self(x, y)

        y_hat = self(x).output
        y_hat = y_hat.argmax(dim=1)
        return y_hat

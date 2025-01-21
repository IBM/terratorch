from typing import Any
import lightning
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback
from segmentation_models_pytorch.losses import FocalLoss, JaccardLoss
from torch import Tensor, nn
from torchgeo.datasets.utils import unbind_samples
from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from terratorch.models.model import AuxiliaryHead, Model, ModelOutput
from terratorch.registry.registry import MODEL_FACTORY_REGISTRY
from terratorch.tasks.loss_handler import LossHandler
from terratorch.tasks.base_task import TerraTorchTask

class ObjectDetectionTask(TerraTorchTask):
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
            model_factory (str): ModelFactory class to be used to instantiate the model.
            model_args (Dict): Arguments passed to the model factory.
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
        self.model_factory = MODEL_FACTORY_REGISTRY.build(model_factory)
        super().__init__(task="object_detection")
        self.train_loss_handler = LossHandler(self.train_metrics.prefix)
        self.val_loss_handler = LossHandler(self.val_metrics.prefix)
        self.test_loss_handler = LossHandler(self.test_metrics.prefix)

    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        * :class:`~torchmetrics.detection.mean_ap.MeanAveragePrecision`: Mean average
          precision (mAP) and mean average recall (mAR). Precision is the number of
          true positives divided by the number of true positives + false positives.
          Recall is the number of true positives divived by the number of true positives
          + false negatives. Uses 'macro' averaging. Higher values are better.

        .. note::
           * 'Micro' averaging suits overall performance evaluation but may not
             reflect minority class accuracy.
           * 'Macro' averaging gives equal weight to each class, and is useful for
             balanced performance assessment across imbalanced classes.
        """
        metrics = MetricCollection([MeanAveragePrecision(average='macro')])
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the train loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        batch_size = len(x)  # Use list length instead of x.shape[0]
        y = [
            {'boxes': batch['boxes'][i], 'labels': batch['labels'][i]}
            for i in range(batch_size)
        ] # Extract bounding box and label information for each image
        loss = self(x, y).output
        # loss = { # From Faster-RCNN
        #     loss_classifier
        #     loss_box_reg
        #     loss_objectness
        #     loss_rpn_box_reg
        # }
        loss["loss"] = sum(loss.values())  # log_loss() below requires 'loss' item
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=batch_size)

        #metrics handling ??

        return loss["loss"]

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metrics.compute()
        # metrics = {
        #      train/map
        #      train/map_50
        #      train/map_75
        #      train/map_small
        #      train/map_medium
        #      train/map_large
        #      train/mar_1
        #      train/mar_10
        #      train/mar_100
        #      train/mar_small (missing from summary)
        #      train/mar_medium (missing from summary)
        #      train/mar_large (missing from summary)
        #      train/map_per_class (missing from summary)
        #      train/mar_100_per_class
        #      train/classes
        # }
        metrics.pop('train/classes', None)
        self.log_dict(metrics, sync_dist=True)
        self.train_metrics.reset()

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        batch_size = len(x)
        y = [
            {'boxes': batch['boxes'][i], 'labels': batch['labels'][i]}
            for i in range(batch_size)
        ]
        y_hat = self(x).output
        # y_hat = list of { # From Faster-RCNN
        #     boxes: tensor(N, 4)
        #     labels: tensor(N)
        #     scores: tensor(N)
        # }

        metrics = self.val_metrics(y_hat, y)
        # metrics = {
        #     val/map
        #     val/map_50
        #     val/map_75
        #     val/map_small
        #     val/map_medium
        #     val/map_large
        #     val/mar_1
        #     val/mar_10
        #     val/mar_100
        #     val/mar_small
        #     val/mar_medium
        #     val/mar_large
        #     val/map_per_class
        #     val/mar_100_per_class
        #     val/classes
        # }
        metrics.pop('val/classes', None)

        # Currently we don't know how to compute loss value from y_hat
        #loss = self.train_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        val_loss: Tensor = sum(metrics.values())  # FIXME
        loss = {'loss': val_loss}
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=batch_size)

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

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        # metrics = {
        #      val/map
        #      val/map_50
        #      val/map_75
        #      val/map_small
        #      val/map_medium
        #      val/map_large
        #      val/mar_1
        #      val/mar_10
        #      val/mar_100
        #      val/mar_small
        #      val/mar_medium
        #      val/mar_large
        #      val/map_per_class
        #      val/mar_100_per_class
        #      val/classes
        # }
        metrics.pop('val/classes', None)
        self.log_dict(metrics, sync_dist=True)
        self.val_metrics.reset()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        batch_size = len(x)
        y = [
            {'boxes': batch['boxes'][i], 'labels': batch['labels'][i]}
            for i in range(batch_size)
        ]
        y_hat = self(x, y).output
        metrics = self.test_metrics(y_hat, y)
        metrics.pop('test/classes', None)

        # Currently we don't know how to compute loss value from y_hat
        #loss = self.train_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        test_loss: Tensor = sum(metrics.values())  # FIXME
        loss = {'loss': test_loss}
        self.test_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=batch_size)

        self.log_dict(metrics, batch_size=batch_size)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> list[dict[str, Tensor]]:
        """Compute the predicted bounding boxes.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        batch_size = len(x)
        y_hat = self(x).output
        return y_hat

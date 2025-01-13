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
        # Workaround for issue https://github.com/kornia/kornia/issues/3066
        # until kornia 0.7.4 or 8.5 is released.
        x = x.to(self.device)
        batch_size = len(x)  # Use list length instead of x.shape[0]
        y = [
            {'boxes': batch['boxes'][i], 'labels': batch['labels'][i]}
            for i in range(batch_size)
        ] # Extract bounding box and label information for each image
        loss_dict = self(x, y).output
        # loss_dict = { # From Faster-RCNN
        #     loss_classifier: tensor(torch.Size([])) (torch.float32 on cuda:0)
        #     loss_box_reg: tensor(torch.Size([])) (torch.float32 on cuda:0)
        #     loss_objectness: tensor(torch.Size([])) (torch.float32 on cuda:0)
        #     loss_rpn_box_reg: tensor(torch.Size([])) (torch.float32 on cuda:0)
        # }
        train_loss: Tensor = sum(loss_dict.values())
        loss_dict['loss'] = train_loss  # self.train_loss_handler.log_loss() below requires item 'loss'

        # Choose one from followings
        #self.log_dict(loss_dict, batch_size=batch_size)
        self.train_loss_handler.log_loss(self.log, loss_dict=loss_dict, batch_size=batch_size)

        return train_loss

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metrics.compute()
        # metrics = { # From Faster-RCNN
        #     train/map: tensor(torch.Size([])) (torch.float32 on cpu)
        #     train/map_50: tensor(torch.Size([])) (torch.float32 on cpu)
        #     train/map_75: tensor(torch.Size([])) (torch.float32 on cpu)
        #     train/map_small: tensor(torch.Size([])) (torch.float32 on cpu)
        #     train/map_medium: tensor(torch.Size([])) (torch.float32 on cpu)
        #     train/map_large: tensor(torch.Size([])) (torch.float32 on cpu)
        #     train/mar_1: tensor(torch.Size([])) (torch.float32 on cpu)
        #     train/mar_10: tensor(torch.Size([])) (torch.float32 on cpu)
        #     train/mar_100: tensor(torch.Size([])) (torch.float32 on cpu)
        #     train/mar_small: tensor(torch.Size([])) (torch.float32 on cpu)
        #     train/mar_medium: tensor(torch.Size([])) (torch.float32 on cpu)
        #     train/mar_large: tensor(torch.Size([])) (torch.float32 on cpu)
        #     train/map_per_class: tensor(torch.Size([])) (torch.float32 on cpu)
        #     train/mar_100_per_class: tensor(torch.Size([])) (torch.float32 on cpu)
        #     train/classes: tensor(torch.Size([0])) (torch.int32 on cpu)
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
        # Workaround for issue https://github.com/kornia/kornia/issues/3066
        # until kornia 0.7.4 or 8.5 is released.
        x = x.to(self.device)
        batch_size = len(x)
        y = [
            {'boxes': batch['boxes'][i], 'labels': batch['labels'][i]}
            for i in range(batch_size)
        ]
        y_hat = self(x).output
        metrics = self.val_metrics(y_hat, y)
        metrics.pop('val/classes', None)
        val_loss: Tensor = sum(metrics.values())  # FIXME
        loss_dict = {'loss': val_loss}
        self.log_dict(metrics, batch_size=batch_size)
        self.val_loss_handler.log_loss(self.log, loss_dict=loss_dict, batch_size=batch_size)  # XXX FIXME

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
        # metrics = { # From Faster-RCNN
        #     val/map: tensor(torch.Size([])) (torch.float32 on cpu)
        #     val/map_50: tensor(torch.Size([])) (torch.float32 on cpu)
        #     val/map_75: tensor(torch.Size([])) (torch.float32 on cpu)
        #     val/map_small: tensor(torch.Size([])) (torch.float32 on cpu)
        #     val/map_medium: tensor(torch.Size([])) (torch.float32 on cpu)
        #     val/map_large: tensor(torch.Size([])) (torch.float32 on cpu)
        #     val/mar_1: tensor(torch.Size([])) (torch.float32 on cpu)
        #     val/mar_10: tensor(torch.Size([])) (torch.float32 on cpu)
        #     val/mar_100: tensor(torch.Size([])) (torch.float32 on cpu)
        #     val/mar_small: tensor(torch.Size([])) (torch.float32 on cpu)
        #     val/mar_medium: tensor(torch.Size([])) (torch.float32 on cpu)
        #     val/mar_large: tensor(torch.Size([])) (torch.float32 on cpu)
        #     val/map_per_class: tensor(torch.Size([])) (torch.float32 on cpu)
        #     val/mar_100_per_class: tensor(torch.Size([])) (torch.float32 on cpu)
        #     val/classes: tensor(torch.Size([10])) (torch.int32 on cpu)
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
        # Workaround for issue https://github.com/kornia/kornia/issues/3066
        # until kornia 0.7.4 or 8.5 is released.
        x = x.to(self.device)
        batch_size = len(x)
        y = [
            {'boxes': batch['boxes'][i], 'labels': batch['labels'][i]}
            for i in range(batch_size)
        ]
        y_hat = self(x, y).output
        metrics = self.test_metrics(y_hat, y)

        metrics.pop('test/classes', None)

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
        # Workaround for issue https://github.com/kornia/kornia/issues/3066
        # until kornia 0.7.4 or 8.5 is released.
        x = x.to(self.device)
        batch_size = len(x)
        y_hat = self(x, y).output
        return y_hat

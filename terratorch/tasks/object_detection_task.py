# modified from https://torchgeo.readthedocs.io/en/latest/_modules/torchgeo/trainers/detection.html#ObjectDetectionTask
# and from https://torchgeo.readthedocs.io/en/latest/_modules/torchgeo/trainers/instance_segmentation.html#InstanceSegmentationTask

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for object detection."""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torchgeo.datasets import RGBBandsMissingError, unbind_samples
from torchgeo.trainers import BaseTask

from terratorch.registry import MODEL_FACTORY_REGISTRY
from terratorch.tasks.loss_handler import LossHandler
from terratorch.tasks.optimizer_factory import optimizer_factory
import pdb
import torch

from torchvision.ops import nms


class ObjectDetectionTask(BaseTask):

    ignore = None
    monitor = 'val_map'
    mode = 'max'

    def __init__(
        self,
        model_factory: str,
        model_args: dict,

        lr: float = 0.001,

        optimizer: str | None = None,
        optimizer_hparams: dict | None = None,
        scheduler: str | None = None,
        scheduler_hparams: dict | None = None,

        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        class_names: list[str] | None = None,

        iou_threshold: float = 0.5,
        score_threshold: float = 0.5

    ) -> None:
        """Initialize a new ObjectDetectionTask instance.

        Args:
            model: Name of the `torchvision
                <https://pytorch.org/vision/stable/models.html#object-detection>`__
                model to use. One of 'faster-rcnn', 'fcos', or 'retinanet'.
            backbone: Name of the `torchvision
                <https://pytorch.org/vision/stable/models.html#classification>`__
                backbone to use. One of 'resnet18', 'resnet34', 'resnet50',
                'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                'wide_resnet50_2', or 'wide_resnet101_2'.
            weights: Initial model weights. True for ImageNet weights, False or None
                for random weights.
            in_channels: Number of input channels to model.
            num_classes: Number of prediction classes (including the background).
            trainable_layers: Number of trainable layers.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to fine-tune the detection
                head.

        .. versionchanged:: 0.4
           *detection_model* was renamed to *model*.

        .. versionadded:: 0.5
           The *freeze_backbone* parameter.

        .. versionchanged:: 0.5
           *pretrained*, *learning_rate*, and *learning_rate_schedule_patience* were
           renamed to *weights*, *lr*, and *patience*.
        """
        
        self.model_factory = MODEL_FACTORY_REGISTRY.build(model_factory)
        self.framework = model_args['framework']
        self.monitor = 'val_segm_map' if self.framework == 'mask-rcnn' else self.monitor
    
        super().__init__()
        self.train_loss_handler = LossHandler(self.train_metrics.prefix)
        self.test_loss_handler = LossHandler(self.test_metrics.prefix)
        self.val_loss_handler = LossHandler(self.val_metrics.prefix)
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.lr = lr
        if optimizer_hparams is not None:
            if "lr" in self.hparams["optimizer_hparams"].keys():
                self.lr = float(self.hparams["optimizer_hparams"]["lr"])
                del self.hparams["optimizer_hparams"]["lr"]

    def configure_models(self) -> None:
        self.model: Model = self.model_factory.build_model(
            "object_detection", **self.hparams["model_args"]
        )
        if self.hparams["freeze_backbone"]:
            self.model.freeze_encoder()
        if self.hparams["freeze_decoder"]:
            self.model.freeze_decoder()

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
        if self.framework == 'mask-rcnn':
            metrics = MetricCollection([MeanAveragePrecision(iou_type=('bbox', 'segm'))])
        else:
            metrics = MetricCollection([MeanAveragePrecision(average='macro')])

        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        optimizer = self.hparams["optimizer"]
        if optimizer is None:
            optimizer = "Adam"
        return optimizer_factory(
            optimizer,
            self.lr,
            self.parameters(),
            self.hparams["optimizer_hparams"],
            self.hparams["scheduler"],
            self.monitor,
            self.hparams["scheduler_hparams"],
        )

    def reformat_batch(self, batch: Any, batch_size: int):
        """Reformat batch to calculate loss and metrics.

        Args:
            batch: The output of your DataLoader.
            batch_size: Size of your batch
        Returns:
            Reformated batch
        """

        if 'masks' in batch.keys():
            y = [
                {'boxes': batch['boxes'][i], 'labels': batch['labels'][i], 'masks': torch.cat([x[None] for x in batch['masks'][i]])}
                for i in range(batch_size)
            ]
        else:

            y = [
                {'boxes': batch['boxes'][i], 'labels': batch['labels'][i]}
                for i in range(batch_size)
            ]

        return y

    def apply_nms_sample(self, y_hat, iou_threshold=0.5, score_threshold=0.5):

        boxes, scores, labels = y_hat['boxes'], y_hat['scores'], y_hat['labels']
        masks = y_hat['masks'] if "masks" in y_hat.keys() else None

        # Filter based on score threshold
        keep_score = scores > score_threshold
        boxes, scores, labels = boxes[keep_score], scores[keep_score], labels[keep_score]
        if masks is not None:
            masks = masks[keep_score]

        # Apply NMS
        keep_nms = nms(boxes, scores, iou_threshold)

        y_hat['boxes'], y_hat['scores'], y_hat['labels'] = boxes[keep_nms], scores[keep_nms], labels[keep_nms]

        if masks is not None:
            y_hat['masks'] = masks[keep_nms]

        return y_hat

    def apply_nms_batch(self, y_hat: Any, batch_size: int):

        for i in range(batch_size):
            y_hat[i] = self.apply_nms_sample(y_hat[i], iou_threshold=self.iou_threshold, score_threshold=self.score_threshold)

        return y_hat

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        #print("training")
        
        x = batch['image']
        batch_size = x.shape[0]
        y = self.reformat_batch(batch, batch_size)
        loss_dict = self(x, y)
        if isinstance(loss_dict, dict) is False:
            loss_dict = loss_dict.output
        train_loss: Tensor = sum(loss_dict.values())
        self.log_dict(loss_dict, batch_size=batch_size)
        self.log("train_loss", train_loss)
        return train_loss

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
        batch_size = x.shape[0]
        y = self.reformat_batch(batch, batch_size)
        y_hat = self(x)
        if isinstance(y_hat, dict) is False:
            y_hat = y_hat.output

        y_hat = self.apply_nms_batch(y_hat, batch_size)
        
        if self.framework == 'mask-rcnn':

            for i in range(len(y_hat)):
                if y_hat[i]['masks'].shape[0] > 0:
                    y_hat[i]['masks']= y_hat[i]['masks'].squeeze(1).to(torch.uint8)
                    
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

            dataset = self.trainer.datamodule.val_dataset
            batch['prediction_boxes'] = [b['boxes'].cpu() for b in y_hat]
            batch['prediction_labels'] = [b['labels'].cpu() for b in y_hat]
            batch['prediction_scores'] = [b['scores'].cpu() for b in y_hat]
            
            if "masks" in y_hat[0].keys(): 
                batch['prediction_masks'] = [b['masks'].cpu() for b in y_hat]
            batch['image'] = batch['image'].cpu()
            sample = unbind_samples(batch)[0]

            fig: Figure | None = None
            try:
                fig = dataset.plot(sample)
            except RGBBandsMissingError:
                pass

            if fig:
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f'image/{batch_idx}', fig, global_step=self.global_step
                )
                plt.close()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        
        x = batch['image']
        batch_size = x.shape[0]
        y = self.reformat_batch(batch, batch_size)
        y_hat = self(x)
        if isinstance(y_hat, dict) is False:
            y_hat = y_hat.output

        y_hat = self.apply_nms_batch(y_hat, batch_size)
        
        if self.framework == 'mask-rcnn':

            for i in range(len(y_hat)):
                if y_hat[i]['masks'].shape[0] > 0:
                    y_hat[i]['masks']= y_hat[i]['masks'].squeeze(1).to(torch.uint8)


        metrics = self.test_metrics(y_hat, y)

        # https://github.com/Lightning-AI/torchmetrics/pull/1832#issuecomment-1623890714
        metrics.pop('test_classes', None)

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
        x = batch['image']
        batch_size = x.shape[0]
        y_hat: list[dict[str, Tensor]] = self(x)
        if isinstance(y_hat, dict) is False:
            y_hat = y_hat.output

        y_hat = self.apply_nms_batch(y_hat, batch_size)

        return y_hat
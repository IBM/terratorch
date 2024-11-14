from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelFBetaScore,
)

from terratorch.models.model import ModelOutput
from terratorch.tasks import ClassificationTask


# from geobench
def _balanced_binary_cross_entropy_with_logits(outputs: Tensor, targets: Tensor) -> Tensor:
    """Compute balance binary cross entropy for multi-label classification.

    Args:
        outputs: model outputs
        targets: targets to compute binary cross entropy on
    """
    classes = targets.shape[-1]
    outputs = outputs.view(-1, classes)
    targets = targets.view(-1, classes).float()
    loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
    loss = loss[targets == 0].mean() + loss[targets == 1].mean()
    return loss


class MultiLabelClassificationTask(ClassificationTask):
    def configure_losses(self) -> None:
        if self.hparams["loss"] == "bce":
            self.criterion: nn.Module = nn.BCEWithLogitsLoss()
        elif self.hparams["loss"] == "balanced_bce":
            self.criterion = _balanced_binary_cross_entropy_with_logits
        else:
            super().configure_losses()

    def configure_metrics(self) -> None:
        metrics = MetricCollection(
            {
                "Overall_Accuracy": MultilabelAccuracy(
                    num_labels=self.hparams["model_args"]["num_classes"], average="micro"
                ),
                "Average_Accuracy": MultilabelAccuracy(
                    num_labels=self.hparams["model_args"]["num_classes"], average="macro"
                ),
                "Multilabel_F1_Score": MultilabelFBetaScore(
                    num_labels=self.hparams["model_args"]["num_classes"], beta=1.0, average="micro"
                ),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    @staticmethod
    def to_multilabel_prediction(y: ModelOutput) -> Tensor:
        y_hat = y.output
        return torch.sigmoid(y_hat)

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        x = batch["image"]
        y = batch["label"]
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k:batch[k] for k in other_keys}

        model_output: ModelOutput = self(x, **rest)
        loss = self.train_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        y_hat = self.to_multilabel_prediction(model_output)
        self.train_metrics.update(y_hat, y)

        return loss["loss"]

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        x = batch["image"]
        y = batch["label"]
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k:batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)
        loss = self.val_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        y_hat = self.to_multilabel_prediction(model_output)
        self.val_metrics.update(y_hat, y)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        x = batch["image"]
        y = batch["label"]
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k:batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)
        loss = self.test_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.test_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        y_hat = self.to_multilabel_prediction(model_output)
        self.test_metrics.update(y_hat, y)

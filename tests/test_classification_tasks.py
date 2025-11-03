import unittest
from unittest.mock import patch
import torch
from torch import nn

from terratorch.tasks.classification_tasks import ClassificationTask
from terratorch.models.model import ModelOutput
from terratorch.tasks.utils import _instantiate_from_path

class DummyModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        return ModelOutput(output=torch.randn(batch_size, self.num_classes))

class DummyLoss(nn.Module):
    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = factor

    def forward(self, preds, targets):
        targets_onehot = nn.functional.one_hot(targets, num_classes=preds.size(1)).float()
        return ((preds - targets_onehot) ** 2).mean() * self.factor

class TestClassificationTaskCustomLoss(unittest.TestCase):

    def setUp(self):
        self.model_args = {"num_classes": 3}

    @patch("terratorch.tasks.classification_tasks._instantiate_from_path")
    def test_custom_loss_str_path(self, mock_instantiate):
        # _instantiate_from_path soll unsere DummyLoss zurückgeben
        mock_instantiate.return_value = DummyLoss(factor=2.0)

        task = ClassificationTask(
            model_args=self.model_args,
            model=DummyModel(num_classes=3),
            loss="dummy.DummyLoss",
            custom_loss=True,
            custom_loss_kwargs={"factor": 2.0}
        )
        task.configure_losses()
        self.assertIsInstance(task.criterion, DummyLoss)
        self.assertEqual(task.criterion.factor, 2.0)

    def test_custom_loss_nn_module(self):
        # Loss wird direkt als nn.Module übergeben
        custom_loss = DummyLoss(factor=3.0)
        task = ClassificationTask(
            model_args=self.model_args,
            model=DummyModel(num_classes=3),
            loss=custom_loss,
            custom_loss=True
        )
        task.configure_losses()
        self.assertIsInstance(task.criterion, DummyLoss)
        self.assertEqual(task.criterion.factor, 3.0)

    def test_training_step_with_custom_loss(self):
        custom_loss = DummyLoss(factor=1.5)
        task = ClassificationTask(
            model_args=self.model_args,
            model=DummyModel(num_classes=3),
            loss=custom_loss,
            custom_loss=True
        )
        task.configure_losses()
        # Dummy-Metrics konfigurieren
        task.configure_metrics()

        batch = {
            "image": torch.randn(2, 3, 64, 64),
            "label": torch.randint(0, 3, (2,))
        }
        loss = task.training_step(batch, batch_idx=0)
        self.assertIsInstance(loss, torch.Tensor)

class TestInstantiateFromPath(unittest.TestCase):

    def test_instantiate_class_directly(self):
        module_path = "tests.test_classification_tasks.DummyLoss" 
        obj = _instantiate_from_path(module_path, factor=2.0)
        self.assertIsInstance(obj, DummyLoss)
        self.assertEqual(obj.factor, 2.0)

    def test_instantiate_with_wrong_path(self):
        with self.assertRaises(ImportError):
            _instantiate_from_path("nonexistent.module.Class")

    def test_instantiate_with_wrong_class_name(self):
        import torch
        with self.assertRaises(AttributeError):
            _instantiate_from_path("torch.nn.NonExistentClass")

if __name__ == "__main__":
    unittest.main()

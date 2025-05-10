# Assisted by watsonx Code Assistant 
import unittest
from unittest.mock import MagicMock

from terratorch.tasks import ObjectDetectionTask
from terratorch.models import ObjectDetectionModelFactory
import torch

from torchmetrics import MetricCollection

class TestObjectDetectionTask(unittest.TestCase):

    def setUp(self):
        self.model_factory = "ObjectDetectionModelFactory"
        self.model_args1 = {
            'framework': 'mask-rcnn',
            'backbone': 'prithvi_eo_v2_300',
            'num_classes': 12,
            'backbone_pretrained': True,
            'backbone_bands': ['RED', 'GREEN', 'BLUE'],
            'necks': [
                {'name': 'SelectIndices', 'indices': [5, 11, 17, 23]},
                {'name': 'ReshapeTokensToImage'},
                {'name': 'LearnedInterpolateToPyramidal'},
                {'name': 'FeaturePyramidNetworkNeck'}
            ]
        }
        self.model_args2 = {
            'framework': 'mask-rcnn',
            'backbone': 'timm_resnet50',
            'backbone_pretrained': True,
            'num_classes': 12,
            'in_channels': 3,
            'necks': [{'name': 'FeaturePyramidNetworkNeck'}]
        }
        self.task1 = ObjectDetectionTask(
            model_factory=self.model_factory,
            model_args=self.model_args1,
            lr=0.001,
            optimizer="Adam",
            optimizer_hparams={},
            scheduler="None",
            scheduler_hparams={},
            freeze_backbone=False,
            freeze_decoder=False,
            class_names=None,
            iou_threshold=0.5,
            score_threshold=0.5
        )
        self.task2 = ObjectDetectionTask(
            model_factory=self.model_factory,
            model_args=self.model_args2,
            lr=0.001,
            optimizer="Adam",
            optimizer_hparams={},
            scheduler="None",
            scheduler_hparams={},
            freeze_backbone=False,
            freeze_decoder=False,
            class_names=None,
            iou_threshold=0.5,
            score_threshold=0.5
        )

#     def test_init(self):
#         # Test that the instance attributes are set correctly
#         self.assertIsInstance(self.task1.model_factory, ObjectDetectionModelFactory)
#         self.assertEqual(self.task1.lr, 0.001)
#         self.assertEqual(self.task1.optimizer, "Adam")
#         self.assertEqual(self.task1.scheduler, "None")
#         self.assertEqual(self.task1.iou_threshold, 0.5)
#         self.assertEqual(self.task1.score_threshold, 0.5)

#         self.assertIsInstance(self.task2.model_factory, ObjectDetectionModelFactory)
#         self.assertEqual(self.task2.lr, 0.001)
#         self.assertEqual(self.task2.optimizer, "Adam")
#         self.assertEqual(self.task2.scheduler, "None")
#         self.assertEqual(self.task2.iou_threshold, 0.5)
#         self.assertEqual(self.task2.score_threshold, 0.5)

    def test_configure_models(self):
        # Test that the model is configured correctly
        self.task1.configure_models()
        # ObjectDetectionModelFactory.build_model.assert_called_once_with("object_detection", **self.task1.hparams["model_args"])

        self.task2.configure_models()
        # ObjectDetectionModelFactory.build_model.assert_called_once_with("object_detection", **self.task2.hparams["model_args"])

    def test_configure_metrics(self):
        # Test that the metrics are configured correctly
        self.task1.configure_metrics()
        self.assertIsInstance(self.task1.train_metrics, MetricCollection)
        self.assertIsInstance(self.task1.val_metrics, MetricCollection)
        self.assertIsInstance(self.task1.test_metrics, MetricCollection)

    # def test_configure_optimizers(self):
    #     # Test that the optimizer and scheduler are configured correctly
    #     optimizer_factory_mock = MagicMock()
    #     optimizer_factory_mock.return_value = "optimizer"
    #     self.task1.configure_optimizers = MagicMock(return_value=optimizer_factory_mock)
    #     optimizer = self.task1.configure_optimizers()
    #     self.assertIsInstance(optimizer, dict)
    #     self.task1.configure_optimizers.assert_called_once_with()

    # def test_reformat_batch(self):
    #     # Test that the batch is reformatted correctly
    #     batch = {"image": torch.randn(2, 3, 224, 224)}
    #     batch_size = 2
    #     y = self.task1.reformat_batch(batch, batch_size)
    #     self.assertEqual(len(y), batch_size)
    #     self.assertIsInstance(y[0], dict)
    #     self.assertIn("boxes", y[0])
    #     self.assertIn("labels", y[0])
    #     self.assertIn("masks", y[0]) if "masks" in batch else True

    def test_apply_nms_sample(self):
        # Test that NMS is applied correctly to a single sample
        y_hat = {"boxes": torch.randn(10, 4), "scores": torch.randn(10), "labels": torch.randint(0, 2, (10,))}
        iou_threshold = 0.5
        score_threshold = 0.5
        result = self.task1.apply_nms_sample(y_hat, iou_threshold, score_threshold)
        self.assertEqual(len(result["boxes"]), len(result["scores"]))
        self.assertEqual(len(result["boxes"]), len(result["labels"]))

    def test_apply_nms_batch(self):
        # Test that NMS is applied correctly to a batch of samples
        y_hat = [{"boxes": torch.randn(10, 4), "scores": torch.randn(10), "labels": torch.randint(0, 2, (10,))}] * 2
        batch_size = 2
        result = self.task1.apply_nms_batch(y_hat, batch_size)
        for sample in result:
            self.assertEqual(len(sample["boxes"]), len(sample["scores"]))
            self.assertEqual(len(sample["boxes"]), len(sample["labels"]))

if __name__ == '__main__':

    unittest.main()
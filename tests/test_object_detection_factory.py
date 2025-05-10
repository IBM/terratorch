# Assisted by watsonx Code Assistant 
import unittest
from unittest.mock import MagicMock

from torch import nn
from terratorch.models.object_detection_model_factory import ObjectDetectionModel, BackboneWrapper, ModelOutputObjectDetection, ObjectDetectionModelFactory
from terratorch.models.necks import Neck, build_neck_list
from terratorch.models.utils import extract_prefix_keys
from terratorch.registry import BACKBONE_REGISTRY, DECODER_REGISTRY, MODEL_FACTORY_REGISTRY

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, feature_pyramid_network, misc
import torchvision

class TestObjectDetectionModelFactory(unittest.TestCase):

    def setUp(self):
        self.model_factory = ObjectDetectionModelFactory()
        self.backbone = 'prithvi_eo_v2_300'
        self.necks = [{'name': 'SelectIndices', 'indices': [5, 11, 17, 23]},
                      {'name': 'ReshapeTokensToImage'},
                      {'name': 'LearnedInterpolateToPyramidal'},
                      {'name': 'FeaturePyramidNetworkNeck'}]
        
        self.kwargs = {'backbone_pretrained': True,'backbone_bands': ['RED', 'GREEN', 'BLUE']}

    def test_build_model_faster_rcnn(self):
        # Test building a model with 'faster-rcnn' framework
        model = self.model_factory.build_model(
            "object_detection",
            self.backbone,
            "faster-rcnn",
            num_classes=10,
            necks=self.necks,
            **self.kwargs
        )
        self.assertIsInstance(model, ObjectDetectionModel)
        self.assertIsInstance(model.torchvision_model, torchvision.models.detection.FasterRCNN)

    def test_build_model_fcos(self):
        # Test building a model with 'fcos' framework
        model = self.model_factory.build_model(
            "object_detection",
            self.backbone,
            "fcos",
            num_classes=10,
            necks=self.necks,
            **self.kwargs
        )
        self.assertIsInstance(model, ObjectDetectionModel)
        self.assertIsInstance(model.torchvision_model, torchvision.models.detection.FCOS)

    def test_build_model_retinanet(self):
        # Test building a model with 'retinanet' framework
        model = self.model_factory.build_model(
            "object_detection",
            self.backbone,
            "retinanet",
            num_classes=10,
            necks=self.necks,
            **self.kwargs
        )
        self.assertIsInstance(model, ObjectDetectionModel)
        self.assertIsInstance(model.torchvision_model, torchvision.models.detection.RetinaNet)

    def test_build_model_mask_rcnn(self):
        # Test building a model with 'mask-rcnn' framework
        model = self.model_factory.build_model(
            "object_detection",
            self.backbone,
            "mask-rcnn",
            num_classes=10,
            necks=self.necks,
            **self.kwargs
        )
        self.assertIsInstance(model, ObjectDetectionModel)
        self.assertIsInstance(model.torchvision_model, torchvision.models.detection.MaskRCNN)

    def test_build_model_invalid_framework(self):
        # Test building a model with an invalid framework
        with self.assertRaises(ValueError):
            self.model_factory.build_model(
                "object_detection",
                self.backbone,
                "invalid_framework",
                num_classes=10,
                necks=self.necks,
                **self.kwargs
            )

# class TestBackboneWrapper(unittest.TestCase):

#     def setUp(self):
#         self.backbone = MagicMock(spec=nn.Module)
#         self.necks = [MagicMock(spec=Neck), MagicMock(spec=Neck)]
#         self.channel_list = [16, 32, 64, 128, 256]

#     def test_forward(self):
#         # Test the forward method of BackboneWrapper
#         wrapper = BackboneWrapper(self.backbone, self.necks, self.channel_list)
#         x = MagicMock()
#         result = wrapper(x)
#         self.backbone.assert_called_once_with(x)
#         self.necks[0].assert_called_once_with(self.backbone.output)
#         self.necks[1].assert_called_once_with(self.necks[0].output)
#         self.assertEqual(result, self.necks[1].output)

# class TestObjectDetectionModel(unittest.TestCase):

#     def setUp(self):
#         self.torchvision_model = MagicMock(spec=nn.Module)
#         self.model_name = "faster-rcnn"

#     def test_forward(self):
#         # Test the forward method of ObjectDetectionModel
#         model = ObjectDetectionModel(self.torchvision_model, self.model_name)
#         x = MagicMock()
#         result = model(x)
#         self.assertIsInstance(result, ModelOutputObjectDetection)
#         self.assertIs(result.output, self.torchvision_model(x))

#     def test_freeze_encoder(self):
#         # Test the freeze_encoder method of ObjectDetectionModel
#         model = ObjectDetectionModel(self.torchvision_model, self.model_name)
#         model.freeze_encoder()
#         self.backbone.assert_called_once_with()
#         for param in self.backbone.parameters():
#             self.assertFalse(param.requires_grad)

if __name__ == '__main__':
    unittest.main()

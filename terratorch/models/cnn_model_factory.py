# Copyright contributors to the Terratorch project

"""
This is just an example of a possible structure to include torchvision models
"""

import os
from functools import partial

import torch
from torchgeo.models import get_weight
from torchgeo.trainers import utils
from torchvision.models._api import WeightsEnum

from terratorch.models.model import Model, ModelFactory, ModelOutput
from terratorch.models.utils import extract_prefix_keys
from torchvision.models import resnet as R
import torchvision.models.detection
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, feature_pyramid_network, misc

from terratorch.registry import MODEL_FACTORY_REGISTRY

BACKBONE_LAT_DIM_MAP = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'resnext50_32x4d': 2048,
    'resnext101_32x8d': 2048,
    'wide_resnet50_2': 2048,
    'wide_resnet101_2': 2048,
}

BACKBONE_WEIGHT_MAP = {
    'resnet18': R.ResNet18_Weights.DEFAULT,
    'resnet34': R.ResNet34_Weights.DEFAULT,
    'resnet50': R.ResNet50_Weights.DEFAULT,
    'resnet101': R.ResNet101_Weights.DEFAULT,
    'resnet152': R.ResNet152_Weights.DEFAULT,
    'resnext50_32x4d': R.ResNeXt50_32X4D_Weights.DEFAULT,
    'resnext101_32x8d': R.ResNeXt101_32X8D_Weights.DEFAULT,
    'wide_resnet50_2': R.Wide_ResNet50_2_Weights.DEFAULT,
    'wide_resnet101_2': R.Wide_ResNet101_2_Weights.DEFAULT,
}


@MODEL_FACTORY_REGISTRY.register
class CNNModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        model: str = 'faster-rcnn',
        backbone: str = 'resnet50',
        num_classes: int = 1000,
        trainable_layers: int = 3,
        weights: str | bool = True,
        **kwargs,
    ) -> Model:
        """Build a classifier from torchvision

        Args:
            task (str): Must be "object_detection".
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
            num_classes: Number of prediction classes (including the background).
            trainable_layers: Number of trainable layers.

        Returns:
            Model: Torchvision model wrapped in ObjectDetectionModelWrapper.
        """
        if task != "object_detection":
            msg = f"torchvision models can only perform classification, but got task {task}"
            raise Exception(msg)
        backbone_kwargs, kwargs = extract_prefix_keys(kwargs, "backbone_")

        if backbone in BACKBONE_LAT_DIM_MAP:
            kwargs = { # for model backbone parameter
                'backbone_name': backbone,
                'trainable_layers': trainable_layers,
            }
            if weights:
                kwargs['weights'] = BACKBONE_WEIGHT_MAP[backbone]
            else:
                kwargs['weights'] = None

            latent_dim = BACKBONE_LAT_DIM_MAP[backbone]
        else:
            raise ValueError(f"Backbone type '{backbone}' is not valid.")

        if model == 'faster-rcnn':
            model_backbone = resnet_fpn_backbone(**kwargs)
            anchor_generator = AnchorGenerator(
                sizes=((32), (64), (128), (256), (512)), aspect_ratios=((0.5, 1.0, 2.0))
            )

            roi_pooler = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2
            )

            self.model = torchvision.models.detection.FasterRCNN(
                model_backbone,
                num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
            )
        elif model == 'fcos':
            kwargs['extra_blocks'] = feature_pyramid_network.LastLevelP6P7(256, 256)
            kwargs['norm_layer'] = (
                misc.FrozenBatchNorm2d if weights else torch.nn.BatchNorm2d
            )

            model_backbone = resnet_fpn_backbone(**kwargs)
            anchor_generator = AnchorGenerator(
                sizes=((8,), (16,), (32,), (64,), (128,), (256,)),
                aspect_ratios=((1.0,), (1.0,), (1.0,), (1.0,), (1.0,), (1.0,)),
            )

            self.model = torchvision.models.detection.FCOS(
                model_backbone, num_classes, anchor_generator=anchor_generator
            )
        elif model == 'retinanet':
            kwargs['extra_blocks'] = feature_pyramid_network.LastLevelP6P7(
                latent_dim, 256
            )
            model_backbone = resnet_fpn_backbone(**kwargs)

            anchor_sizes = (
                (16, 20, 25),
                (32, 40, 50),
                (64, 80, 101),
                (128, 161, 203),
                (256, 322, 406),
                (512, 645, 812),
            )
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

            head = RetinaNetHead(
                model_backbone.out_channels,
                anchor_generator.num_anchors_per_location()[0],
                num_classes,
                norm_layer=partial(torch.nn.GroupNorm, 32),
            )

            self.model = torchvision.models.detection.RetinaNet(
                model_backbone,
                num_classes,
                anchor_generator=anchor_generator,
                head=head,
            )
        else:
            raise ValueError(f"Model type '{model}' is not valid.")

        return ObjectDetectionModelWrapper(self.model, model)


class ObjectDetectionModelWrapper(Model):
    def __init__(self, torchvision_model, model_name) -> None:
        super().__init__()
        self.torchvision_model = torchvision_model
        self.model_name = model_name

    def forward(self, *args, **kwargs):
        return ModelOutput(self.torchvision_model(*args, **kwargs))

    def freeze_encoder(self):
        for param in self.torchvision_model.backbone.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        if self.model_name == 'faster-rcnn':
            for param in self.torchvision_model.rpn.parameters():
                param.requires_grad = False
                for param in self.torchvision_model.roi_heads.parameters():
                    param.requires_grad = False
        elif self.model_name == 'fcos':
            for param in self.torchvision_model.head.parameters():
                param.requires_grad = False
        elif self.model_name == 'retinanet':
            for param in self.torchvision_model.head.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Model type '{self.model_name}' is not valid.")

# Copyright contributors to the Terratorch project

from dataclasses import dataclass
from torch import nn
import pdb
from terratorch.models.model import (
    AuxiliaryHead,
    AuxiliaryHeadWithDecoderWithoutInstantiatedHead,
    Model,
    ModelFactory,
)
from terratorch.models.necks import Neck, build_neck_list
# from terratorch.models.pixel_wise_model import PixelWiseModel
from terratorch.models.model import ModelOutput
from terratorch.models.utils import extract_prefix_keys
from terratorch.registry import BACKBONE_REGISTRY, DECODER_REGISTRY, MODEL_FACTORY_REGISTRY

import torchvision.models.detection
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, feature_pyramid_network, misc

from terratorch.tasks.loss_handler import LossHandler
from terratorch.tasks.optimizer_factory import optimizer_factory

SUPPORTED_TASKS = ['object_detection']

def _get_backbone(backbone: str | nn.Module, **backbone_kwargs) -> nn.Module:
    if isinstance(backbone, nn.Module):
        return backbone
    return BACKBONE_REGISTRY.build(backbone, **backbone_kwargs)



def _check_all_args_used(kwargs):
    if kwargs:
        msg = f"arguments {kwargs} were passed but not used."
        raise ValueError(msg)


@MODEL_FACTORY_REGISTRY.register
class ObjectDetectionModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        framework: str,
        num_classes: int | None = None,
        necks: list[dict] | None = None,
        # aux_decoders: list[AuxiliaryHead] | None = None,
        # rescale: bool = True,  # noqa: FBT002, FBT001
        **kwargs,
    ) -> Model:
        """Generic model factory that combines an encoder and decoder, together with a head, for a specific task.

        Further arguments to be passed to the backbone, decoder or head. They should be prefixed with
        `backbone_`, `decoder_` and `head_` respectively.

        Args:
            task (str): Task to be performed. Currently supports "segmentation" and "regression".
            backbone (str, nn.Module): Backbone to be used. If a string, will look for such models in the different
                registries supported (internal terratorch registry, timm, ...). If a torch nn.Module, will use it
                directly. The backbone should have and `out_channels` attribute and its `forward` should return a list[Tensor].
            framework (str): object detection framework to be used between "faster-rcnn", "fcos", "retinanet".
            num_classes (int, optional): Number of classes. None for regression tasks.
            necks (list[dict]): nn.Modules to be called in succession on encoder features
                before passing them to the decoder. Should be registered in the NECKS_REGISTRY registry.
                Expects each one to have a key "name" and subsequent keys for arguments, if any.
                Defaults to None, which applies the identity function.
            aux_decoders (list[AuxiliaryHead] | None): List of AuxiliaryHead decoders to be added to the model.
                These decoders take the input from the encoder as well.
            rescale (bool): Whether to apply bilinear interpolation to rescale the model output if its size
                is different from the ground truth. Only applicable to pixel wise models
                (e.g. segmentation, pixel wise regression). Defaults to True.


        Returns:
            nn.Module: Full model with encoder, decoder and head.
        """
        task = task.lower()
        if task not in SUPPORTED_TASKS:
            msg = f"Task {task} not supported. Please choose one of {SUPPORTED_TASKS}"
            raise NotImplementedError(msg)

        backbone_kwargs, kwargs = extract_prefix_keys(kwargs, "backbone_")
        backbone = _get_backbone(backbone, **backbone_kwargs)

        try:
            out_channels = backbone.out_channels
        except AttributeError as e:
            msg = "backbone must have out_channels attribute"
            raise AttributeError(msg) from e
        pdb.set_trace()
        if necks is None:
            necks = []
        neck_list, channel_list = build_neck_list(necks, out_channels)
                                                  
        neck_module = nn.Sequential(*neck_list)

        combined_backbone = BackboneWrapper(backbone, neck_module)

        if framework == 'faster-rcnn':

            pdb.set_trace()
            anchor_generator = AnchorGenerator(
                sizes=((32), (64), (128), (256), (512)), aspect_ratios=((0.5, 1.0, 2.0))
            )

            roi_pooler = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2
            )

            model = torchvision.models.detection.FasterRCNN(
                combined_backbone,
                num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
            )
        elif framework == 'fcos':

            anchor_generator = AnchorGenerator(
                sizes=((8,), (16,), (32,), (64,), (128,), (256,)),
                aspect_ratios=((1.0,), (1.0,), (1.0,), (1.0,), (1.0,), (1.0,)),
            )

            model = torchvision.models.detection.FCOS(
                combined_backbone, num_classes, anchor_generator=anchor_generator
            )
        elif framework == 'retinanet':

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
                combined_backbone.out_channels,
                anchor_generator.num_anchors_per_location()[0],
                num_classes,
                norm_layer=partial(torch.nn.GroupNorm, 32),
            )

            model = torchvision.models.detection.RetinaNet(
                combined_backbone,
                num_classes,
                anchor_generator=anchor_generator,
                head=head,
            )
        else:
            raise ValueError(f"Model type '{model}' is not valid.")

        # some decoders already include a head
        # for these, we pass the num_classes to them
        # others dont include a head
        # for those, we dont pass num_classes

        return ObjectDetectionModel(model, framework)


class BackboneWrapper(nn.Module):
    def __init__(self, backbone, necks):
        super().__init__()
        self.backbone = backbone
        self.necks = necks
        self.out_channels = self.backbone.out_channels[-1] if len(self.necks) == 0 else self.necks[-1].channel_list[-1]

    def forward(self, x, **kwargs):

        x = self.backbone(x, **kwargs)
        x = necks(x)
        return x


class ObjectDetectionModel(Model):
    def __init__(self, torchvision_model, model_name) -> None:
        super().__init__()
        self.torchvision_model = torchvision_model
        self.model_name = model_name

    def forward(self, *args, **kwargs):
        return ModelOutputObjectDetection(self.torchvision_model(*args, **kwargs))

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

@dataclass
class ModelOutputObjectDetection(ModelOutput):
    output: dict

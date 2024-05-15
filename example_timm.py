""" The EfficientNet Family in PyTorch

An implementation of EfficienNet that covers variety of related models with efficient architectures:

* EfficientNet-V2
  - `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298

* EfficientNet (B0-B8, L2 + Tensorflow pretrained AutoAug/RandAug/AdvProp/NoisyStudent weight ports)
  - EfficientNet: Rethinking Model Scaling for CNNs - https://arxiv.org/abs/1905.11946
  - CondConv: Conditionally Parameterized Convolutions for Efficient Inference - https://arxiv.org/abs/1904.04971
  - Adversarial Examples Improve Image Recognition - https://arxiv.org/abs/1911.09665
  - Self-training with Noisy Student improves ImageNet classification - https://arxiv.org/abs/1911.04252

* MixNet (Small, Medium, and Large)
  - MixConv: Mixed Depthwise Convolutional Kernels - https://arxiv.org/abs/1907.09595

* MNasNet B1, A1 (SE), Small
  - MnasNet: Platform-Aware Neural Architecture Search for Mobile - https://arxiv.org/abs/1807.11626

* FBNet-C
  - FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable NAS - https://arxiv.org/abs/1812.03443

* Single-Path NAS Pixel1
  - Single-Path NAS: Designing Hardware-Efficient ConvNets - https://arxiv.org/abs/1904.02877

* TinyNet
    - Model Rubik's Cube: Twisting Resolution, Depth and Width for TinyNets - https://arxiv.org/abs/2010.14819
    - Definitions & weights borrowed from https://github.com/huawei-noah/CV-Backbones/tree/master/tinynet_pytorch

* And likely more...

The majority of the above models (EfficientNet*, MixNet, MnasNet) and original weights were made available
by Mingxing Tan, Quoc Le, and other members of their Google Brain team. Thanks for consistently releasing
the models and weights open source!

Hacked together by / Copyright 2019, Ross Wightman
"""
from functools import partial
from typing import List

import matplotlib.pyplot as plt
import terratorch
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from prettytable import PrettyTable
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import GroupNormAct, create_classifier, create_conv2d, get_norm_act_layer
from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model
from torch.utils.checkpoint import checkpoint

if __name__ == "__main__":
    print(timm.list_models("prithvi*"))
    print(timm.list_pretrained("prithvi*"))
    # dataset = GenericNonGeoSegmentationDataset("/Users/cpi/Downloads/Burn Scars/training/", image_grep="subsetted*_merged.tif", label_grep="subsetted*.mask.tif", class_names=["Normal", "Burn Scar"], num_classes=2)
    # dataset.plot(dataset[4], suptitle="Samples")
    # plt.show()

    model = timm.create_model(
        "prithvi_swin_B", num_frames=1, pretrained=True, features_only=True, pretrain_img_size=224, in_chans=6
    ).to("cuda")
    print(model)
    tensor = torch.ones((2, 6, 512, 512)).cuda()
    print([f.shape for f in model(tensor)])
    # def count_parameters(model):
    #     table = PrettyTable(["Modules", "Parameters"])
    #     total_params = 0
    #     for name, parameter in model.named_parameters():
    #         if not parameter.requires_grad:
    #             continue
    #         params = parameter.numel()
    #         table.add_row([name, params])
    #         total_params += params
    #     print(table)
    #     print(f"Total Trainable Params: {total_params}")
    #     return total_params

    # count_parameters(model)

    # out = model(torch.zeros(1, 6, 224, 224).to("cuda"))
    # for o in out:
    #     print(o)

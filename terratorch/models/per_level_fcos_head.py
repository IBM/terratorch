import math
from typing import Callable, Optional

import torch
from torch import Tensor, nn


class PerLevelFCOSClassificationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        num_levels: int,
        num_convs: int = 4,
        prior_probability: float = 0.01,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    *sum(
                        ([nn.Conv2d(in_channels, in_channels, 3, 1, 1), norm_layer(in_channels), nn.ReLU()]
                         for _ in range(num_convs)),
                        [],
                    )
                )
                for _ in range(num_levels)
            ]
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, 3, 1, 1)
        for stack in self.convs:
            for layer in stack:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

    def forward(self, features: list[Tensor]) -> Tensor:
        outputs = []
        for conv, feature in zip(self.convs, features):
            logits = self.cls_logits(conv(feature))
            n, _, h, w = logits.shape
            outputs.append(logits.view(n, -1, self.num_classes, h, w).permute(0, 3, 4, 1, 2).reshape(n, -1, self.num_classes))
        return torch.cat(outputs, dim=1)


class PerLevelFCOSRegressionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_levels: int,
        num_convs: int = 4,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    *sum(
                        ([nn.Conv2d(in_channels, in_channels, 3, 1, 1), norm_layer(in_channels), nn.ReLU()]
                         for _ in range(num_convs)),
                        [],
                    )
                )
                for _ in range(num_levels)
            ]
        )
        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, 3, 1, 1)
        self.bbox_ctrness = nn.Conv2d(in_channels, num_anchors, 3, 1, 1)
        for layer in (self.bbox_reg, self.bbox_ctrness):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
        for stack in self.convs:
            for layer in stack:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.zeros_(layer.bias)

    def forward(self, features: list[Tensor]) -> tuple[Tensor, Tensor]:
        regressions = []
        centernesses = []
        for conv, feature in zip(self.convs, features):
            head_feature = conv(feature)
            bbox = nn.functional.relu(self.bbox_reg(head_feature))
            ctr = self.bbox_ctrness(head_feature)
            n, _, h, w = bbox.shape
            regressions.append(bbox.view(n, -1, 4, h, w).permute(0, 3, 4, 1, 2).reshape(n, -1, 4))
            centernesses.append(ctr.view(n, -1, 1, h, w).permute(0, 3, 4, 1, 2).reshape(n, -1, 1))
        return torch.cat(regressions, dim=1), torch.cat(centernesses, dim=1)

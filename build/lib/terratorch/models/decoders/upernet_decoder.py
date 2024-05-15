import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

"""
Adapted from https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py
"""


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, inplace=False) -> None:  # noqa: FBT002
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# class PSPModule(nn.Module):
#     # In the original inmplementation they use precise RoI pooling
#     # Instead of using adaptative average pooling
#     def __init__(self, in_channels: int, bin_sizes: list[int] | None = None):
#         super().__init__()
#         if bin_sizes is None:
#             bin_sizes = [1, 2, 3, 6]
#         out_channels = in_channels // len(bin_sizes)
#         self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(
#                 in_channels + (out_channels * len(bin_sizes)),
#                 in_channels,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(0.1),
#         )

#     def _make_stages(self, in_channels, out_channels, bin_sz):
#         prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
#         conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         bn = nn.BatchNorm2d(out_channels)
#         relu = nn.ReLU(inplace=True)
#         return nn.Sequential(prior, conv, bn, relu)

#     def forward(self, features):
#         h, w = features.size()[2], features.size()[3]
#         pyramids = [features]
#         pyramids.extend(
#             [F.interpolate(stage(features), size=(h, w), mode="bilinear", align_corners=True) for stage in self.stages]
#         )
#         output = self.bottleneck(torch.cat(pyramids, dim=1))
#         return output


# def up_and_add(x, y):
#     return F.interpolate(x, size=(y.size(2), y.size(3)), mode="bilinear", align_corners=True) + y


# class FPNFuse(nn.Module):
#     def __init__(self, feature_channels=None, fpn_out=256):
#         super().__init__()
#         if feature_channels is None:
#             feature_channels = [256, 512, 1024, 2048]
#         if not feature_channels[0] == fpn_out:
#             msg = f"First index of feature channel ({feature_channels[0]}) did not match fpn_out ({fpn_out})"
#             raise Exception(msg)
#         self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1) for ft_size in feature_channels[1:]])
#         self.smooth_conv = nn.ModuleList(
#             [nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] * (len(feature_channels) - 1)
#         )
#         self.conv_fusion = nn.Sequential(
#             nn.Conv2d(
#                 len(feature_channels) * fpn_out,
#                 fpn_out,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(fpn_out),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, features):
#         features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1, strict=False)]
#         p = [up_and_add(features[i], features[i - 1]) for i in reversed(range(1, len(features)))]
#         p = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, p, strict=False)]
#         p = list(reversed(p))
#         p.append(features[-1])  # P = [P1, P2, P3, P4]
#         h, w = p[0].size(2), p[0].size(3)
#         p[1:] = [F.interpolate(feature, size=(h, w), mode="bilinear", align_corners=True) for feature in p[1:]]

#         x = self.conv_fusion(torch.cat(p, dim=1))
#         return x


# class UperNetDecoder(nn.Module):
#     def __init__(self, embed_dim: list[int]) -> None:
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.output_embed_dim = embed_dim[0]
#         self.PPN = PSPModule(embed_dim[-1])
#         self.FPN = FPNFuse(embed_dim, fpn_out=self.output_embed_dim)

#     def forward(self, x: Tensor):
#         x = [f.clone() for f in x]
#         x[-1] = self.PPN(x[-1])
#         x = self.FPN(x)

#         return x


# Adapted from MMSegmentation
class UperNetDecoder(nn.Module):
    """UperNetDecoder. Adapted from MMSegmentation."""

    def __init__(
        self,
        embed_dim: list[int],
        pool_scales: tuple[int] = (1, 2, 3, 6),
        channels: int = 256,
        align_corners: bool = True,  # noqa: FBT001, FBT002
    ):
        """Constructor

        Args:
            embed_dim (list[int]): Input embedding dimension for each input.
            pool_scales (tuple[int], optional): Pooling scales used in Pooling Pyramid
                Module applied on the last feature. Default: (1, 2, 3, 6).
            channels (int, optional): Channels used in the decoder. Defaults to 256.
            align_corners (bool, optional): Whter to align corners in rescaling. Defaults to True.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.output_embed_dim = channels
        self.channels = channels
        self.align_corners = align_corners
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.embed_dim[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = ConvModule(
            self.embed_dim[-1] + len(pool_scales) * self.channels, self.channels, 3, padding=1, inplace=True
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for embed_dim in self.embed_dim[:-1]:  # skip the top layer
            l_conv = ConvModule(
                embed_dim,
                self.channels,
                1,
                inplace=False,
            )
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                inplace=False,
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(len(self.embed_dim) * self.channels, self.channels, 3, padding=1, inplace=True)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # build laterals
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + torch.nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = torch.nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet."""

    def __init__(self, pool_scales, in_channels, channels, align_corners):
        """Constructor

        Args:
            pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
                Module.
            in_channels (int): Input channels.
            channels (int): Channels after modules, before conv_seg.
            align_corners (bool): align_corners argument of F.interpolate.
        """
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels

        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(self.in_channels, self.channels, 1, inplace=True),
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = torch.nn.functional.interpolate(
                ppm_out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

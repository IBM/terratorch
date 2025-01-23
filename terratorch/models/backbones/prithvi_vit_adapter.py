import logging
import math
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.nn.init import normal_, trunc_normal_

from terratorch.datasets.utils import HLSBands, generate_bands_intervals
from terratorch.models.backbones.detr_ops.modules.ms_deform_attn import MSDeformAttn
from terratorch.models.backbones.prithvi_mae import PrithviViT
from terratorch.models.backbones.prithvi_vit import (
    PRETRAINED_BANDS,
    checkpoint_filter_fn_vit,
    pretrained_weights,
    prithvi_cfgs,
)
from terratorch.models.backbones.vit_adapter_modules import (
    InteractionBlock,
    SpatialPriorModule,
    deform_inputs,
)

logger = logging.getLogger(__name__)


class PrithviViTAdapter(PrithviViT):  # type: ignore
    def __init__(
        self,
        # Adapter params
        interaction_indexes: list[list[int]],
        conv_inplane: int,
        add_vit_feature: bool,
        deform_num_heads: int,
        n_points: int,
        init_values: float,
        with_cffn: bool,
        cffn_ratio: float,
        deform_ratio: float,
        use_extra_extractor: bool,
        with_cp: bool,
        # PrithviViT params
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int, int] = (1, 16, 16),
        num_frames: int = 1,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        coords_encoding: list[str] | None = None,
        coords_scale_learn: bool = False,
        drop_path: float = 0.0,
        **kwargs: dict[str, Any],
    ) -> None:
        if num_frames != 1:
            msg = "PrithviViTAdapter only supports num_frames=1"
            raise ValueError(msg)
        if isinstance(patch_size, tuple) and patch_size[0] != 1:
            msg = "PrithviViTAdapter only supports patch_size[0]=1"
            raise ValueError(msg)
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            coords_encoding=coords_encoding,
            coords_scale_learn=coords_scale_learn,
            drop_path=drop_path,
            **kwargs,
        )

        self.cls_token = None  # type: ignore
        self.num_block = len(self.blocks)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature

        embed_dim = self.embed_dim
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(
            inplanes=conv_inplane,
            embed_dim=embed_dim,
            in_channels=self.in_chans,
            with_cp=False,
        )
        self.interactions = nn.Sequential(
            *[
                InteractionBlock(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=((i == len(interaction_indexes) - 1) and use_extra_extractor),
                    with_cp=with_cp,
                    drop=0.0,
                )
                for i in range(len(interaction_indexes))
            ]
        )
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm | nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d | nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m: nn.Module) -> None:
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(
        self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward_features(
        self,
        x: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
    ) -> list[torch.Tensor]:
        # add time dim for patch embedding.
        if len(x.shape) == 4 and self.patch_embed.input_size[0] == 1:
            # add time dim
            x = x.unsqueeze(2)
        if x.shape[2] != 1:
            msg = "Input tensor must have 1 frame"
            raise ValueError(msg)
        t, h, w = x.shape[-3:]

        deform_inputs1, deform_inputs2 = deform_inputs(x.squeeze(2))

        # SPM forward
        c1, c2, c3, c4 = self.spm(x.squeeze(2))
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        H = int(x.shape[-2] / self.patch_embed.patch_size[1])  # noqa: N806
        W = int(x.shape[-1] / self.patch_embed.patch_size[2])  # noqa: N806
        x = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self.interpolate_pos_encoding(x, t, h, w)

        # We don't have dropout in Prithvi
        # x = self.pos_drop(x + pos_embed)
        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        if self.temporal_encoding and temporal_coords is not None:
            num_tokens_per_frame = x.shape[1] // self.num_frames
            temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
            x = x + temporal_encoding
        if self.location_encoding and location_coords is not None:
            location_encoding = self.location_embed_enc(location_coords)
            x = x + location_encoding

        # Interaction
        outs = []
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(
                x,
                c,
                self.blocks[indexes[0] : indexes[-1] + 1],
                deform_inputs1,
                deform_inputs2,
                H,
                W,
            )
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode="bilinear", align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


def checkpoint_filter_fn_vit_adapter(
    state_dict: dict,
    model: PrithviViTAdapter,
    pretrained_bands: list[HLSBands | str | int],
    model_bands: list[HLSBands | int],
) -> dict:
    return checkpoint_filter_fn_vit(state_dict, model, pretrained_bands, model_bands)


def _create_prithvi_adapter(
    variant: str,
    pretrained: bool = False,  # noqa: FBT001, FBT002
    model_bands: list[HLSBands | int] | None = None,
    ckpt_path: str = None,
    pretrained_bands: list[HLSBands | str | int] | None = None,
    num_frames: int = 1,
    encoder_only: bool = True,
    **kwargs,
) -> PrithviViTAdapter:
    # Load default config
    model_args = prithvi_cfgs[variant].copy()

    pretrained_bands = pretrained_bands or model_args.get("bands", PRETRAINED_BANDS)

    if model_bands is None:
        model_bands: list[HLSBands | int] = pretrained_bands
        logger.info(
            f"Model bands not passed. Assuming bands are ordered in the same way as {pretrained_bands}."
            f"Pretrained patch_embed layer may be misaligned with current bands"
        )
    else:
        model_bands = [HLSBands.try_convert_to_hls_bands_enum(b) for b in model_bands]
        model_bands = generate_bands_intervals(model_bands)

    kwargs["in_chans"] = len(model_bands)
    kwargs["num_frames"] = num_frames
    model_args.update(kwargs)

    assert encoder_only, "PrithviViTAdapter only supports encoder_only=True"
    model = PrithviViTAdapter(**model_args)

    checkpoint_filter_wrapper_fn = checkpoint_filter_fn_vit_adapter

    if pretrained:
        if variant not in pretrained_weights:
            raise ValueError(
                f"No pre-trained model found for variant {variant} (pretrained models: {pretrained_weights.keys()})"
            )

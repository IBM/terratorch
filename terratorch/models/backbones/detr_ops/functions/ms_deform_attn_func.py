# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------


from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch.autograd import Function
from torch.autograd.function import once_differentiable

try:
    import MultiScaleDeformableAttention as MSDA  # noqa: N817
except ImportError as err:
    msg = (
        "Failed to import MultiScaleDeformableAttention. "
        "Please make sure that the MultiScaleDeformableAttention is installed. "
        "You can install it with:\n"
        'pip install "MultiScaleDeformableAttention @ git+https://github.com/fundamentalvision/Deformable-DETR.git#subdirectory=models/ops"\n'
        "This only works with CUDA."
    )
    raise ImportError(msg) from err


class MSDeformAttnFunction(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)  # type: ignore
    def forward(
        ctx: Any,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        value_level_start_index: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        im2col_step: int,
    ) -> torch.Tensor:
        ctx.im2col_step = im2col_step
        output: torch.Tensor = MSDA.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable  # type: ignore
    @torch.amp.custom_bwd(device_type="cuda")  # type: ignore
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, torch.Tensor, torch.Tensor, None]:
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = MSDA.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: list[list[int]],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape  # noqa: N806
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape  # noqa: N806
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)  # type: ignore
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):  # noqa: N806
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()

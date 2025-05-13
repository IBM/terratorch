"""This module contains logic for tiled inference.
    This does some additional things over the obvious fold -> predict -> unfold logic,
    e.g. cropping out areas around model prediction to reduce artifacts

    It additionally rebatches after the fold operation to gain speed up.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import tqdm


@dataclass
class TiledInferenceParameters:
    """Parameters to be used for inference.

    Args:
        h_crop (int): height of the crop region
        h_stride (int): size of the stride on the y-axis
        w_crop (int): width of the crop region
        w_stride (int): size of the stride on the x-axis
        delta (int): size of the border cropped from each tile. Defaults to None, which computes this automatically,
          with a minimum of 16.
        average_patches (bool): Whether to average the overlapping regions. Defaults to True.
        batch_size (int): Number of patches per forward pass. Defaults to 16.
    """

    h_crop: int
    h_stride: int
    w_crop: int
    w_stride: int
    delta: int = None
    average_patches: bool = True
    batch_size: int = 16
    verbose: bool = False


@dataclass
class InferenceInput:
    batch: int
    input_coords: tuple[slice, slice]
    input_data: torch.Tensor
    output_crop: None | tuple[slice, slice]


def tiled_inference(
    model_forward: Callable,
    input_batch: torch.Tensor,
    out_channels: int,
    inference_parameters: TiledInferenceParameters,
    **kwargs
) -> torch.Tensor:
    """
    Divide an image into (potentially) overlapping tiles and perform inference on them.
    Additionally, rebatch for increased GPU utilization.

    Args:
        model_forward (Callable): Callable that return the output of the model.
        input_batch (torch.Tensor): Input batch to be processed
        out_channels (int): Number of output channels
        inference_parameters (TiledInferenceParameters): Parameters to be used for the process.

    Returns:
        torch.Tensor: The result of the inference
    """

    shape = input_batch.shape
    device = input_batch.device
    # Move inputs to CPU to avoid out-of-memory errors
    input_batch = input_batch.cpu()

    batch_size = shape[0]
    # omit bands and take last two dimensions
    h_img, w_img = shape[-2], shape[-1]

    preds = input_batch.new_zeros((batch_size, out_channels, h_img, w_img))

    # this list will contain tuples. Inside the tuples:
    #   0. batch
    #   1. Coordinates where this should end up in the preds
    #   2. output/input
    #   3. Optionally, for inputs, how to crop the output
    # it is important this allocation follows the same order as in the original code

    # Stage 1: deal with border patches
    # Deal with patches near the right border

    coordinates_and_inputs: list[InferenceInput] = []
    for i in range(0, h_img - inference_parameters.h_crop + 1, inference_parameters.h_crop):
        patch = input_batch[..., i : i + inference_parameters.h_crop, w_img - inference_parameters.w_crop : w_img]
        coordinates_and_inputs += [
            InferenceInput(
                b,
                (slice(i, i + inference_parameters.h_crop), slice(w_img - inference_parameters.w_crop, w_img)),
                patch[b],
                None,
            )
            for b in range(batch_size)
        ]

    # Deal with patches near the bottom of the image
    for i in range(0, w_img - inference_parameters.w_crop + 1, inference_parameters.w_crop):
        patch = input_batch[..., h_img - inference_parameters.h_crop : h_img, i : i + inference_parameters.w_crop]
        coordinates_and_inputs += [
            InferenceInput(
                b,
                (slice(h_img - inference_parameters.h_crop, h_img), slice(i, i + inference_parameters.w_crop)),
                patch[b],
                None,
            )
            for b in range(batch_size)
        ]

    # Deal with last patches at the right bottom of the image
    patch = input_batch[..., h_img - inference_parameters.h_crop : h_img, w_img - inference_parameters.w_crop : w_img]
    coordinates_and_inputs += [
        InferenceInput(
            b,
            (slice(h_img - inference_parameters.h_crop, h_img), slice(w_img - inference_parameters.w_crop, w_img)),
            patch[b],
            None,
        )
        for b in range(batch_size)
    ]

    if inference_parameters.delta:
        delta_x = inference_parameters.delta
        delta_y = inference_parameters.delta
    else:
        delta_x = min(16, (inference_parameters.w_crop - inference_parameters.w_stride) // 2)
        delta_y = min(16, (inference_parameters.h_crop - inference_parameters.h_stride) // 2)

    # Stage 2: process internally with patch overlap {2*delta/inference_parameters.w_crop}
    nr = h_img
    nc = w_img
    for row in range(0, nr - inference_parameters.h_crop + 1, inference_parameters.h_stride):
        for col in range(0, nc - inference_parameters.w_crop + 1, inference_parameters.w_stride):
            patch = input_batch[..., row : row + inference_parameters.h_crop, col : col + inference_parameters.w_crop]
            if row == 0 or col == 0:
                coordinates_and_inputs += [
                    InferenceInput(
                        b,
                        (slice(row, row + inference_parameters.h_crop), slice(col, col + inference_parameters.w_crop)),
                        patch[b],
                        None,
                    )
                    for b in range(batch_size)
                ]
            else:
                coordinates_and_inputs += [
                    InferenceInput(
                        b,
                        (
                            slice(row + delta_y, row + inference_parameters.h_crop - delta_y),
                            slice(col + delta_x, col + inference_parameters.w_crop - delta_x),
                        ),
                        patch[b],
                        (
                            slice(delta_y, inference_parameters.h_crop - delta_y),
                            slice(delta_x, inference_parameters.w_crop - delta_x),
                        ),
                    )
                    for b in range(batch_size)
                ]

    # NOTE: the output may be SLIGHTLY different using batched inputs because of layers such as nn.LayerNorm
    # During inference, these layers compute batch statistics that affect the output.
    # However, this should still be correct.
    with torch.no_grad():
        preds_count = input_batch.new_zeros(batch_size, preds.shape[-2], preds.shape[-1])
        for start in tqdm.tqdm(range(0, len(coordinates_and_inputs), inference_parameters.batch_size),
                               desc="Tiled inference", disable=not inference_parameters.verbose):
            end = min(len(coordinates_and_inputs), start + inference_parameters.batch_size)
            batch = coordinates_and_inputs[start:end]
            tensor_input = torch.stack([b.input_data for b in batch], dim=0)
            tensor_input = tensor_input.to(device)
            output = model_forward(tensor_input, **kwargs).cpu()
            output = [output[i] for i in range(len(batch))]
            for batch_input, predicted in zip(batch, output, strict=True):
                if batch_input.output_crop is not None:
                    predicted = predicted[..., batch_input.output_crop[0], batch_input.output_crop[1]]
                if inference_parameters.average_patches:
                    preds[
                        batch_input.batch,
                        :,
                        batch_input.input_coords[0],
                        batch_input.input_coords[1],
                    ] += predicted
                else:
                    preds[
                        batch_input.batch,
                        :,
                        batch_input.input_coords[0],
                        batch_input.input_coords[1],
                    ] = predicted

                preds_count[
                    batch_input.batch,
                    batch_input.input_coords[0],
                    batch_input.input_coords[1],
                ] += 1

    if (preds_count == 0).sum() != 0:
        msg = "Some pixels did not receive a classification!"
        raise RuntimeError(msg)
    if inference_parameters.average_patches:
        return preds / preds_count.unsqueeze(1)
    return preds

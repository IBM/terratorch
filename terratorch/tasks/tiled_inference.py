"""This module contains logic for tiled inference.
    This does some additional things over the obvious fold -> predict -> unfold logic,
    e.g. cropping out areas around model prediction to reduce artifacts

    It additionally rebatches after the fold operation to gain speed up.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import time
import copy 

from torch.nn.parallel import DistributedDataParallel as DDP

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
    """

    h_crop: int
    h_stride: int
    w_crop: int
    w_stride: int
    delta: int = None
    average_patches: bool = True


@dataclass
class InferenceInput:
    batch: int
    input_coords: tuple[slice, slice]
    input_data: torch.Tensor
    output_crop: None | tuple[slice, slice]

class VectorInference:

    def __init__(self, process_batch_size=None, coordinates_and_inputs=None, inference_parameters=None, model_forward=None, preds=None, preds_count=None):

        self.process_batch_size = process_batch_size
        self.coordinates_and_inputs = coordinates_and_inputs
        self.inference_parameters = inference_parameters
        self.model_forward = DDP(model_forward)
        self.preds = preds
        self.preds_count = preds_count

        self.vectorized_loop = torch.vmap(self.model_eval)

    def model_eval(self, batch):
        batch = torch.unsqueeze(batch,0)
        return self.model_forward(batch)

    def model_inference(self):

        starts_list = list(range(0, len(self.coordinates_and_inputs), self.process_batch_size))

        tensor_inputs = []

        for start in starts_list:
            end = min(len(self.coordinates_and_inputs), start + self.process_batch_size)
            batch = self.coordinates_and_inputs[start:end]
            tensor_input = torch.stack([b.input_data for b in batch], dim=0)

            time_init = time.time()
            output = self.vectorized_loop(tensor_inputs)
            print(f"Time elapsed: {time.time() - time_init} s")

        #for start in starts_list:

            end = min(len(self.coordinates_and_inputs), start + self.process_batch_size)
            batch = self.coordinates_and_inputs[start:end]

            output = [output[i] for i in range(len(batch))]
            for batch_input, predicted in zip(batch, output, strict=True):
                if batch_input.output_crop is not None:
                    predicted = predicted[..., batch_input.output_crop[0], batch_input.output_crop[1]]
                if self.inference_parameters.average_patches:
                    self.preds[
                        batch_input.batch,
                        :,
                        batch_input.input_coords[0],
                        batch_input.input_coords[1],
                    ] += predicted
                else:
                    self.preds[
                        batch_input.batch,
                        :,
                        batch_input.input_coords[0],
                        batch_input.input_coords[1],
                    ] = predicted

                self.preds_count[
                    batch_input.batch,
                    batch_input.input_coords[0],
                    batch_input.input_coords[1],
                ] += 1

        return self.preds, self.preds_count

def tiled_inference(
    model_forward: Callable,
    input_batch: torch.Tensor,
    out_channels: int,
    inference_parameters: TiledInferenceParameters,
) -> torch.Tensor:
    """
    Like divide an image into (potentially) overlapping tiles and perform inference on them.
    Additionally rebatch for increased GPU utilization.

    Args:
        model_forward (Callable): Callable that return the output of the model.
        input_batch (torch.Tensor): Input batch to be processed
        out_channels (int): Number of output channels
        inference_parameters (TiledInferenceParameters): Parameters to be used for the process.

    Returns:
        torch.Tensor: The result of the inference
    """

    batch_size, h_img, w_img = input_batch.shape[0], *input_batch.shape[-2:]
    preds = input_batch.new_zeros((batch_size, out_channels, h_img, w_img))

    # this list will contain tuples. Inside the tuples:
    #   0. batch
    #   1. Coordinates where this should end up in the preds
    #   2. output/input
    #   3. Optionally, for inputs, how to crop the output
    # it is important this allocation follows the same order as in the original code

    # Stage 1: deal with border patches
    # Deal with patches near the right border
    time_init = time.time()
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

    print(f"Time elapsed: {time.time() - time_init} s")
    time_init = time.time()
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

    print(f"Time elapsed: {time.time() - time_init} s")
    time_init = time.time()
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
    print(f"Time elapsed: {time.time() - time_init} s")
    time_init = time.time()

    if inference_parameters.delta:
        delta_x = inference_parameters.delta
        delta_y = inference_parameters.delta
    else:
        delta_x = min(16, (inference_parameters.w_crop - inference_parameters.w_stride) // 2)
        delta_y = min(16, (inference_parameters.h_crop - inference_parameters.h_stride) // 2)

    time_init = time.time()
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

    print(f"Time elapsed: {time.time() - time_init} s")

    # NOTE: the output may be SLIGHTLY different using batched inputs because of layers such as nn.LayerNorm
    # During inference, these layers compute batch statistics that affect the output.
    # However, this should still be correct.
    # TODO: make this configurable by user?
    process_batch_size = 16
    with torch.no_grad():
        torch.set_num_interop_threads(24)

        preds_count = input_batch.new_zeros(batch_size, preds.shape[-2], preds.shape[-1])
        vectorized_instance = VectorInference(process_batch_size=process_batch_size,
                                              coordinates_and_inputs=coordinates_and_inputs,
                                              inference_parameters=inference_parameters,
                                              model_forward=model_forward, preds=preds, preds_count=preds_count)

        time_init = time.time()
        starts_list = list(range(0, len(coordinates_and_inputs), process_batch_size))

        vectorized_instance.model_inference()
        print(f"Model Time elapsed: {time.time() - time_init} s")
        print(preds)
    if (preds_count == 0).sum() != 0:
        msg = "Some pixels did not receive a classification!"
        raise RuntimeError(msg)
    if inference_parameters.average_patches:
        return preds / preds_count.unsqueeze(1)
    return preds

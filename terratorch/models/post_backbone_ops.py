from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from terratorch.registry import POST_BACKBONE_OPS_REGISTRY


class PostBackboneOp:
    """Base class for PostBackboneOps

    If the operation has an effect on the length of embeddings or their number, it must implement
    `self.process_channel_list` which returns the new channel list.
    """

    def __call__(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        msg = "PostBackboneOps must implement this"
        raise NotImplementedError(msg)

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return channel_list


@POST_BACKBONE_OPS_REGISTRY.register
class SelectIndices(PostBackboneOp):
    def __init__(self, indices: list[int]):
        self.indices = indices

    def __call__(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        return [features[i] for i in self.indices]

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return [channel_list[i] for i in self.indices]


@POST_BACKBONE_OPS_REGISTRY.register
class InterpolateToHierarchical(PostBackboneOp):
    def __init__(self, scale_factor: int = 2, mode: str = "nearest"):
        """Spatially interpolate embeddings so that embedding[i - 1] is scale_factor times larger than embedding[i]

        Useful to make non-hierarchical backbones compatible with hierarachical ones
        Args:
            scale_factor (int): Amount to scale embeddings by each layer. Defaults to 2.
            mode (str): Interpolation mode to be passed to torch.nn.functional.interpolate. Defaults to 'nearest'.
        """
        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        scale_exponents = list(range(len(features), 0, -1))
        for x, exponent in zip(features, scale_exponents, strict=True):
            out.append(F.interpolate(x, scale_factor = self.scale_factor ** exponent, mode=self.mode))

        return out

@POST_BACKBONE_OPS_REGISTRY.register
class MaxpoolToHierarchical(PostBackboneOp):
    def __init__(self, kernel_size: int = 2):
        """Spatially downsample embeddings so that embedding[i - 1] is scale_factor times smaller than embedding[i]

        Useful to make non-hierarchical backbones compatible with hierarachical ones
        Args:
            kernel_size (int). Base kernel size to use for maxpool. Defaults to 2.
        """
        self.kernel_size = kernel_size

    def __call__(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        scale_exponents = list(range(len(features)))
        for x, exponent in zip(features, scale_exponents, strict=True):
            if exponent == 0:
                out.append(x.clone())
            else:
                out.append(F.max_pool2d(x, kernel_size=self.kernel_size ** exponent))

        return out
@POST_BACKBONE_OPS_REGISTRY.register
class ReshapeTokensToImage(PostBackboneOp):
    def __init__(self, remove_cls_token=True, effective_time_dim: int = 1):  # noqa: FBT002
        """Reshape output of transformer encoder so it can be passed to a conv net.

        Args:
            remove_cls_token (bool, optional): Whether to remove the cls token from the first position.
                Defaults to True.
            effective_time_dim (int, optional): The effective temporal dimension the transformer processes.
                For a ViT, his will be given by `num_frames // tubelet size`. This is used to determine
                the temporal dimension of the embedding, which is concatenated with the embedding dimension.
                For example:
                - A model which processes 1 frame with a tubelet size of 1 has an effective_time_dim of 1.
                    The embedding produced by this model has embedding size embed_dim * 1.
                - A model which processes 3 frames with a tubelet size of 1 has an effective_time_dim of 3.
                    The embedding produced by this model has embedding size embed_dim * 3.
                - A model which processes 12 frames with a tubelet size of 4 has an effective_time_dim of 3.
                    The embedding produced by this model has an embedding size embed_dim * 3.
                Defaults to 1.
        """
        self.remove_cls_token = remove_cls_token
        self.effective_time_dim = effective_time_dim

    def __call__(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        for x in features:
            if self.remove_cls_token:
                x_no_token = x[:, 1:, :]
            else:
                x_no_token = x
            number_of_tokens = x_no_token.shape[1]
            tokens_per_timestep = number_of_tokens // self.effective_time_dim
            h = int(np.sqrt(tokens_per_timestep))
            encoded = rearrange(
                x_no_token,
                "batch (t h w) e -> batch (t e) h w",
                batch=x_no_token.shape[0],
                t=self.effective_time_dim,
                h=h,
            )
            out.append(encoded)
        return out

def apply_ops(ops: list[PostBackboneOp], embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
    cloned_embeddings = [e.clone() for e in embeddings]
    for op in ops:
        cloned_embeddings = op(cloned_embeddings)
    return cloned_embeddings

def apply_ops_to_channel_list(ops: list[PostBackboneOp], channel_list: list[int]) -> list[int]:
    channel_list_copy = channel_list.copy()
    for op in ops:
        channel_list_copy = op.process_channel_list(channel_list_copy)
    return channel_list_copy

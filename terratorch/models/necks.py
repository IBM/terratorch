from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import math

from terratorch.registry import NECK_REGISTRY, TERRATORCH_NECK_REGISTRY


class Neck(ABC, nn.Module):
    """Base class for Neck

    A neck must must implement `self.process_channel_list` which returns the new channel list.
    """

    def __init__(self, channel_list: list[int]) -> None:
        super().__init__()
        self.channel_list = channel_list

    @abstractmethod
    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return channel_list

    @abstractmethod
    def forward(self, channel_list: list[torch.Tensor]) -> list[torch.Tensor]: ...


@TERRATORCH_NECK_REGISTRY.register
class SelectIndices(Neck):
    def __init__(self, channel_list: list[int], indices: list[int]):
        """Select indices from the embedding list

        Args:
            indices (list[int]): list of indices to select.
        """
        super().__init__(channel_list)
        self.indices = indices

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        features = [features[i] for i in self.indices]
        return features

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        channel_list = [channel_list[i] for i in self.indices]
        return channel_list

@TERRATORCH_NECK_REGISTRY.register
class PermuteDims(Neck):
    def __init__(self, channel_list: list[int], new_order: list[int]):
        """Permute dimensions of each element in the embedding list

        Args:
            new_order (list[int]): list of indices to be passed to tensor.permute()
        """
        super().__init__(channel_list)
        self.new_order = new_order

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        features = [feat.permute(*self.new_order).contiguous() for feat in features]
        return features

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return super().process_channel_list(channel_list)

@TERRATORCH_NECK_REGISTRY.register
class InterpolateToPyramidal(Neck):
    def __init__(self, channel_list: list[int], scale_factor: int = 2, mode: str = "nearest"):
        """Spatially interpolate embeddings so that embedding[i - 1] is scale_factor times larger than embedding[i]

        Useful to make non-pyramidal backbones compatible with hierarachical ones
        Args:
            scale_factor (int): Amount to scale embeddings by each layer. Defaults to 2.
            mode (str): Interpolation mode to be passed to torch.nn.functional.interpolate. Defaults to 'nearest'.
        """
        super().__init__(channel_list)
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        scale_exponents = list(range(len(features), 0, -1))
        for x, exponent in zip(features, scale_exponents, strict=True):
            out.append(F.interpolate(x, scale_factor=self.scale_factor**exponent, mode=self.mode))

        return out

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return super().process_channel_list(channel_list)


@TERRATORCH_NECK_REGISTRY.register
class MaxpoolToPyramidal(Neck):
    def __init__(self, channel_list: list[int], kernel_size: int = 2):
        """Spatially downsample embeddings so that embedding[i - 1] is scale_factor times smaller than embedding[i]

        Useful to make non-pyramidal backbones compatible with hierarachical ones
        Args:
            kernel_size (int). Base kernel size to use for maxpool. Defaults to 2.
        """
        super().__init__(channel_list)
        self.kernel_size = kernel_size

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        scale_exponents = list(range(len(features)))
        for x, exponent in zip(features, scale_exponents, strict=True):
            if exponent == 0:
                out.append(x.clone())
            else:
                out.append(F.max_pool2d(x, kernel_size=self.kernel_size**exponent))

        return out

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return super().process_channel_list(channel_list)


@TERRATORCH_NECK_REGISTRY.register
class ReshapeTokensToImage(Neck):
    def __init__(self, channel_list: list[int], remove_cls_token=True, effective_time_dim: int = 1):  # noqa: FBT002
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
        super().__init__(channel_list)
        self.remove_cls_token = remove_cls_token
        self.effective_time_dim = effective_time_dim

    def collapse_dims(self, x):
        """
        When the encoder output has more than 3 dimensions, is necessary to 
        reshape it. 
        """
        shape = x.shape
        batch = x.shape[0]
        e = x.shape[-1]
        collapsed_dim = np.prod(x.shape[1:-1])

        return x.reshape(batch, collapsed_dim, e)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        for x in features:
            if self.remove_cls_token:
                x_no_token = x[:, 1:, :]
            else:
                x_no_token = x
            x_no_token = self.collapse_dims(x_no_token)
            number_of_tokens = x_no_token.shape[1]
            tokens_per_timestep = number_of_tokens // self.effective_time_dim
            h = int(math.sqrt(tokens_per_timestep))

            encoded = rearrange(
                x_no_token,
                "batch (t h w) e -> batch (t e) h w",
                batch=x_no_token.shape[0],
                t=self.effective_time_dim,
                h=h,
            )
            out.append(encoded)
        return out

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return super().process_channel_list(channel_list)

@TERRATORCH_NECK_REGISTRY.register
class AddBottleneckLayer(Neck):
    """Add a layer that reduces the channel dimension of the final embedding by half, and concatenates it

    Useful for compatibility with some smp decoders.
    """

    def __init__(self, channel_list: list[int]):
        super().__init__(channel_list)
        self.bottleneck = nn.Conv2d(channel_list[-1], channel_list[-1]//2, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        new_embedding = self.bottleneck(features[-1])
        features.append(new_embedding)
        return features

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return [*channel_list, channel_list[-1] // 2]

@TERRATORCH_NECK_REGISTRY.register
class LearnedInterpolateToPyramidal(Neck):
    """Use learned convolutions to transform the output of a non-pyramidal encoder into pyramidal ones

    Always requires exactly 4 embeddings
    """

    def __init__(self, channel_list: list[int]):
        super().__init__(channel_list)
        if len(channel_list) != 4:
            msg = "This class can only handle exactly 4 input embeddings"
            raise Exception(msg)
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(channel_list[0], channel_list[0] // 2, 2, 2),
            nn.BatchNorm2d(channel_list[0] // 2),
            nn.GELU(),
            nn.ConvTranspose2d(channel_list[0] // 2, channel_list[0] // 4, 2, 2),
        )
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(channel_list[1], channel_list[1] // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.embedding_dim = [channel_list[0] // 4, channel_list[1] // 2, channel_list[2], channel_list[3]]

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        scaled_inputs = []
        scaled_inputs.append(self.fpn1(features[0]))
        scaled_inputs.append(self.fpn2(features[1]))
        scaled_inputs.append(self.fpn3(features[2]))
        scaled_inputs.append(self.fpn4(features[3]))
        return scaled_inputs

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return [channel_list[0] // 4, channel_list[1] // 2, channel_list[2], channel_list[3]]


def build_neck_list(ops: list[dict], channel_list: list[int]) -> tuple[list[Neck], list[int]]:
    neck_list = []
    cur_channel_list = channel_list.copy()
    for cur_op in ops:
        op: Neck = NECK_REGISTRY.build(
            cur_op["name"], cur_channel_list, **{k: v for k, v in cur_op.items() if k != "name"}
        )
        cur_channel_list = op.process_channel_list(cur_channel_list)
        neck_list.append(op)

    return neck_list, cur_channel_list

from collections.abc import Callable

import numpy as np
import torch
from einops import rearrange


def apply_ops(ops: list[Callable], embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
    cloned_embeddings = [e.clone() for e in embeddings]
    for op in ops:
        cloned_embeddings = op(cloned_embeddings)
    return cloned_embeddings

class SelectIndices(Callable):
    def __init__(self, indices: list[int]):
        self.indices = indices

    def __call__(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        return [features[i] for i in self.indices]


class ReshapeTokensToImage(Callable):
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
            tokens_per_timestep = number_of_tokens // self.patch_embed.effective_time_dim
            h = int(np.sqrt(tokens_per_timestep))
            encoded = rearrange(
                x_no_token,
                "batch (t h w) e -> batch (t e) h w",
                e=self.embed_dim,
                t=self.patch_embed.effective_time_dim,
                h=h,
            )
            out.append(encoded)
        return out

from torch import nn, Tensor
import torch 
from terratorch.registry import BACKBONE_REGISTRY
import pdb
import warnings

class DecoderNotFoundError(Exception):
    pass

def extract_prefix_keys(d: dict, prefix: str) -> dict:
    extracted_dict = {}
    remaining_dict = {}
    for k, v in d.items():
        if k.startswith(prefix):
            extracted_dict[k[len(prefix) :]] = v
        else:
            remaining_dict[k] = v

    return extracted_dict, remaining_dict


def pad_images(imgs: Tensor, patch_size: int | list, padding: str) -> Tensor:
    p_t = 1
    if isinstance(patch_size, int):
         p_h = p_w = patch_size
    elif len(patch_size) == 1:
        p_h = p_w = patch_size[0]
    elif len(patch_size) == 2:
        p_h, p_w = patch_size
    elif len(patch_size) == 3:
        p_t, p_h, p_w = patch_size
    else:
        raise ValueError(f'patch size {patch_size} not valid, must be int or list of ints with length 1, 2 or 3.')

    # Double the patch size to ensure the resulting number of patches is divisible by 2 (required for many decoders)
    p_h, p_w = p_h * 2, p_w * 2

    if p_t > 1 and len(imgs.shape) < 5:
        raise ValueError(f"Multi-temporal padding requested (p_t = {p_t}) "
                         f"but no multi-temporal data provided (data shape = {imgs.shape}).")

    h, w = imgs.shape[-2:]
    t = imgs.shape[-3] if len(imgs.shape) > 4 else 1
    t_pad, h_pad, w_pad = (p_t - t % p_t) % p_t, (p_h - h % p_h) % p_h, (p_w - w % p_w) % p_w
    if t_pad > 0:
        # Multi-temporal padding
        imgs = torch.stack([
            nn.functional.pad(img, (0, w_pad, 0, h_pad, 0, t_pad), mode=padding)
            for img in imgs  # Apply per image to avoid NotImplementedError from torch.nn.functional.pad
        ])
    elif h_pad > 0 or w_pad > 0:
        imgs = torch.stack([
            nn.functional.pad(img, (0, w_pad, 0, h_pad), mode=padding)
            for img in imgs  # Apply per image to avoid NotImplementedError from torch.nn.functional.pad
        ])
    return imgs

def _get_backbone(backbone: str | nn.Module, **backbone_kwargs) -> nn.Module:
    use_temporal = backbone_kwargs.pop('use_temporal', False)
    pooling = backbone_kwargs.pop('temporal_pooling', 'mean')
    concat = backbone_kwargs.pop('temporal_concat', None)
    n_timestamps = backbone_kwargs.pop('temporal_n_timestamps', None)
    features_permute_op = backbone_kwargs.pop('temporal_features_permute_op', None)
    subset_lengths = backbone_kwargs.pop('temporal_subset_lengths', None)

    if isinstance(backbone, nn.Module):
        model = backbone
    else:
        model = BACKBONE_REGISTRY.build(backbone, **backbone_kwargs)

    # Apply TemporalWrapper inside _get_backbone
    if use_temporal:
        model = TemporalWrapper(model, pooling=pooling, concat=concat, n_timestamps=n_timestamps,
                                features_permute_op=features_permute_op, subset_lengths=subset_lengths)

    return model


class TemporalWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, pooling: str = 'mean', concat: bool = None, n_timestamps: int | None = None,
                 features_permute_op: list[int]| None = None, subset_lengths: list[int]| None = None):
        """
        Wrapper for applying a temporal encoder across multiple time steps.

        Args:
            encoder (nn.Module): The feature extractor (backbone).
            pooling (str): Type of pooling ('keep', 'concat', 'mean', 'max', 'diff'), 'diff' requires exactly two temporal elements after optional subset aggregation.
            concat (bool): Deprecated - 'concat' now integrated as pooling option.
            n_timestamps (int): Used to compute backbone out_channels when constructing encoder–decoder pipelines in case of 'concat' pooling.
            features_permute_op (list): Permutation operation to perform on the features before aggregation.
                This is in case the features to do not match either 'BCHW' or 'BLC' formats. It is reversed once
                aggregation has happened.
            subset_lengths (list[int], optional): If set, performs two-step temporal aggregation: 
                (1) split timesteps into defined subsets (e.g., [2, 3] → first 2 and last 3 timesteps),
                average within each; (2) apply the selected `pooling` across the resulting subset means. 
                Lengths must match the total number of timesteps. For pooling='diff', exactly two subsets are required.
        """
        super().__init__()

        # Warn if deprecated args are used
        if concat is not None:
            warnings.warn(
                "'concat' is deprecated in TemporalWrapper. "
                "Use pooling='concat' instead.",
                DeprecationWarning
                )

        # Check supported pooling modes
        supported_poolings = ["mean", "max", "diff", "keep", "concat"]
        if pooling not in supported_poolings:
            raise ValueError(f"Unsupported pooling '{pooling}', choose from {supported_poolings}.")
        
        self.encoder = encoder
        self.pooling = pooling
        self.features_permute_op = features_permute_op
        self.subset_lengths = subset_lengths

        # Validate subset_lengths defines two subsets for diff pooling and matches n_timestamps if set
        if subset_lengths is not None:
            if pooling == "diff" and len(subset_lengths) != 2:
                raise ValueError(
                    f"`diff` pooling requires exactly two subsets in `subset_lengths`, "
                    f"but got {len(subset_lengths)}: {subset_lengths}"
                )

            if n_timestamps is not None and sum(subset_lengths) != n_timestamps:
                raise ValueError(
                    f"The sum of `subset_lengths` must equal `n_timestamps` "
                    f"(got sum={sum(subset_lengths)}, n_timestamps={n_timestamps})."
                )

        # Precompute reverse permutation for restoring original dims after processing
        if features_permute_op is not None:
            self.reverse_permute_op = [None] * len(features_permute_op)
            for i, p in enumerate(features_permute_op):
                self.reverse_permute_op[p] = i
        else:
            self.reverse_permute_op = None

        if hasattr(encoder, "out_channels"):
            if pooling == "concat":
                if n_timestamps is None:
                    warnings.warn(
                        "Cannot derive `out_channels` for 'concat' pooling without `n_timestamps`"
                        "(Required to build Encoder–Decoder models).",
                        UserWarning,
                    )
                    self.out_channels = None
                else:
                    self.out_channels = [c * n_timestamps for c in encoder.out_channels]
            else:
                self.out_channels = encoder.out_channels
    
    
    def pool_temporal(self, stacked: torch.Tensor, pooling: str, subset_lengths: list[int] | None):
        """
        Pool per-timestep outputs based on the specified pooling method.
        """

        T = stacked.shape[1]  # number of timesteps

        if subset_lengths is not None:
            if sum(subset_lengths) != T:
                raise ValueError(
                    f"Sum of `subset_lengths` ({sum(subset_lengths)}) must equal timesteps ({T})."
                )

            # Split into subsets and average within each
            subset_splits = torch.split(stacked, subset_lengths, dim=1)
            stacked = torch.stack(
                [subset.mean(dim=1) for subset in subset_splits],
                dim=1
            )

        if pooling == "concat":
            return stacked.flatten(1, 2) # concat over time: [B, T*C, H, W]
        
        elif pooling == "max":
            return torch.max(stacked, dim=1).values # max over time: [B, C, H, W]
        
        elif pooling == "diff":
            if stacked.shape[1] != 2:
                raise ValueError(
                    f"`diff` pooling requires exactly two temporal elements "
                    f"(from input or after subset aggregation), got {stacked.shape[1]}. "
                    f"Consider to use `subset_lengths` to define two subsets."
                )
            return stacked[:, 0] - stacked[:, 1] # difference between two timesteps: [B, C, H, W]
        
        elif pooling == "mean":
            return torch.mean(stacked, dim=1) # mean over time: [B, C, H, W]
        
        else: 
            return stacked # "keep": return [B, T, C, H, W] sequence
        
    def reshape_5d(self, tensor: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """
        Reshape latent to [B, T, C, ...]. Appends dummy dim for ViTs.
        """
        if tensor.dim() == 4:  # Conv backbone: [BT, C, H, W]
            C, H, W = tensor.shape[1:]
            return tensor.reshape(B, T, C, H, W)
        elif tensor.dim() == 3:  # ViT backbone: [BT, L, C]
            L, C = tensor.shape[-2:]
            x = tensor.reshape(B, T, L, C)
            return x.permute(0, 1, 3, 2).unsqueeze(-1)
        raise ValueError(f"Expected latent tensor to be 3D or 4D, but got {tensor.dim()}.")
    
    def vit_postprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reorders L and C dim and removes helper size-1 dim if present.
        """
        if tensor.shape[-1] != 1:
            return tensor
        return tensor.squeeze(-1).movedim(-1, -2)
    
    def permute_op(self, tensor: torch.Tensor, permute_op: list[int]) -> torch.Tensor:
        """
        Apply a permutation operation to the tensor.
        """
        if permute_op is not None:
            if len(permute_op) != len(tensor.shape):
                raise ValueError(f"Expected permute_op to have same number of dimensions as tensor, but got {len(permute_op)} and {len(tensor.shape)}")
            return torch.permute(tensor, permute_op)
        return tensor
        
    def forward(self, 
                x: torch.Tensor | dict[str, torch.Tensor]
                ) -> list[torch.Tensor | dict[str, torch.Tensor]]:
        """
        Forward pass for temporal processing.

        Args:
            x: Input tensor of shape [B, C, T, H, W] or dict with shape {modality: [B, C_mod, T, H, W]}.
        Returns:
            List: A list of processed tensors/dicts, one per feature map.
        """

        # Handle dict input (multimodal) or single tensor
        is_dict = isinstance(x, dict)
        sample = next(iter(x.values())) if is_dict else x

        # Input must have 5 dims: [B, C, T, H, W]
        if sample.dim() != 5:
            raise ValueError(f"Expected input shape [B, C, T, H, W], got {tuple(sample.shape)}")

        # Flatten temporal dimension into batch for encoder forward pass
        if is_dict:
            B, _, T, H, W = sample.shape
            flat_input = {
                k: v.permute(0, 2, 1, 3, 4).reshape(-1, v.shape[1], *v.shape[3:])
                for k, v in x.items()
            }
        else:
            B, C, T, H, W = sample.shape
            flat_input = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # Run encoder backbone
        feat = self.encoder(flat_input)
        if not isinstance(feat, (list, tuple)):
            feat = [feat]
        else:
            feat = list(feat)

        outputs = []

        # Postprocess each feature map returned by encoder (layer outputs)
        for feature_map in feat:
            if isinstance(feature_map, dict): # multimodal output
                mod_keys = feature_map.keys()
                pooled = {}
                for k in mod_keys:
                    # Apply optional permutation before processing
                    feature_map[k] = self.permute_op(feature_map[k], self.features_permute_op)

                    # Reshape back to [B, T, ...]
                    mod_stacked = self.reshape_5d(feature_map[k], B, T)

                    # Temporal pooling
                    pooled[k] = self.pool_temporal(mod_stacked, self.pooling, self.subset_lengths)

                    # Reverse permutation to restore original dim order
                    pooled[k] = self.permute_op(pooled[k], self.reverse_permute_op)

                    # ViT postprocessing if needed
                    if pooled[k].shape[-1] == 1: 
                        pooled[k] = self.vit_postprocess(pooled[k])

            else: # single-modality feature map
                feature_map = self.permute_op(feature_map, self.features_permute_op)
                stacked = self.reshape_5d(feature_map, B, T)
                pooled = self.pool_temporal(stacked, self.pooling, self.subset_lengths)
                pooled = self.permute_op(pooled, self.reverse_permute_op)

                if pooled.shape[-1] == 1: 
                    pooled = self.vit_postprocess(pooled)
                    
            outputs.append(pooled)

        return outputs
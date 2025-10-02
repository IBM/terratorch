from torch import nn, Tensor
import torch 
from terratorch.registry import BACKBONE_REGISTRY
from typing import Dict
import pdb
import warnings
from terratorch.registry import BACKBONE_REGISTRY
import pdb
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.image_list import ImageList
from typing import Any, Optional

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

    if isinstance(backbone, nn.Module):
        model = backbone
    else:
        model = BACKBONE_REGISTRY.build(backbone, **backbone_kwargs)

    # Apply TemporalWrapper inside _get_backbone
    if use_temporal:
        model = TemporalWrapper(model, pooling=pooling, concat=concat, n_timestamps=n_timestamps,
                                features_permute_op=features_permute_op)

    return model


class TemporalWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, pooling: str = 'mean', concat: bool = None, n_timestamps: int | None = None,
                 features_permute_op: list[int] = None):
        """
        Wrapper for applying a temporal encoder across multiple time steps.

        Args:
            encoder (nn.Module): The feature extractor (backbone).
            pooling (str): Type of pooling ('mean', 'max', 'diff') with 'diff' working only with 2 timestamps.
            concat (bool): Deprecated - 'concat' now integrated as pooling option.
            n_timestamps (int): Used only to compute backbone out_channels when constructing encoder–decoder pipelines in case of 'concat' pooling.
            features_permute_op (list): Permutation operation to perform on the features before aggregation.
                This is in case the features to do not match either 'BCHW' or 'BLC' formats. It is reversed once
                aggregation has happened.
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
    

    def subtract_along_dim1(self, tensor: torch.Tensor):
        # Diff pooling: Difference between first and second timestep
        return tensor[:, 0, ...] - tensor[:, 1, ...]
    
    def pool_temporal(self, stacked: torch.Tensor, pooling: str):
        """
        Pool per-timestep outputs based on the specified pooling method.
        """
        if pooling == "concat":
            return stacked.flatten(1, 2) # concat over time → [B, T*C, ...] 
        elif pooling == "max":
            return torch.max(stacked, dim=1).values # max over time
        elif pooling == "diff":
            return self.subtract_along_dim1(stacked) # difference between first two timesteps
        elif pooling == "mean":
            return torch.mean(stacked, dim=1) # mean over time
        else: 
            return stacked # "keep" → return [B, T, ...] sequence
        
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
        if isinstance(x, Dict):
            first_key = [key for key in x.keys() if "image" in key][0]
            x_dim = x[first_key].dim()
        else:
            x_dim = x.dim()
        if x_dim != 5:
            raise ValueError(f"Expected input shape [B, C, T, H, W], but got {x.shape}")


        if isinstance(x, Dict):
            check_diff = (x[first_key].shape[2] != 2)
        else:
            check_diff = (x.shape[2] != 2)
        if (self.pooling_type == 'diff') & check_diff:
            raise ValueError(f"Expected 2 timestamps for aggregation method 'diff'")


        if isinstance(x, Dict):
            batch_size, _, timesteps, _, _ = x[first_key].shape
        else:
            batch_size, _, timesteps, _, _ = x.shape

        # Initialize lists to store feature maps at each timestamp
        num_feature_maps = None  # Will determine dynamically
        features_per_map = []  # Stores feature maps across timestamps

        for t in range(timesteps):
            if isinstance(x, Dict):
                feat = self.encoder({key: val[:, :, t, :, :] for key, val in x.items()})
            else:
                feat = self.encoder(x[:, :, t, :, :])  # Extract features at timestamp t
                
            if not isinstance(feat, list):  # If the encoder outputs a single feature map, convert to list
                if isinstance(feat, tuple):
                    feat = list(feat)
                else:
                    feat = [feat]

            if num_feature_maps is None:
                num_feature_maps = len(feat)  # Determine how many feature maps the encoder produces

            for i, feature_map in enumerate(feat):
                if len(features_per_map) <= i:
                    features_per_map.append([])  # Create list for each feature map

                feature_map = feature_map[0] if isinstance(feature_map, tuple) else feature_map
                if self.features_permute_op is not None:
                    if len(self.features_permute_op) != len(feature_map.shape):
                        ValueError(f"Expected features_permute_op to have same number of dimensions of features, but got {len(self.features_permute_op)} and {len(feature_map.shape)}")
                    feature_map = torch.permute(feature_map, self.features_permute_op)


                features_per_map[i].append(feature_map)  # Store feature map at time t


        
                    
        # Stack features along the temporal dimension
        for i in range(num_feature_maps):
            try:
                features_per_map[i] = torch.stack(features_per_map[i], dim=2)  # Shape: [B, C', T, H', W']
            except RuntimeError as e:
                raise

        # Apply pooling or concatenation
        if self.concat:
            features_per_map_agg = [feat.reshape(batch_size, -1, feat.shape[-2], feat.shape[-1]) if len(feat.shape) == 5 else feat.reshape(batch_size, feat.shape[-3], -1) for feat in features_per_map]
        elif self.pooling_type == "max":
            features_per_map_agg = [torch.max(feat, dim=2)[0] for feat in features_per_map]  # Max pooling across T
        elif self.pooling_type == "diff":
            features_per_map_agg = [feat[:, :, 0, ...] - feat[:, :, 1, ...] for feat in features_per_map]
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
                    pooled[k] = self.pool_temporal(mod_stacked, self.pooling)

                    # Reverse permutation to restore original dim order
                    pooled[k] = self.permute_op(pooled[k], self.reverse_permute_op)

                    # ViT postprocessing if needed
                    if pooled[k].shape[-1] == 1: 
                        pooled[k] = self.vit_postprocess(pooled[k])

            else: # single-modality feature map
                feature_map = self.permute_op(feature_map, self.features_permute_op)
                stacked = self.reshape_5d(feature_map, B, T)
                pooled = self.pool_temporal(stacked, self.pooling)
                pooled = self.permute_op(pooled, self.reverse_permute_op)

                if pooled.shape[-1] == 1: 
                    pooled = self.vit_postprocess(pooled)
                    
            outputs.append(pooled)

        return outputs


# def _get_backbone(backbone: str | nn.Module, **backbone_kwargs) -> nn.Module:

#     use_temporal = backbone_kwargs.pop('use_temporal', None)
#     temporal_pooling = backbone_kwargs.pop('temporal_pooling', None)
#     concat = backbone_kwargs.pop('temporal_concat', None)
#     n_timestamps = backbone_kwargs.pop('temporal_n_timestamps', None)
#     features_permute_op = backbone_kwargs.pop('temporal_features_permute_op', None)
    
#     if isinstance(backbone, nn.Module):
#         model = backbone
#     else:
#         model = BACKBONE_REGISTRY.build(backbone, **backbone_kwargs)

#     # Apply TemporalWrapper inside _get_backbone
#     if use_temporal:
#         model = TemporalWrapper(model, pooling=temporal_pooling, concat=concat, n_timestamps=n_timestamps, features_permute_op=features_permute_op)

#     return model

# def subtract_along_dim2(tensor: torch.Tensor):
#     return tensor[:, :, 0, ...] - tensor[:, :, 1, ...]


# class TemporalWrapper(nn.Module):
#     def __init__(self, encoder: nn.Module, pooling="mean", concat=False, n_timestamps=None, features_permute_op = None):
#         """
#         Wrapper for applying a temporal encoder across multiple time steps.

#         Args:
#             encoder (nn.Module): The feature extractor (backbone).
#             pooling (str): Type of pooling ('mean', 'max', 'diff') with 'diff' working only with 2 timestamps.
#             concat (bool): Whether to concatenate features instead of pooling.
#             n_timestamps (int): Number of timestamps. Necessary in case of concat.
#             features_permute_op (list): Permutation operation to perform on the features before aggregation. This is in case the features to do not match either 'BCHW' or 'BLC' formats. It is reversed once aggregation has happened.
#         """
#         super().__init__()
#         self.encoder = encoder
#         self.concat = concat
#         self.pooling_type = pooling
#         self.n_timestamps = n_timestamps
#         self.features_permute_op = features_permute_op

#         if ((not concat) & (pooling not in ["mean", "max", "diff"])):
#             raise ValueError("Pooling must be 'mean', 'max' or 'diff'")
#         # Ensure the encoder has an out_channels attribute
#         if hasattr(encoder, "out_channels"):
#             self.out_channels = [x * (1 if not concat else self.n_timestamps) for x in encoder.out_channels]
#         else:
#             raise AttributeError("Encoder must have an `out_channels` attribute.")


#     def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
#         """
#         Forward pass for temporal processing.

#         Args:
#             x (Tensor): Input tensor of shape [B, C, T, H, W].

#         Returns:
#             List[Tensor]: A list of processed tensors, one per feature map.
#         """

#         if x.dim() != 5:
#             raise ValueError(f"Expected input shape [B, C, T, H, W], but got {x.shape}")

#         if (self.pooling_type == 'diff') & (x.shape[2] != 2):
#             raise ValueError(f"Expected 2 timestamps for aggregation method 'diff'")

#         batch_size, _, timesteps, _, _ = x.shape

#         # Initialize lists to store feature maps at each timestamp
#         num_feature_maps = None  # Will determine dynamically
#         features_per_map = []  # Stores feature maps across timestamps

#         for t in range(timesteps):
#             feat = self.encoder(x[:, :, t, :, :])  # Extract features at timestamp t
#             if not isinstance(feat, list):  # If the encoder outputs a single feature map, convert to list
#                 if isinstance(feat, tuple):
#                     feat = list(feat)
#                 else:
#                     feat = [feat]

#             if num_feature_maps is None:
#                 num_feature_maps = len(feat)  # Determine how many feature maps the encoder produces

#             for i, feature_map in enumerate(feat):
#                 if len(features_per_map) <= i:
#                     features_per_map.append([])  # Create list for each feature map
#                 feature_map = feature_map[0] if isinstance(feature_map, tuple) else feature_map
#                 if self.features_permute_op is not None:
#                     if len(self.features_permute_op) != len(feature_map.shape):
#                         ValueError(f"Expected features_permute_op to have same number of dimensions of features, but got {len(self.features_permute_op)} and {len(feature_map.shape)}")
#                     # print('Old shape:', feature_map.shape)
#                     feature_map = torch.permute(feature_map, self.features_permute_op)
#                     # print('New shape:', feature_map.shape)
                    
#                 features_per_map[i].append(feature_map)  # Store feature map at time t            
#         # Stack features along the temporal dimension
#         for i in range(num_feature_maps):
#             try:
#                 features_per_map[i] = torch.stack(features_per_map[i], dim=2)  # Shape: [B, C', T, H', W']
#             except RuntimeError as e:
#                 raise
#         # Apply pooling or concatenation
#         if self.concat:
#             features_per_map_agg = [feat.reshape(batch_size, -1, feat.shape[-2], feat.shape[-1]) if len(feat.shape) == 5 else feat.reshape(batch_size, feat.shape[-3], -1) for feat in features_per_map]
#         elif self.pooling_type == "max":
#             features_per_map_agg = [torch.max(feat, dim=2)[0] for feat in features_per_map]  # Max pooling across T
#         elif self.pooling_type == "diff":
#             features_per_map_agg = [feat[:, :, 0, ...] - feat[:, :, 1, ...] for feat in features_per_map]
#         else:
#             features_per_map_agg = [torch.mean(feat, dim=2) for feat in features_per_map]
        
#         if self.features_permute_op is not None:
#             # use position in the permutation op as the value of the permuation and the original value of permutation as the index
#             reverse_permuation_op = [None] * len(self.features_permute_op)
#             for i in range(len(self.features_permute_op)):
#                 reverse_permuation_op[self.features_permute_op[i]] = i
#             features_per_map_agg = [torch.permute(feat, reverse_permuation_op) for feat in features_per_map_agg]

#         return features_per_map_agg
    

class TerratorchGeneralizedRCNNTransform(GeneralizedRCNNTransform):
    
    def init(min_size: int,
             max_size: int,
             image_mean: list[float],
             image_std: list[float],
             size_divisible: int = 32,
             fixed_size: Optional[tuple[int, int]] = None,
             **kwargs: Any):
        
        super().__init__(min_size,
                         max_size,
                         image_mean,
                         image_std,
                         size_divisible,
                         fixed_size,
                         **kwargs)
        
    def forward(
        self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None
    ) -> tuple[ImageList, Optional[list[dict[str, Tensor]]]]:
        images = [img for img in images]
        if targets is not None:
            # make a copy of targets to avoid modifying it in-place
            # once torchscript supports dict comprehension
            # this can be simplified as follows
            # targets = [{k: v for k,v in t.items()} for t in targets]
            targets_copy: list[dict[str, Tensor]] = []
            for t in targets:
                data: dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index
        image_sizes = [img.shape[-2:] for img in images]
        images = [x[None] for x in images]
        images = torch.cat(images, 0)
        image_sizes_list: list[tuple[int, int]] = []
        for image_size in image_sizes:
            torch._assert(
                len(image_size) == 2,
                f"Input tensors expected to have in the last two elements H and W, instead got {image_size}",
            )
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets
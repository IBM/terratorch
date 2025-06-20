from torch import nn, Tensor
import torch 
from terratorch.registry import BACKBONE_REGISTRY
import pdb

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
    use_temporal = backbone_kwargs.pop('use_temporal', None)
    temporal_pooling = backbone_kwargs.pop('temporal_pooling', None)
    concat = backbone_kwargs.pop('temporal_concat', None)
    if isinstance(backbone, nn.Module):
        model = backbone
    else:
        model = BACKBONE_REGISTRY.build(backbone, **backbone_kwargs)

    # Apply TemporalWrapper inside _get_backbone
    if use_temporal:
        model = TemporalWrapper(model, pooling=temporal_pooling, concat=concat)

    return model

def subtract_along_dim2(tensor: torch.Tensor):
    return tensor[:, :, 0, ...] - tensor[:, :, 1, ...]


class TemporalWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, pooling="mean", concat=False, n_timestamps=None, features_permute_op = None):
        """
        Wrapper for applying a temporal encoder across multiple time steps.

        Args:
            encoder (nn.Module): The feature extractor (backbone).
            pooling (str): Type of pooling ('mean', 'max', 'diff') with 'diff' working only with 2 timestamps.
            concat (bool): Whether to concatenate features instead of pooling.
            n_timestamps (int): Number of timestamps. Necessary in case of concat.
            features_permute_op (list): Permutation operation to perform on the features before aggregation. This is in case the features to do not match either 'BCHW' or 'BLC' formats. It is reversed once aggregation has happened.
        """
        super().__init__()
        self.encoder = encoder
        self.concat = concat
        self.pooling_type = pooling
        self.n_timestamps = n_timestamps
        self.features_permute_op = features_permute_op

        if pooling not in ["mean", "max", "diff"]:
            raise ValueError("Pooling must be 'mean', 'max' or 'diff'")

        # Ensure the encoder has an out_channels attribute
        if hasattr(encoder, "out_channels"):
            self.out_channels = encoder.out_channels * (1 if not concat else self.n_timestamps)
        else:
            raise AttributeError("Encoder must have an `out_channels` attribute.")


    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass for temporal processing.

        Args:
            x (Tensor): Input tensor of shape [B, C, T, H, W].

        Returns:
            List[Tensor]: A list of processed tensors, one per feature map.
        """

        if x.dim() != 5:
            raise ValueError(f"Expected input shape [B, C, T, H, W], but got {x.shape}")

        if (self.pooling_type == 'diff') & (x.shape[2] != 2):
            raise ValueError(f"Expected 2 timestamps for aggregation method 'diff'")

        batch_size, _, timesteps, _, _ = x.shape

        # Initialize lists to store feature maps at each timestamp
        num_feature_maps = None  # Will determine dynamically
        features_per_map = []  # Stores feature maps across timestamps

        for t in range(timesteps):
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
                    # print('Old shape:', feature_map.shape)
                    feature_map = torch.permute(feature_map, self.features_permute_op)
                    # print('New shape:', feature_map.shape)
                    
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
            features_per_map_agg = [torch.mean(feat, dim=2) for feat in features_per_map]
        
        if self.features_permute_op is not None:
            # use position in the permutation op as the value of the permuation and the original value of permutation as the index
            reverse_permuation_op = [None] * len(self.features_permute_op)
            for i in range(len(self.features_permute_op)):
                reverse_permuation_op[self.features_permute_op[i]] = i
            features_per_map_agg = [torch.permute(feat, reverse_permuation_op) for feat in features_per_map_agg]

        return features_per_map_agg

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
    def __init__(self, encoder: nn.Module, pooling: str, concat=None, n_timestamps=None, features_permute_op = None):
        """
        Wrapper for applying a temporal encoder across multiple time steps.

        Args:
            encoder (nn.Module): The feature extractor (backbone).
            pooling (str): Type of pooling ('mean', 'max', 'diff') with 'diff' working only with 2 timestamps.
            concat (bool): Deprecated - 'concat' now intagrated as pooling option.
            n_timestamps (int): Deprecated 
            features_permute_op (list): Permutation operation to perform on the features before aggregation. This is in case the features to do not match either 'BCHW' or 'BLC' formats. It is reversed once aggregation has happened.
        """
        super().__init__()

        if concat or n_timestamps is not None:
            print("Warning: 'concat' and 'n_timestamps' are deprecated in TemporalWrapper. Use concate as 'pooling' type instead.")

        supported_poolings = ["mean", "max", "diff", "keep", "concat"]
        if pooling not in supported_poolings:
            raise ValueError(f"Unsupported pooling '{pooling}', choose from {supported_poolings}.")
        
        self.encoder = encoder
        self.pooling = pooling
        self.features_permute_op = features_permute_op

    def pool_temporal(self, stacked: torch.Tensor, pooling: str):
        """
        Pool per-timestep outputs based on the specified pooling method.
        """
        if pooling == "concat":
            return stacked.flatten(1, 2)    
        elif pooling == "max":
            return torch.max(stacked, dim=2).values
        elif pooling == "diff":
            return subtract_along_dim2(stacked)
        elif pooling == "mean":
            return torch.mean(stacked, dim=2)
        else: 
            return stacked
        
    def reshape_5d(self, tensor: torch.Tensor, batch_size: int, timesteps: int) -> torch.Tensor:
        """
        Reshape `tensor` to 5d: [B, C, T, H, W], channel dim C will be 1 for ViT outputs.
        """
        H_lat, W_lat = tensor.shape[-2:]
        return tensor.view(batch_size, -1, timesteps, H_lat, W_lat)
    
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

        is_dict = isinstance(x, dict)
        sample = next(iter(x.values())) if is_dict else x

        if sample.dim() != 5:
            raise ValueError(f"Expected input shape [B, C, T, H, W], got {tuple(sample.shape)}")
        
        if self.features_permute_op is not None:
            self.reverse_permute_op = [None] * len(self.features_permute_op)
            for i, p in enumerate(self.features_permute_op):
                self.reverse_permute_op[p] = i
        else:
            self.reverse_permute_op = None

        if is_dict:
            B, _, T, H, W = sample.shape
            flat_input = {
                k: v.permute(0, 2, 1, 3, 4).reshape(-1, v.shape[1], *v.shape[3:])
                for k, v in x.items()
            }

        else:
            B, C, T, H, W = sample.shape
            flat_input = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        feat = self.encoder(flat_input)

        if isinstance(feat, tuple):
            feat = list(feat)
        elif isinstance(feat, list):
            feat = feat
        else:
            feat = [feat]

        outputs = []

        for feature_map in feat:
            if isinstance(feature_map, dict):
                mod_keys = feature_map.keys()
                pooled = {}
                for k in mod_keys:
                    
                    feature_map[k] = self.permute_op(feature_map[k], self.features_permute_op)
                    mod_stacked = self.reshape_5d(feature_map[k], B, T)
                    pooled[k] = self.pool_temporal(mod_stacked, self.pooling)
                    pooled[k] = self.permute_op(pooled[k], self.reverse_permute_op)
    
                    if pooled[k].shape[1] == 1: # Squeeze if only one channel (ViT output)
                        pooled[k] = pooled[k].squeeze(1)

            else:
                feature_map = self.permute_op(feature_map, self.features_permute_op)
                stacked = self.reshape_5d(feature_map, B, T)
                pooled = self.pool_temporal(stacked, self.pooling)
                pooled = self.permute_op(pooled, self.reverse_permute_op)

                if pooled.shape[1] == 1: # Squeeze if only one channel dimension (ViT output) - TODO: Check if this should be removed
                    pooled = pooled.squeeze(1)
                
            outputs.append(pooled)

        return outputs

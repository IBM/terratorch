# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
import torch
from torch import nn
from functools import partial

from .encoder_embeddings import ImageEncoderEmbedding
from .tm_utils import Block, LayerNorm
from .modality_info import MODALITY_INFO


def build_modality_embeddings(modalities, img_size=None, dim=None, patch_size=None):
    mod_embeddings = {}
    mod_name_mapping = {}
    for modality in modalities:
        # New modalities can be provided as {'name': <num_channels>}
        if isinstance(modality, dict):
            for key, value in modality.items():
                if isinstance(value, nn.Module):
                    mod_embeddings[key] = value
                elif isinstance(value, int):
                    mod_embeddings[key] = ImageEncoderEmbedding(num_channels=value, dim_tokens=dim, image_size=img_size,
                                                                    patch_size=patch_size, sincos_pos_emb=True)
                else:
                    raise ValueError(f'Modalities must be provided as a list of strings or dicts, or as a dict with '
                                     f'the values being nn.Module or int (number of channels of the modality). '
                                     f'Found {key}: {value} ({type(value)})')
                mod_name_mapping[key] = key
            continue

        # Cover multiple naming conventions
        modality_renamed = (modality.lower()
                            .replace('s2', 'sen2')
                            .replace('s1', 'sen1')
                            .replace('text', 'caption')
                            .replace('coordinates', 'coords')
                            )

        # Get modality key in MODALITY_INFO
        if 'sen2l2a' in modality_renamed:
            key = 'untok_sen2l2a@224'
        elif 'sen2l1c' in modality_renamed:
            key = 'untok_sen2l1c@224'
        elif 'sen1rtc' in modality_renamed:
            key = 'untok_sen1rtc@224'
        elif 'sen1grd' in modality_renamed:
            key = 'untok_sen1grd@224'
        elif 'rgb' in modality_renamed:
            key = 'untok_sen2rgb@224'
        elif 'dem' in modality_renamed:
            key = 'untok_dem@224'
        elif 'caption' in modality_renamed:
            raise NotImplementedError('Captions are not yet supported.')
        elif 'coords' in modality_renamed:
            raise NotImplementedError('Captions are not yet supported.')
        else:
            key = modality

        if key in MODALITY_INFO.keys():
            mod_info = MODALITY_INFO[key]
            mod_embeddings[key] = mod_info['encoder_embedding'](image_size=img_size, dim_tokens=dim, **mod_info)
            mod_name_mapping[modality] = key  # Requires manual mapping for loading model weights
        else:
            raise NotImplementedError(f'Could not find modality {modality} in default modality info.')

    return mod_embeddings, mod_name_mapping


class TerraMindViT(nn.Module):
    """Modified TerraMind model, adapted to behave as a raw data-only ViT.

    Args:
        img_size (int): Input image size.
        modalities (list, dict, optional): List of modality keys and dicts, or dict with modality keys and values being
            ints (num_channels of modality) or nn.Module (patch embedding layer).
        merge_method (str, optional): Specify how the output is merged for further processing. One of 'mean', 'max',
            'concat', 'dict', or None. 'mean', 'max', and 'concat' are dropping all sequence modality tokens, split all
            image modality tokens and reduce the by applying the appropriate method. 'dict' splits all tokens into a
            dictionary {'modality': torch.Tensor}. Defaults to 'mean'.
        patch_size (int): Patch size.
        in_chans (int): Number of input image channels.
        dim (int): Patch embedding dimension.
        encoder_depth (int): Depth of ViT / number of encoder blocks.
        num_heads (int): Number of attention heads in each ViT block.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        proj_bias (bool): If True, adds a bias to the attention out proj layer.
        mlp_bias (bool): If True, adds a learnable bias for the feedforward.
        drop_path_rate (float): Stochastic depth rate.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        modality_drop_rate (float): Drop modality inputs during training.
        act_layer (nn.Module): Activation layer.
        norm_layer (nn.Module): Normalization layer.
        gated_mlp (bool): If True, makes the feedforward gated (e.g., for SwiGLU)
        qk_norm (bool): If True, normalizes the query and keys (as in ViT-22B)
        use_act_checkpoint (bool): If True, use activation checkpointing.
        encoder_norm (bool): If True, adds a norm layer after the last encoder block.
    """
    def __init__(
        self,
        img_size: int = 224,
        modalities: list | dict[str, int | nn.Module] | None = None,
        merge_method: str | None = 'mean',
        patch_size: int = 16,
        in_chans: int = 3,
        dim: int = 768,
        encoder_depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        modality_drop_rate: float = 0.0,
        act_layer: torch.Tensor = nn.GELU,
        norm_layer: partial | nn.Module = partial(LayerNorm, eps=1e-6),
        gated_mlp: bool = False,  # Make the feedforward gated for e.g. SwiGLU
        qk_norm: bool = False,
        encoder_norm: bool = True,
    ):
        super().__init__()

        if modalities is None or len(modalities) == 0:
            # Init new image modality
            modalities = [{'image': in_chans}]
        elif isinstance(modalities, dict):
            modalities = [modalities]
        elif not isinstance(modalities, list):
            raise ValueError(f'Modalities must be None, a list of modality keys or a dict with ints/embedding layers.')

        # Build embedding layers for all defined modalities
        mod_embeddings, mod_name_mapping = build_modality_embeddings(modalities, img_size=img_size, dim=dim,
                                                                     patch_size=patch_size)
        self.encoder_embeddings = nn.ModuleDict(mod_embeddings)
        self.mod_name_mapping = mod_name_mapping
        self.modalities = list(mod_name_mapping.keys())  # Further code expects list
        self.output_mod_name_mapping = {v: k for k, v in mod_name_mapping.items()}

        self.img_size = img_size
        self.merge_method = merge_method
        self.image_modalities = [key for key, value in self.encoder_embeddings.items()
                                 if isinstance(value, ImageEncoderEmbedding)]
        self.modality_drop_rate = modality_drop_rate
        assert 0 <= self.modality_drop_rate <= 1, "modality_drop_rate must be in [0, 1]"
        # New learned parameter for handling missing modalities
        if self.merge_method == 'concat':
            self.missing_mod_token = nn.Parameter(torch.Tensor(dim))

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, encoder_depth)]

        self.encoder = nn.ModuleList([
            Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias,
                  mlp_bias=mlp_bias, drop_path=dpr[i], drop=drop_rate, attn_drop=attn_drop_rate, act_layer=act_layer,
                  norm_layer=norm_layer, gated_mlp=gated_mlp, qk_norm=qk_norm)
            for i in range(encoder_depth)
        ])

        # Needed for terratorch decoders
        if merge_method == 'concat':
            self.out_channels = [dim * len(self.image_modalities) for i in range(encoder_depth)]
        else:
            self.out_channels = [dim for i in range(encoder_depth)]

        self.encoder_norm = norm_layer(dim) if encoder_norm else nn.Identity()

        # Weight init
        self.init_weights()

    def init_weights(self):
        """Weight initialization following MAE's initialization scheme"""

        for name, m in self.named_modules():
            # Skipping tokenizers to avoid reinitializing them
            if "tokenizer" in name:
                continue
            # Linear
            elif isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm) or isinstance(m, LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

            # Embedding
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            # Conv2d
            elif isinstance(m, nn.Conv2d):
                if '.proj' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = set()

        for mod, emb_module in self.encoder_embeddings.items():
            if hasattr(emb_module, 'no_weight_decay'):
                to_skip = emb_module.no_weight_decay()
                to_skip = set([f'encoder_embeddings.{mod}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def forward(self, d: dict[str, torch.Tensor] | torch.Tensor | None = None, **kwargs) -> list[torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            d (dict, torch.Tensor): Dict of inputs or input tensor with shape (B, C, H, W)

            Alternatively, keyword arguments with modality=tensor.

        Returns:
            list[torch.Tensor]: List of transformer layer outputs. Shape (B, L, D).
        """
        # Handle single image modality
        if isinstance(d, torch.Tensor):
            # Assuming first modality
            d = {self.modalities[0]: d}
        elif d is None:
            d = {}
            assert len(kwargs), "No input provided."

        # Add additional keyword args to input dict
        for key, value in kwargs.items():
            d[key] = value

        if self.training and self.modality_drop_rate:
            # Drop random modalities during training
            for key in random.sample(list(d.keys()), k=len(d) - 1):
                if random.random() < self.modality_drop_rate:
                    _ = d.pop(key)

        x = []
        num_tokens = []
        image_mod = []
        for mod, tensor in d.items():
            assert mod in self.mod_name_mapping.keys(), \
                f'No patch embedding layer found for modality {mod}.'

            mod_dict = self.encoder_embeddings[self.mod_name_mapping[mod]](tensor)
            # Add embeddings to patchified data
            x.append(mod_dict['x'] + mod_dict['emb'])
            num_tokens.append(mod_dict['x'].shape[-2])
            image_mod.append(self.mod_name_mapping[mod] in self.image_modalities)

        # Concatenate along token dim
        x = torch.cat(x, dim=1)  # Shape: (B, N, D)

        out = []
        for block in self.encoder:
            x = block(x)
            out.append(x.clone())

        out[-1] = self.encoder_norm(x)  # Shape: (B, N, D)

        def _unstack_image_modalities(x):
            x = torch.split(x, num_tokens, dim=1)  # Split tokens by modality
            x = [m for m, keep in zip(x, image_mod) if keep]  # Drop sequence modalities
            x = torch.stack(x, dim=1)  # (B, M, N, D)
            return x

        # Merge tokens from different modalities
        if self.merge_method == 'mean':
            out = [_unstack_image_modalities(x) for x in out]
            out = [x.mean(dim=1) for x in out]

        elif self.merge_method == 'max':
            out = [_unstack_image_modalities(x) for x in out]
            out = [x.max(dim=1)[0] for x in out]

        elif self.merge_method == 'concat':
            out = [_unstack_image_modalities(x) for x in out]
            if len(d) < len(self.image_modalities):
                # Handle missing modalities with missing_mod_token
                num_missing = len(self.image_modalities) - len(d)
                missing_tokens = self.missing_mod_token.repeat(out[-1].shape[0], num_missing, out[-1].shape[2], 1)
                out = [torch.cat([x, missing_tokens], dim=1) for x in out]
            # Concat along embedding dim
            out = [torch.cat(x.unbind(dim=1), dim=-1) for x in out]

        elif self.merge_method == 'dict':
            out = [torch.split(x, num_tokens, dim=1) for x in out]
            out = [{self.output_mod_name_mapping[mod]: x[i] for i, mod in enumerate(d.keys())} for x in out]

        elif self.merge_method is None:
            pass  # Do nothing
        else:
            raise NotImplementedError(f'Merging method {self.merge_method} is not implemented. '
                                      f'Select one of mean, max or concat.')

        return out

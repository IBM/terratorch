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

import random
import warnings

import numpy as np
import torch
from torch import nn
from functools import partial
from einops import rearrange
import torch.nn.functional as F

from .encoder_embeddings import ImageEncoderEmbedding, ImageTokenEncoderEmbedding
from .decoder_embeddings import ImageTokenDecoderEmbedding
from .tm_utils import LayerNorm
from .modality_info import MODALITY_INFO
from .generate import GenerationSampler, build_chained_generation_schedules
from .terramind import TerraMind
from terratorch.models.backbones.terramind.tokenizer.tokenizer_register import (
    terramind_v1_tokenizer_s2l2a,
    terramind_v1_tokenizer_s1rtc,
    terramind_v1_tokenizer_s1grd,
    terramind_v1_tokenizer_dem,
    terramind_v1_tokenizer_lulc,
    terramind_v1_tokenizer_ndvi
)

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
        if modality_renamed in MODALITY_INFO.keys():
            key = modality_renamed
        elif 'sen2l2a' in modality_renamed:
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
        elif 'lulc' in modality_renamed:
            key = 'tok_lulc@224'
        elif 'ndvi' in modality_renamed:
            key = 'tok_ndvi@224'
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


def build_output_modality_embeddings(modalities, img_size=None, dim=None, patch_size=None):
    mod_embeddings = {}
    mod_name_mapping = {}
    for modality in modalities:
        # Cover multiple naming conventions
        modality_renamed = (modality.lower()
                            .replace('s2', 'sen2')
                            .replace('s1', 'sen1')
                            .replace('text', 'caption')
                            .replace('coordinates', 'coords')
                            )

        # Get modality key in MODALITY_INFO
        if 'sen2' in modality_renamed:
            key = 'tok_sen2l2a@224'
        elif 'sen1rtc' in modality_renamed:
            key = 'tok_sen1rtc@224'
        elif 'sen1' in modality_renamed:  # Default to S1GRD if not specified
            key = 'tok_sen1grd@224'
        elif 'dem' in modality_renamed:
            key = 'tok_dem@224'
        elif 'lulc' in modality_renamed:
            key = 'tok_lulc@224'
        elif 'ndvi' in modality_renamed:
            key = 'tok_ndvi@224'
        elif 'caption' in modality_renamed:
            raise NotImplementedError('Captions are not yet supported.')
        elif 'coords' in modality_renamed:
            raise NotImplementedError('Captions are not yet supported.')
        else:
            key = modality

        if key in MODALITY_INFO.keys():
            mod_info = MODALITY_INFO[key]
            mod_embeddings[key] = mod_info['decoder_embedding'](image_size=img_size, dim_tokens=dim, **mod_info)
            mod_name_mapping[modality] = key  # Requires manual mapping for loading model weights
        else:
            raise NotImplementedError(f'Could not find modality {modality} in default modality info.'
                                      f'Available modalities: S2L2A, S1RTC, S1GRD, DEM, LULC, NDVI.')

    return mod_embeddings, mod_name_mapping



def build_tokenizer(input_modalities, output_modalities, pretrained):
    tokenizer_dict = {
        'tok_sen2l2a@224': terramind_v1_tokenizer_s2l2a,
        'tok_sen1rtc@224': terramind_v1_tokenizer_s1rtc,
        'tok_sen1grd@224': terramind_v1_tokenizer_s1grd,
        'tok_dem@224': terramind_v1_tokenizer_dem,
        'tok_lulc@224': terramind_v1_tokenizer_lulc,
        'tok_ndvi@224': terramind_v1_tokenizer_ndvi
    }
    # TODO: Add loading only encoder/decoder
    tokenizer = {}
    for modality in input_modalities:
        if modality in tokenizer_dict:
            tokenizer[modality] = tokenizer_dict[modality](pretrained=pretrained)

    for modality in output_modalities:
        if modality in tokenizer_dict and modality not in tokenizer:
            tokenizer[modality] = tokenizer_dict[modality](pretrained=pretrained)
        else:
            warnings.warn(f'Tokenizer for output modality {modality} not found.')

    tokenizer = nn.ModuleDict(tokenizer)

    return tokenizer


class TerraMindGeneration(nn.Module):
    """Modified TerraMind model for a "Thinking in Modalities" approach.

    Args:
        img_size (int): Input image size.
        modalities (list, dict, optional): List of modality keys and dicts, or dict with modality keys and values being
            ints (num_channels of modality) or nn.Module (patch embedding layer).
        output_modalities (list, optional): List of tokenized modalities used for the TiM approach. The TiM outputs are
            generated in the same order as specified in the given list. Defaults to [tbd].
        decoding_steps (list, int): Number of decoding steps for each TiM modality. Defaults to 1.
        temps (list, float): Sampling temperatures for each TiM modality. Defaults to 1.0.
        top_p (float): Top-p sampling threshold for TiM modalities. Ignored if set to 0.0. Defaults to 0.8.
        top_k (int): Top-k sampling threshold for TiM modalities. Ignored if set to 0. Defaults to 0.
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
            modalities: list[str] | dict[str, int | nn.Module] | None = None,
            output_modalities: list[str] | None = None,
            pretrained: bool = False,
            decoding_steps: list[int] | int = 1,
            temps: list[float] | float = 1.0,
            top_p: float = 0.8,
            top_k: int = 0,
            timesteps: int = 50,
            patch_size: int = 16,
            dim: int = 768,
            encoder_depth: int = 12,
            decoder_depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            mlp_bias: bool = True,
            act_layer: torch.Tensor = nn.GELU,
            norm_layer: partial | nn.Module = partial(LayerNorm, eps=1e-6),
            gated_mlp: bool = False,
            qk_norm: bool = False,
            standardize: bool = False,
            offset: dict[str, float] | None = None,
            pretraining_mean: dict[str, list] | None = None,
            pretraining_std: dict[str, list] | None = None,
    ):
        super().__init__()

        if modalities is None or len(modalities) == 0:
            raise ValueError('Input modalities not provided.')
        elif isinstance(modalities, dict):
            modalities = [modalities]
        elif not isinstance(modalities, list):
            raise ValueError(f'Modalities must be a list of modality keys or a dict with embedding layers.')

        self.output_modalities = output_modalities or ["sen2l2a@224"]
        if isinstance(decoding_steps, list):
            assert len(decoding_steps) == len(self.output_modalities), \
                "Number of decoding steps must match number of output modalities."
        else:
            decoding_steps = [decoding_steps] * len(self.output_modalities)
        if isinstance(temps, list):
            assert len(temps) == len(self.output_modalities), \
                "Number of temperatures must match number of output modalities."
        else:
            temps = [temps] * len(self.output_modalities)

        # Parameters for generation schedule
        self.top_p = top_p
        self.top_k = top_k
        self.timesteps = timesteps
        self.decoding_steps = decoding_steps
        self.temps = temps

        # Init embeddings
        self.encoder_embeddings, mod_name_mapping = build_modality_embeddings(
            modalities, img_size=img_size, dim=dim, patch_size=patch_size)
        self.decoder_embeddings, decoder_name_mapping = build_output_modality_embeddings(
            self.output_modalities, img_size=img_size, dim=dim, patch_size=patch_size)
        self.output_modalities = list(decoder_name_mapping.values())  # Update output modality names
        self.mod_name_mapping = decoder_name_mapping | mod_name_mapping  # Merging dicts
        self.modalities = list(mod_name_mapping.keys())  # Further code expects list
        self.image_modalities = [key for key, value in self.encoder_embeddings.items()
             if isinstance(value, ImageEncoderEmbedding) or isinstance(value, ImageTokenEncoderEmbedding)]
        self.output_image_modalities = [key for key, value in self.decoder_embeddings.items()
                                        if isinstance(value, ImageTokenDecoderEmbedding)]
        self.output_mod_name_mapping = {v: k for k, v in decoder_name_mapping.items()}
        self.standardize = standardize
        self.offset = offset or {}

        if offset is not None:
            for mod, o in offset.items():
                # Add offset to mean values
                assert mod in self.mod_name_mapping, f"offset {mod} not defined in input or output modalities"
                pretraining_mean[self.mod_name_mapping[mod]] = \
                    np.array(pretraining_mean[self.mod_name_mapping[mod]], dtype=np.float32) + o

        self.pretraining_mean = {mod: torch.tensor(mean)[None, :, None, None]
                                 for mod, mean in pretraining_mean.items()} if pretraining_mean else {}
        self.pretraining_std = {mod: torch.tensor(std)[None, :, None, None]
                                for mod, std in pretraining_std.items()} if pretraining_std else {}

        # Build MAE model
        mae_model = TerraMind(
            encoder_embeddings=self.encoder_embeddings,
            decoder_embeddings=self.decoder_embeddings,
            modality_info=MODALITY_INFO,
            dim=dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            mlp_bias=mlp_bias,
            act_layer=act_layer,
            norm_layer=norm_layer,
            gated_mlp=gated_mlp,
            qk_norm=qk_norm,
        )

        self.sampler = GenerationSampler(mae_model)

        self.tokenizer = build_tokenizer(input_modalities=list(self.encoder_embeddings.keys()),
                                         output_modalities=list(self.decoder_embeddings.keys()),
                                         pretrained=pretrained)

    def to(self, device):
        super().to(device)
        # Move standardization values to device
        for mod, mean in self.pretraining_mean.items():
            self.pretraining_mean[mod] = mean.to(device)
        for mod, std in self.pretraining_std.items():
            self.pretraining_std[mod] = std.to(device)
        return self

    def forward(
            self,
            d: dict[str, torch.Tensor] | torch.Tensor | None = None,
            standardize: bool | None = None,
            offset: dict[str, float] | None = None,
            timesteps: int = None,
            verbose: bool = False,
            **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            d (dict, torch.Tensor): Dict of inputs or input tensor with shape (B, C, H, W)

            Alternatively, keyword arguments with modality=tensor.

        Returns:
            dict[str, torch.Tensor]: Dict of generated images
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

        # Get batch size and device
        B = d[list(d.keys())[0]].shape[0]
        device = d[list(d.keys())[0]].device

        standardize = standardize if standardize is not None else self.standardize
        if standardize:
            for mod, value in d.items():
                d[mod] = ((value - self.pretraining_mean[self.mod_name_mapping[mod]]) /
                          self.pretraining_std[self.mod_name_mapping[mod]])

        # Define the initial input
        input_dict = {}
        # Default values if no images are provided
        img_num_tokens, image_size = 196, (224, 224)
        for mod, value in d.items():
            patch_size = self.encoder_embeddings[self.mod_name_mapping[mod]].patch_size
            num_tokens = int((value.shape[-1] / patch_size[-1]) * (value.shape[-2] / patch_size[-2]))

            if self.mod_name_mapping[mod] in self.image_modalities:
                img_num_tokens = num_tokens
                image_size = (value.shape[-2], value.shape[-1])

            # Run tokenizer encoder for tokenized input modalities
            if self.mod_name_mapping[mod] in self.tokenizer:
                if 'lulc' in self.mod_name_mapping[mod]:
                    # TODO Hack: One hot encoding for LULC classes. Generalize code.
                    if len(value.shape) == 3:
                        value = F.one_hot(value.to(int), num_classes=10).permute(0, 3, 1, 2).to(torch.float32)
                    elif len(value.shape) == 4 and value.shape[1] == 1:
                        value = F.one_hot(value.to(int).squeeze(1), num_classes=10).permute(0, 3, 1, 2).to(torch.float32)
                    elif len(value.shape) == 4 and value.shape[1] == 10:
                        # Correct shape
                        pass
                    else:
                        raise ValueError('Expect LULC data with 10 classes. '
                            'Either with class indexes and shape [B, H, W] or one hot encoded [B, 10, H, W].')

                # Tokenize
                value = self.tokenizer[self.mod_name_mapping[mod]].encode(value)[2]

            input_dict[self.mod_name_mapping[mod]] = {
                "tensor": value,
                "input_mask": torch.zeros(B, num_tokens, dtype=torch.bool, device=device),
                "target_mask": torch.ones(B, num_tokens, dtype=torch.bool, device=device),
            }

        # Initialize output modalities
        sum_tokens = 0
        for mod in self.output_modalities:
            mod_num_tokens = img_num_tokens if mod in self.output_image_modalities else 196  # TODO: Not tested for sequence data.
            sum_tokens += mod_num_tokens

            if mod in input_dict:
                warnings.warn(f'The modality {mod} is used as input and output which is not possible for the same '
                              f'patches. Random sampling 50% of tokens as input and output.')
                input_dict[mod]['input_mask'] = torch.rand((B, mod_num_tokens), device=device) < 0.5
                input_dict[mod]['target_mask'] = ~input_dict[mod]['input_mask']
                input_dict[mod]['tensor'] = input_dict[mod]['tensor'].reshape(B, -1).to(torch.long)
            else:
                input_dict[mod] = {
                    'tensor': torch.zeros((B, mod_num_tokens), dtype=torch.int64, device=device),
                    'input_mask': torch.ones((B, mod_num_tokens), dtype=torch.bool, device=device),
                    'target_mask': torch.zeros((B, mod_num_tokens), dtype=torch.bool, device=device),
                }

        # Predict tokens of output modalities
        schedule = build_chained_generation_schedules(
            cond_domains=[self.mod_name_mapping[m] for m in self.modalities],
            target_domains=self.output_modalities,
            tokens_per_target=[img_num_tokens] * len(self.output_modalities),
            autoregression_schemes=["roar"] * len(self.output_modalities),
            decoding_steps=self.decoding_steps,
            token_decoding_schedules=["linear"] * len(self.output_modalities),
            temps=self.temps,
            temp_schedules=["constant"] * len(self.output_modalities),
            cfg_scales=[1.0] * len(self.output_modalities),
            cfg_schedules=["constant"] * len(self.output_modalities),
            cfg_grow_conditioning=True,
        )

        out_dict = self.sampler.generate(
            input_dict,
            schedule,
            verbose=False,
            seed=random.randint(-(2 ** 31), 2 ** 31 - 1),
            top_p=self.top_p,
            top_k=self.top_k,
            num_tokens=sum_tokens,
        )

        # TODO Vary timesteps based on codebook diversity
        timesteps = timesteps or self.timesteps
        out = {}
        for mod in self.output_modalities:
            tok = out_dict[mod]['tensor']
            if mod in self.output_image_modalities:
                patch_size = self.tokenizer[mod].patch_size
                tok = rearrange(tok, "b (nh nw) -> b nh nw",
                                nh=image_size[0] // patch_size, nw=image_size[1] // patch_size)
            out[self.output_mod_name_mapping[mod]] = self.tokenizer[mod].decode_tokens(
                tok,
                image_size=image_size,
                timesteps=timesteps,
                verbose=verbose
            )

        if standardize:
            for mod, value in out.items():
                out[mod] = (value * self.pretraining_std[self.mod_name_mapping[mod]] +
                            self.pretraining_mean[self.mod_name_mapping[mod]])

        return out

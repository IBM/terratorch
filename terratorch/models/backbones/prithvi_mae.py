# Copyright (c) IBM Corp. 2024. All rights reserved.
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
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# transformers: https://github.com/huggingface/transformers
# --------------------------------------------------------
import logging

import torch
import torch.nn as nn
from functools import partial
from typing import List, Tuple, Union
from einops import rearrange
from timm.layers import to_2tuple
from timm.models.vision_transformer import Block


class PrithviMAE(nn.Module):
    """ Prithvi Masked Autoencoder"""

    def __init__(self,
                 img_size: int | Tuple[int] = 224,
                 patch_size: int = 16,
                 num_frames: int = 3,
                 tubelet_size: int = 1,
                 in_chans: int = 3,
                 embed_dim: int = 1024,
                 depth: int = 24,
                 num_heads: int = 16,
                 decoder_embed_dim: int = 512,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 mlp_ratio: float = 4.,
                 norm_layer: nn.Module = nn.LayerNorm,
                 norm_pix_loss: bool = False,
                 coords_encoding: List[str] | None = None,
                 coords_drop_rate: float = 0.0,
                 coords_scale_learn: bool = False,
                 drop_channels_rate: float = 0.0,
                 encoder_only: bool = False,
                 **kwargs,
                 ):
        super().__init__()

        self.encoder = PrithviViT(
            img_size=img_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            coords_encoding=coords_encoding,
            coords_drop_rate=coords_drop_rate,
            coords_scale_learn=coords_scale_learn,
            drop_channels_rate=drop_channels_rate,
        )

        self.encoder_only = encoder_only

        if not encoder_only:
            self.decoder = MAEDecoder(
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                grid_size=self.encoder.grid_size,
                in_chans=in_chans,
                encoder_embed_dim=embed_dim,
                decoder_embed_dim=decoder_embed_dim,
                depth=decoder_depth,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                coords_encoding=coords_encoding,
                coords_scale_learn=coords_scale_learn,
            )
        else:
            self.decoder = nn.Identity()

        self.norm_pix_loss = norm_pix_loss

    def patchify(self, imgs):
        """
        imgs: B, C, T, H, W
        x: B, L, D
        """
        s = self.encoder.patch_embed.tubelet_size
        p = q = self.encoder.patch_embed.patch_size
        x = rearrange(imgs, 'b c (t s) (h p) (w q) -> b (t h w) (s p q c)', s=s, p=p, q=q)

        return x

    def unpatchify(self, x, img_size=None):
        """
        x: B, L, D
        imgs: B, C, T, H, W
        """
        s = self.encoder.patch_embed.tubelet_size
        if img_size:
            img_size = to_2tuple(img_size)
            p, q = img_size
        else:
            p = q = self.encoder.patch_embed.patch_size
        gs = self.decoder.grid_size
        imgs = rearrange(x, 'b (t h w) (s p q c) -> b c (t s) (h p) (w q)',
                         h=gs[1], w=gs[2], t=gs[0], s=s, p=p, q=q)
        return imgs

    def forward_loss(self, pixel_values, pred, mask):
        """
        pixel_values: B, C, T, H, W
        target: B, L, D
        pred: B, L, D
        mask: B, L. 0 is keep, 1 is remove,
        """
        target = self.patchify(pixel_values)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self,
                pixel_values,
                temporal_coords: None | torch.Tensor = None,
                location_coords: None | torch.Tensor = None,
                mask_ratio=0.75
                ):
        latent, mask, ids_restore = self.encoder(pixel_values, temporal_coords, location_coords, mask_ratio)
        pred = self.decoder(latent, ids_restore, temporal_coords, location_coords)
        loss = self.forward_loss(pixel_values, pred, mask)
        return loss, pred, mask

    def forward_features(
        self,
        x,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
    ) -> List[torch.Tensor]:
        return self.encoder.forward_features(x, temporal_coords, location_coords)


class PrithviViT(nn.Module):
    """ Prithvi ViT Encoder"""
    def __init__(self,
                 img_size: int | Tuple[int] = 224,
                 patch_size: int = 16,
                 num_frames: int = 1,
                 tubelet_size: int = 1,
                 in_chans: int = 3,
                 embed_dim: int = 1024,
                 depth: int = 24,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.,
                 norm_layer: nn.Module = nn.LayerNorm,
                 coords_encoding: List[str] | None = None,
                 coords_drop_rate: float = 0.0,
                 coords_scale_learn: bool = False,
                 drop_channels_rate: float = 0.0,
                 encoder_only: bool = True,  # needed for timm
                 **kwargs,
                 ):
        super().__init__()

        self.in_chans = in_chans
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        img_size = to_2tuple(img_size)
        self.grid_size = (num_frames // tubelet_size, img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.drop_channels = nn.Dropout3d(drop_channels_rate) if drop_channels_rate > 0 else nn.Identity()
        self.feature_info = []
        self.encoder_only = encoder_only

        # 3D patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            num_frames=num_frames,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Optional temporal and location embedding
        coords_encoding = coords_encoding or []
        self.temporal_encoding = 'time' in coords_encoding
        self.location_encoding = 'location' in coords_encoding
        if self.temporal_encoding:
            assert tubelet_size == 1, f"With temporal encoding, tubelet_size must be 1, received {tubelet_size}"
            self.temporal_embed_enc = TemporalEncoder(embed_dim, coords_scale_learn)
            self.drop_temporal = DropPath(coords_drop_rate, scale_by_keep=False)
        if self.location_encoding:
            self.location_embed_enc = LocationEncoder(embed_dim, coords_scale_learn)
            self.drop_location = DropPath(coords_drop_rate, scale_by_keep=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer("pos_embed", torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer layers
        self.blocks = []
        for i in range(depth):
            self.blocks.append(Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
            self.feature_info.append(
                {"num_chs": embed_dim * self.patch_embed.tubelet_size, "reduction": 1, "module": f"blocks.{i}"}
            )
        self.blocks = nn.ModuleList(self.blocks)

        self.norm = norm_layer(embed_dim)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

        # pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size, cls_token=True)
        # self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def random_masking(self, x, mask_ratio):
        """
        Random token sampling and masking.
        x: [B, L, D]
        """
        B, L, D = x.shape
        num_keep = int(L * (1 - mask_ratio))

        # Sample token indexes from equal distribution
        mask = torch.ones([B, L], device=x.device)
        shuffled_ids = mask.multinomial(num_samples=L)
        ids_sampled = shuffled_ids[:, :num_keep]
        ids_restore = torch.argsort(shuffled_ids, dim=1)

        # Mask input
        x_masked = torch.gather(x, dim=1, index=ids_sampled.unsqueeze(-1).repeat(1, 1, D))
        mask[torch.arange(mask.size(0)).unsqueeze(1), ids_sampled] = 0

        return x_masked, mask, ids_restore

    def forward(self, x,
                temporal_coords: None | torch.Tensor = None,
                location_coords: None | torch.Tensor = None,
                mask_ratio=0.75
                ):
        t, h, w = x.shape[-3:]
        # Drop input channels
        x = self.drop_channels(x)

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if self.temporal_encoding:
            num_tokens_per_frame = x.shape[1] // self.num_frames
            temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
            temporal_encoding = self.drop_temporal(temporal_encoding, new_mask=True)
            x = x + temporal_encoding
        if self.location_encoding:
            location_encoding = self.location_embed_enc(location_coords)
            location_encoding = self.drop_location(location_encoding, new_mask=True)
            x = x + location_encoding

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_features(
        self,
        x,
            temporal_coords: None | torch.Tensor = None,
            location_coords: None | torch.Tensor = None,
    ) -> list[torch.Tensor]:
        if len(x.shape) == 4 and self.patch_embed.num_frames == 1:
            x = x.reshape(-1, self.in_chans, 1, *x.shape[-2:])
        t, h, w = x.shape[-3:]

        # embed patches
        x = self.patch_embed(x)
        pos_embed = get_3d_sincos_pos_embed(
                self.embed_dim,
                (
                    t // self.patch_embed.tubelet_size,
                    h // self.patch_embed.patch_size,
                    w // self.patch_embed.patch_size,
                ),
                cls_token=True,
            ).to(x)
        # add pos embed w/o cls token
        x = x + pos_embed[1:, :]

        if self.temporal_encoding:
            num_tokens_per_frame = x.shape[1] // self.patch_embed.num_frames
            temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
            x = x + temporal_encoding
        if self.location_encoding:
            location_encoding = self.location_embed_enc(location_coords)
            x = x + location_encoding

        # append cls token
        cls_token = self.cls_token + pos_embed[:1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        out = []
        for block in self.blocks:
            x = block(x)
            out.append(x.clone())

        x = self.norm(x)
        out[-1] = x
        return out


class MAEDecoder(nn.Module):
    """ ViT Decoder used in the Prithvi MAE"""
    def __init__(self,
                 patch_size: int = 16,
                 tubelet_size: int = 1,
                 grid_size: Union[List[int], Tuple[int, int, int]] = (3, 14, 14),
                 in_chans: int = 3,
                 encoder_embed_dim: int = 1024,
                 decoder_embed_dim: int = 512,
                 depth: int = 8,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.,
                 norm_layer: nn.Module = nn.LayerNorm,
                 coords_encoding: List[str] | None = None,
                 coords_scale_learn: bool = False,
                 ):
        super().__init__()

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.grid_size = grid_size
        num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # Optional temporal and location embedding
        coords_encoding = coords_encoding or []
        self.temporal_encoding = 'time' in coords_encoding
        self.location_encoding = 'location' in coords_encoding
        if self.temporal_encoding:
            self.temporal_embed_dec = TemporalEncoder(decoder_embed_dim, coords_scale_learn)
        if self.location_encoding:
            self.location_embed_dec = LocationEncoder(decoder_embed_dim, coords_scale_learn)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.register_buffer("decoder_pos_embed", torch.zeros(1, num_patches + 1, decoder_embed_dim))

        self.decoder_blocks = []
        for i in range(depth):
            self.decoder_blocks.append(
                Block(decoder_embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim,
                                      tubelet_size * patch_size * patch_size * in_chans,
                                      bias=True)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.unsqueeze(0))


    def forward(self, hidden_states: torch.Tensor,
                ids_restore: torch.Tensor,
                temporal_coords: None | torch.Tensor = None,
                location_coords: None | torch.Tensor = None,
                ):

        # embed tokens
        x = self.decoder_embed(hidden_states)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # unshuffle
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x_.device))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # add pos embed
        x = x + self.decoder_pos_embed
        # remove cls token
        x_ = x[:, 1:, :]

        if self.temporal_encoding:
            num_tokens_per_frame = x.shape[1] // self.patch_embed.input_size[0]
            temporal_encoding = self.temporal_embed_dec(temporal_coords, num_tokens_per_frame)
            # Reuse drop mask from encoder for consistent dropping
            temporal_encoding = self.drop_temporal(temporal_encoding, new_mask=False)
            # Add temporal encoding w/o cls token
            x_ = x_ + temporal_encoding
        if self.location_encoding:
            location_encoding = self.location_embed_dec(location_coords)
            # Reuse drop mask from encoder for consistent dropping
            location_encoding = self.drop_location(location_encoding, new_mask=False)
            # Add location encoding w/o cls token
            x_ = x_ + location_encoding

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # apply Transformer blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.norm(x)

        # predictor projection
        pred = self.pred(x)

        # remove cls token
        pred = pred[:, 1:, :]

        return pred


class PatchEmbed(nn.Module):
    """3D version of timm.models.vision_transformer.PatchEmbed"""
    def __init__(
        self,
        patch_size=16,
        num_frames=3,
        tubelet_size=1,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,  # noqa: FBT002
        bias=True,  # noqa: FBT002
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
        self.in_chans = in_chans
        self.flatten = flatten

        if num_frames % tubelet_size != 0:
            msg = f"num_frames ({num_frames} must be divisible by tubelet size ({tubelet_size}))"
            raise Exception(msg)

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if len(x.shape) == 4 and self.num_frames == 1:
            x = x.reshape(-1, self.in_chans, self.num_frames, *x.shape[-2:])

        if x.shape[-1] / self.patch_size % 1 or x.shape[-2] / self.patch_size % 1:
            logging.warning(f"Input {x.shape[-2:]} is not divisible by patch size {self.patch_size}."
                            f"The border will be ignored, add backbone_padding for pixel-wise tasks.")

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x


class DropPath(nn.Module):
    """ Adapted from timm.models.layers.DropPath. In this version, drop mask can be saved and reused.
        This is useful when applying the same drop mask more than once.
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.drop_mask = None

    def generate_mask(self, x: torch.Tensor):
        self.drop_mask = generate_mask(x, self.drop_prob, self.scale_by_keep)

    def forward(self, x: torch.Tensor, new_mask: bool = True):

        if self.drop_prob == 0. or not self.training:
            return x

        if self.drop_mask is None or new_mask:
            self.generate_mask(x)

        return self.drop_mask * x

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.year_embed_dim = embed_dim // 2
        self.julian_day_embed_dim = embed_dim - self.year_embed_dim

        # If trainable, initialize scale with small number
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, temporal_coords: torch.Tensor, tokens_per_frame: int | None = None):
        """
        temporal_coords: year and day-of-year info with shape (B, T, 2).
        tokens_per_frame: number of tokens for each frame in the sample. If provided, embeddings will be
            repeated over T dimension, and final shape is (B, T*tokens_per_frame, embed_dim).
        """
        shape = temporal_coords.shape[:2] + (-1,)  # B, T, -1

        year = get_1d_sincos_pos_embed(self.year_embed_dim, temporal_coords[:, :, 0].flatten()).reshape(shape)
        julian_day = get_1d_sincos_pos_embed(
            self.julian_day_embed_dim, temporal_coords[:, :, 1].flatten()).reshape(shape)

        embedding = self.scale * torch.cat([year, julian_day], dim=-1)

        if tokens_per_frame is not None:
            embedding = torch.repeat_interleave(embedding, tokens_per_frame, dim=1)

        return embedding  # B, T*tokens_per_frame, embed_dim


class LocationEncoder(nn.Module):
    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.lat_embed_dim = embed_dim // 2
        self.lon_embed_dim = embed_dim - self.lat_embed_dim

        # If trainable, initialize scale with small number
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, location_coords: torch.Tensor):
        """
        location_coords: lat and lon info with shape (B, 2).
        """
        shape = location_coords.shape[:1] + (1, -1)  # B, 1, -1

        lat = get_1d_sincos_pos_embed(self.lat_embed_dim, location_coords[:, 0].flatten()).reshape(shape)
        lon = get_1d_sincos_pos_embed(self.lat_embed_dim, location_coords[:, 1].flatten()).reshape(shape)

        embedding = self.scale * torch.cat([lat, lon], dim=-1)

        return embedding  # B, 1, embed_dim


def generate_mask(x, drop_prob: float = 0., scale_by_keep: bool = True):
    """ Create drop mask for x. Adapted from timm.models.layers.drop_path. """
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return random_tensor


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: 3d tuple of grid size: t, h, w
    return:
    pos_embed: L, D
    """

    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed(w_embed_dim, torch.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed(h_embed_dim, torch.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed(t_embed_dim, torch.arange(t_size))

    w_pos_embed = w_pos_embed.repeat(t_size * h_size, 1)
    h_pos_embed = h_pos_embed.repeat_interleave(w_size, dim=0).repeat((t_size, 1))
    t_pos_embed = t_pos_embed.repeat_interleave(h_size * w_size, dim=0)

    pos_embed = torch.concat((w_pos_embed, h_pos_embed, t_pos_embed), dim=1)

    if cls_token:
        pos_embed = torch.concat([torch.zeros([1, embed_dim]), pos_embed])

    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: 2d tuple of grid size: h, w
    return:
    pos_embed: L, D
    """

    assert embed_dim % 2 == 0
    h_size, w_size = grid_size

    w_embed_dim = embed_dim // 2
    h_embed_dim = embed_dim // 2

    w_pos_embed = get_1d_sincos_pos_embed(w_embed_dim, torch.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed(h_embed_dim, torch.arange(h_size))

    w_pos_embed = w_pos_embed.repeat(h_size, 1)
    h_pos_embed = h_pos_embed.repeat_interleave(w_size, dim=0)

    pos_embed = torch.concat((w_pos_embed, h_pos_embed), dim=1)

    if cls_token:
        pos_embed = torch.concat([torch.zeros([1, embed_dim]), pos_embed])

    return pos_embed


def freq_bands(num_bands, temperature=10000., step=2, device=None) -> torch.Tensor:
    exp = torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(torch.float32) / num_bands
    bands = 1. / (temperature ** exp)
    return bands


def get_1d_sincos_pos_embed(embed_dim, pos, cls_token=False):
    bands = freq_bands(embed_dim, step=2)
    embed_pos = torch.einsum('b,p -> bp', pos, bands)

    emb_sin = embed_pos.sin()
    emb_cos = embed_pos.cos()

    emb = torch.concat([emb_sin, emb_cos], dim=1)

    if cls_token:
        emb = torch.concat([torch.zeros([1, embed_dim]), emb], dim=0)

    return emb


def _init_weights(module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

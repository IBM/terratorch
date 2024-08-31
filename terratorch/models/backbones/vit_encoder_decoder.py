# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

# Copyright contributors to the Terratorch project


from functools import lru_cache

import numpy as np
import torch
from einops import rearrange
from timm.layers import to_2tuple
from timm.models.vision_transformer import Block
from torch import Tensor, nn

B_C_H_W_SHAPE_LEN = 4


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    if embed_dim % 2 != 0:
        msg = "Embed dim must be divisible by 2"
        raise Exception(msg)
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


@lru_cache(maxsize=100, typed=False)
def get_3d_sincos_pos_embed(embed_dim: int, grid_size: tuple[int, int, int], cls_token: bool = False):
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    # All rights reserved.

    # This source code is licensed under the license found in the
    # LICENSE file in the root directory of this source tree.
    # --------------------------------------------------------
    # Position embedding utils
    # --------------------------------------------------------
    """
    grid_size: 3d tuple of grid size: t, h, w
    return:
    pos_embed: L, D
    """

    if embed_dim % 16 != 0:
        msg = "Embed dim must be divisible by 16"
        raise Exception(msg)

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PatchEmbed(nn.Module):
    """Frames of 2D Images to Patch Embedding
    The 3D version of timm.models.vision_transformer.PatchEmbed
    """

    def __init__(
        self,
        img_size=224,
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
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1]),
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if len(x.shape) == B_C_H_W_SHAPE_LEN and self.num_frames == 1:
            x = x.reshape(-1, self.in_chans, self.num_frames, *x.shape[-2:])
        B, C, T, H, W = x.shape  # noqa: N806
        x = self.proj(x)
        # Hp, Wp = x.shape[3], x.shape[4]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x


class TemporalViTEncoder(nn.Module):
    """Encoder from an ViT with capability to take in temporal input.

    This class defines an encoder taken from a ViT architecture.
    """

    # see https://github.com/huggingface/pytorch-image-models/blob/d5f1525334e1b111e4bfdf59fcd38eb9f8c9d3de/timm/models/vision_transformer.py#L407C15-L407C15
    def __init__(
        self,
        pretrain_img_size: int = 224,
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
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        norm_pix_loss: bool = False,  # noqa: FBT001, FBT002
        encoder_only: bool = True,  # noqa: FBT001, FBT002
        **kwargs,  # timm parameters that may be passed  # noqa: ARG002
    ):
        """
        Args:
            pretrain_img_size (int, optional): Input image size for SSL. Ignored for forward_feature. Defaults to 224.
            patch_size (int, optional): Patch size to be used by the transformer. Defaults to 16.
            num_frames (int, optional): Number of frames (temporal dimension) to be input to the encoder. Defaults to 1.
            tubelet_size (int, optional): Tubelet size used in patch embedding. Defaults to 1.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            embed_dim (int, optional): Embedding dimension. Defaults to 1024.
            depth (int, optional): Encoder depth. Defaults to 24.
            num_heads (int, optional): Number of heads used in the encoder blocks. Defaults to 16.
            mlp_ratio (float, optional): Ratio to be used for the size of the MLP in encoder blocks. Defaults to 4.0.
            norm_layer (nn.Module, optional): Norm layer to be used. Defaults to nn.LayerNorm.
            norm_pix_loss (bool, optional): Whether to use Norm Pix Loss. Defaults to False.
            pretrained (str, optional): Path to pretrained encoder weights. Defaults to None.
        """
        super().__init__()
        print(f"Encoder type:{encoder_only}")
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(pretrain_img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim)
        self.feature_info = []
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoder_only = encoder_only
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(self.embed_dim, self.patch_embed.grid_size, cls_token=True)
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(pos_embed).float().unsqueeze(0),
            persistent=False,
        )

        self.blocks = [
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)
        ]
        for i, _ in enumerate(self.blocks):
            self.feature_info.append({"num_chs": embed_dim * num_frames, "reduction": 1, "module": f"blocks.{i}"})
        self.blocks = nn.ModuleList(self.blocks)
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        if not encoder_only:
            self.decoder_embed_dim = decoder_embed_dim
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            # self.decoder_pos_embed = torch.zeros(
            #     1, num_patches + 1, decoder_embed_dim
            # )  # fixed sin-cos embedding

            self.decoder_blocks = nn.ModuleList(
                [
                    Block(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )
                    for i in range(decoder_depth)
                ]
            )

            # decoder_pos_embed = get_3d_sincos_pos_embed(
            #     self.decoder_embed_dim, self.patch_embed.grid_size, cls_token=True
            # )
            # self.register_buffer(
            #     "decoder_pos_embed",
            #     torch.from_numpy(decoder_pos_embed).float().unsqueeze(0),
            #     persistent=False,
            # )

            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(
                decoder_embed_dim,
                tubelet_size * patch_size * patch_size * in_chans,
                bias=True,
            )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        self.embed_dims = [self.embed_dim] * len(self.blocks)

    def initialize_weights(self):
        # initialization

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        if not self.encoder_only:
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: B, C, T, H, W
        x: B, L, D
        """
        p = self.patch_embed.patch_size[0]
        tub = self.patch_embed.tubelet_size
        x = rearrange(imgs, "b c (t tub) (h p) (w q) -> b (t h w) (tub p q c)", tub=tub, p=p, q=p)

        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B, L, D
        imgs: B, C, T, H, W
        """
        p = self.patch_embed.patch_size[0]
        num_p = self.patch_embed.img_size[0] // p
        tub = self.patch_embed.tubelet_size
        imgs = rearrange(
            x,
            "b (t h w) (tub p q c) -> b c (t tub) (h p) (w q)",
            h=num_p,
            w=num_p,
            tub=tub,
            p=p,
            q=p,
        )
        return imgs

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim  # noqa: N806
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # embed patches
        if len(x.shape) == B_C_H_W_SHAPE_LEN and self.patch_embed.num_frames == 1:
            x = x.reshape(-1, self.in_chans, 1, *x.shape[-2:])
        t, h, w = x.shape[-3:]
        x = self.patch_embed(x)
        pos_embed = torch.from_numpy(get_3d_sincos_pos_embed(self.embed_dim, (t, h // 16, w // 16), cls_token=True)).to(
            x
        )
        # add pos embed w/o cls token
        x = x + pos_embed[1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + pos_embed[:1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, (t, h, w)

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor, dim_info: tuple) -> torch.Tensor:
        if self.encoder_only:
            msg = (
                "Cannot run forward decoder method with self.encoder_only. Pass encoder_only=False to use the decoder."
            )
            raise Exception(msg)
        # embed tokens
        x = self.decoder_embed(x)
        t, h, w = dim_info
        decoder_pos_embed = torch.from_numpy(
            get_3d_sincos_pos_embed(self.decoder_embed_dim, (t, h // 16, w // 16), cls_token=True)
        ).to(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask) -> torch.Tensor:
        """
        imgs: B, C, T, H, W
        target: B, L, D
        pred: B, L, D
        mask: B, L. 0 is keep, 1 is remove,
        """
        if len(imgs.shape) == B_C_H_W_SHAPE_LEN and self.num_frames == 1:
            imgs = imgs.reshape(-1, self.in_chans, self.num_frames, *imgs.shape[-2:])
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.0) -> torch.Tensor:
        latent, mask, ids_restore, dim_info = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore, dim_info)
        _ = self.forward_loss(imgs, pred, mask)
        return pred

    def forward_features(self, x) -> list[torch.Tensor]:
        if len(x.shape) == B_C_H_W_SHAPE_LEN and self.patch_embed.num_frames == 1:
            x = x.reshape(-1, self.in_chans, 1, *x.shape[-2:])
        t, h, w = x.shape[-3:]
        # embed patches
        x = self.patch_embed(x)
        pos_embed = torch.from_numpy(get_3d_sincos_pos_embed(self.embed_dim, (t, h // 16, w // 16), cls_token=True)).to(
            x
        )
        # add pos embed w/o cls token
        x = x + pos_embed[1:, :]

        # append cls token
        cls_token = self.cls_token + pos_embed[:1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        out = []
        for blk in self.blocks:
            x = blk(x)
            out.append(x.clone())

        x = self.norm(x)
        out[-1] = x
        return out

    def prepare_features_for_image_model(self, features: list[Tensor]) -> list[Tensor]:
        out = []
        for x in features:
            x_no_token = x[:, 1:, :]
            encoded = rearrange(
                x_no_token,
                "batch (t h w) e -> batch (t e) h w",
                e=self.embed_dim,
                t=self.num_frames,
                h=int(np.sqrt(x_no_token.shape[1] // self.num_frames)),
            )
            out.append(encoded)
        return out

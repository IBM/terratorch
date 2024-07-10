from typing import Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# DropPath code is straight from timm
# (https://huggingface.co/spaces/Roll20/pet_score/blame/main/lib/timm/models/layers/drop.py)
# Primarily since we currently don't have timm in the environment.
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Module):
    '''
    Multi layer perceptron.
    '''

    def __init__(self, features: int, hidden_features: int, dropout: float = 0.0) -> None:
        '''
        Args:
            features: Input/output dimension.
            hidden_features: Hidden dimension.
            dropout: Dropout.
        '''
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            Tensor of shape [..., channel]
        Returns:
            Tensor of same shape as x.
        '''
        return self.net(x)


class MultiheadAttention(nn.Module):
    '''
    Multihead attention layer for inputs of shape [..., sequence, features].

    Uses `scaled_dot_product_attention` to obtain a memory efficient attention computation (https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). This follows:
    - Dao et la. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (https://arxiv.org/abs/2205.14135)
    - Rabe, Staats "Self-attention Does Not Need O(n2) Memory" (https://arxiv.org/abs/2112.05682)

    Note: Even though the documentation page for `scaled_dot_product_attention` states that tensors can have any number of dimensions as long as the shapes are along the lines of `(B, ..., S, E)`, the fused and memory efficient mechanisms we enforce here require a 4D input. Some experimentatino shows that this should be of shape `(B, H, S, E)`, where `H` represents heads. However, as of right now this is not confirmed int he documentation.
    '''

    def __init__(self, features: int, n_heads: int, dropout: float) -> None:
        '''
        Args:
            features: Number of features for inputs to the layer.
            n_heads: Number of attention heads. Should be a factor of features. (I.e. the layer uses features // n_heads.)
            dropout: Dropout.
        '''
        super().__init__()

        if not (features % n_heads) == 0:
            raise ValueError(
                f'Number of features {features} is not divisible by number of heads {n_heads}.'
            )

        self.features = features
        self.n_heads = n_heads
        self.dropout = dropout

        self.qkv_layer = torch.nn.Linear(features, features * 3, bias=False)
        self.w_layer = torch.nn.Linear(features, features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: Tensor of shape [..., sequence, features]
        Returns:
            Tensor of shape [..., sequence, features]
        '''
        if not x.shape[-1] == self.features:
            raise ValueError(f'Expecting tensor with last dimension of size {self.features}.')

        passenger_dims = x.shape[:-2]
        B = np.prod(passenger_dims)
        S = x.shape[-2]
        C = x.shape[-1]
        x = x.reshape(B, S, C)

        # x [B, S, C]
        # q, k, v [B, H, S, C/H]
        q, k, v = (
            self.qkv_layer(x)
            .view(B, S, self.n_heads, 3 * (C // self.n_heads))
            .transpose(1, 2)
            .chunk(chunks=3, dim=3)
        )

        # Let us enforce either flash (A100+) or memory efficient attention.
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=True
        ):
            # x [B, H, S, C//H]
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        # x [B, S, C]
        x = x.transpose(1, 2).reshape(B, S, C)
        # x [B, S, C]
        x = self.w_layer(x)
        # Back to input shape
        x = x.reshape(*passenger_dims, S, self.features)

        with torch.no_grad():
            scale_factor = 1 / math.sqrt(q.size(-1))
            attn_weight = q @ k.transpose(-2, -1) * scale_factor
            attn_weight = torch.softmax(attn_weight, dim=-1).mean(dim=-3).detach()

        return x, attn_weight


class Transformer(nn.Module):
    '''
    Transformer for inputs of shape [..., S, features].
    '''

    def __init__(self, features: int, mlp_multiplier: int, n_heads: int, dropout: float, drop_path: float) -> None:
        '''
        Args:
            features: Number of features for inputs to the layer.
            mlp_multiplier: Model will use features*mlp_multiplier hidden units.
            n_heads: Number of attention heads. Should be a factor of features. (I.e. the layer uses features // n_heads.)
            dropout: Dropout.
            drop_path: DropPath.
        '''
        super().__init__()

        self.features = features
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attention = nn.Sequential(
            nn.LayerNorm(features),
            MultiheadAttention(features, n_heads, dropout),
        )

        self.ff = nn.Sequential(
            nn.LayerNorm(features),
            Mlp(features=features, hidden_features=features * mlp_multiplier, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: Tensor of shape [..., sequence, features]
        Returns:
            Tensor of shape [..., sequence, features]
        '''
        if not x.shape[-1] == self.features:
            raise ValueError(f'Expecting tensor with last dimension of size {self.features}.')

        attention_x, attn_weight = self.attention(x)

        x = x + self.drop_path(attention_x)
        x = x + self.drop_path(self.ff(x))

        return x, attn_weight


class LocalGlobalLocalBlock(nn.Module):
    '''
    Applies alternating block and grid attention. Given a parameter n_blocks, the entire
    module contains 2*n_blocks+1 transformer blocks. The first, third, ..., last apply
    local (block) attention. The second, fourth, ... global (grid) attention.

    This is heavily inspired by Tu et al. "MaxViT: Multi-Axis Vision Transformer"
    (https://arxiv.org/abs/2204.01697).
    '''

    def __init__(
        self, features: int, mlp_multiplier: int, n_heads: int, dropout: float, drop_path: float, n_blocks: int
    ) -> None:
        '''
        Args:
            features: Number of features for inputs to the layer.
            mlp_multiplier: Model will use features*mlp_multiplier hidden units.
            n_heads: Number of attention heads. Should be a factor of features.
            (I.e. the layer uses features // n_heads.)
            dropout: Dropout.
            drop_path: DropPath.
            n_blocks: Number of local-global transformer pairs.
        '''
        super().__init__()

        self.features = features
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout
        self.drop_path = drop_path
        self.n_blocks = n_blocks

        self.transformers = nn.ModuleList(
            [
                Transformer(
                    features=features,
                    mlp_multiplier=mlp_multiplier,
                    n_heads=n_heads,
                    dropout=dropout,
                    drop_path=drop_path,
                )
                for _ in range(2 * n_blocks + 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: Tensor of shape [batch, global_sequence, local_sequence, features]
        Returns:
            Tensor of shape [batch, global_sequence, local_sequence, features]
        '''
        if not x.shape[-1] == self.features:
            raise ValueError(f'Expecting tensor with last dimension of size {self.features}.')
        if len(x.shape) != 4:
            raise ValueError(
                f'Expecting tensor with exactly four dimensions. Input has shape {x.shape}.'
            )

        attn_weights = {'local' : [], 'global' : []}
        layer_deltas = []
        for i, transformer in enumerate(self.transformers):
            if i > 0:
                # We are making exactly 2*n_blocks transposes.
                # So the output has the same shape as input.
                x = x.transpose(1, 2)

            x_in = x

            x, attn_weight = transformer(x)

            if i % 2 == 0:
                attn_weights['local'].append(attn_weight)                
            else:
                attn_weights['global'].append(attn_weight)

            with torch.no_grad():
                layer_delta = torch.sqrt(torch.mean((x - x_in)**2, dim=(1, 2, 3)))
                layer_deltas.append(layer_delta.detach())

        return x, attn_weights, layer_deltas


class PatchEmbed(nn.Module):
    '''
    Patch embedding via 2D convolution.
    '''

    def __init__(self, patch_size: int | tuple[int, ...], channels: int, embed_dim: int):
        super().__init__()

        self.patch_size = patch_size
        self.channels = channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: Tensor of shape [batch, channels, lat, lon].
        Returns:
            Tensor with shape [batch, embed_dim, lat//patch_size, lon//patch_size]
        '''

        B, C, H, W = x.size()

        if W % self.patch_size[1] != 0:
            raise ValueError(
                f'Cannot do patch embedding for tensor of shape {x.size()} with patch size {self.patch_size}. (Dimensions are BSCHW.)'
            )
        if H % self.patch_size[0] != 0:
            raise ValueError(
                f'Cannot do patch embedding for tensor of shape {x.size()} with patch size {self.patch_size}. (Dimensions are BSCHW.)'
            )

        x = self.proj(x)

        return x


class HieraMaxViTEncoderDecoder(nn.Module):
    '''
    Hiera-MaxViT encoder/decoder code.
    '''

    def __init__(
        self,
        embed_dim: int,
        n_blocks: int,
        mlp_multiplier: float,
        n_heads: int,
        dropout: float,
        drop_path: float,
    ) -> None:
        '''
        Args:
            embed_dim: Embedding dimension
            n_blocks: Number of local-global transformer pairs.
            mlp_multiplier: MLP multiplier for hidden features in feed forward networks.
            n_heads: Number of attention heads.
            dropout: Dropout.
            drop_path: DropPath.
        '''
        super().__init__()

        self.embed_dim = embed_dim
        self.n_blocks = n_blocks
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout

        self.lead_time_embedding = nn.Linear(1, embed_dim, bias=True)

        self.lgl_block = LocalGlobalLocalBlock(
            features=embed_dim,
            mlp_multiplier=mlp_multiplier,
            n_heads=n_heads,
            dropout=dropout,
            drop_path=drop_path,
            n_blocks=n_blocks,
        )

    def forward(
        self,
        x: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        Args:
            x: Tensor of shape [batch, global sequence, local sequence, embed_dim]
            lead_time: Tensor of shape [batch].
        Returns:
            Tensor of shape [batch, mask_unit_sequence, local_sequence, embed_dim].
            Identical in shape to the input x.
        '''
        if lead_time is not None:
            # lead_time_embedded [batch, 1, 1, embed_dim]
            lead_time_embedded = self.lead_time_embedding(lead_time.reshape(-1, 1, 1, 1))
            lead_time_embedded = lead_time_embedded.repeat(1, x.shape[1], 1, 1)
            x = torch.cat((lead_time_embedded, x), dim=2)

        x, attn_weights, layer_deltas = self.lgl_block(x)

        if lead_time is not None:
            x = x[:, :, 1:, :]

        return x, attn_weights, layer_deltas


class HieraMaxViT(nn.Module):
    '''
    Encoder-decoder fusing Hiera with MaxViT. See
    - Ryali et al. "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles" (https://arxiv.org/abs/2306.00989)
    - Tu et al. "MaxViT: Multi-Axis Vision Transformer" (https://arxiv.org/abs/2204.01697)
    '''

    def __init__(
        self,
        in_channels: int,
        input_size_time: int,
        in_channels_static: int,
        input_scalers_mu: torch.Tensor,
        input_scalers_sigma: torch.Tensor,
        input_scalers_epsilon: float,
        static_input_scalers_mu: torch.Tensor,
        static_input_scalers_sigma: torch.Tensor,
        static_input_scalers_epsilon: float,
        output_scalers: torch.Tensor,
        n_lats_px: int,
        n_lons_px: int,
        patch_size_px: tuple[int],
        mask_unit_size_px: tuple[int],
        mask_ratio_inputs: float,
        mask_ratio_targets: float,
        embed_dim: int,
        n_blocks_encoder: int,
        n_blocks_decoder: int,
        mlp_multiplier: float,
        n_heads: int,
        dropout: float,
        drop_path: float,
        residual: str,
        masking_mode: str,
    ) -> None:
        '''
        Args:
            in_channels: Number of input channels.
            input_size_time: Number of timestamps in input.
            in_channels_static: Number of input channels for static data.
            input_scalers_mu: Tensor of size (in_channels,). Used to rescale input.
            input_scalers_sigma: Tensor of size (in_channels,). Used to rescale input.
            input_scalers_epsilon: Float. Used to rescale input.
            static_input_scalers_mu: Tensor of size (in_channels_static). Used to rescale static inputs.
            static_input_scalers_sigma: Tensor of size (in_channels_static). Used to rescale static inputs.
            static_input_scalers_epsilon: Float. Used to rescale static inputs.
            output_scalers: Tensor of shape (in_channels,). Used to rescale output.
            n_lats_px: Total latitudes in data. In pixels.
            n_lons_px: Total longitudes in data. In pixels.
            patch_size_px: Patch size for tokenization. In pixels lat/lon.
            mask_unit_size_px: Size of each mask unit. In pixels lat/lon.
            mask_ratio_inputs: Masking ratio for inputs. 0 to 1.
            mask_ratio_targets: Masking ratio for targets. 0 to 1.
            embed_dim: Embedding dimension
            n_blocks_encoder: Number of local-global transformer pairs in encoder.
            n_blocks_decoder: Number of local-global transformer pairs in decoder.
            mlp_multiplier: MLP multiplier for hidden features in feed forward networks.
            n_heads: Number of attention heads.
            dropout: Dropout.
            drop_path: DropPath.
            residual: Indicates whether and how model should work as residual model.
                Accepted values are 'climate', 'temporal' and 'none'
        '''
        super().__init__()

        if mask_ratio_targets > 0.0:
            raise NotImplementedError('Target masking is not implemented.')

        self.in_channels = in_channels
        self.input_size_time = input_size_time
        self.in_channels_static = in_channels_static
        self.n_lats_px = n_lats_px
        self.n_lons_px = n_lons_px
        self.patch_size_px = patch_size_px
        self.mask_unit_size_px = mask_unit_size_px
        self.mask_ratio_inputs = mask_ratio_inputs
        self.mask_ratio_targets = mask_ratio_targets
        self.embed_dim = embed_dim
        self.n_blocks_encoder = n_blocks_encoder
        self.n_blocks_decoder = n_blocks_decoder
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout
        self.drop_path = drop_path
        self.residual = residual

        assert self.n_lats_px % self.mask_unit_size_px[0] == 0
        assert self.n_lons_px % self.mask_unit_size_px[1] == 0
        assert self.mask_unit_size_px[0] % self.patch_size_px[0] == 0
        assert self.mask_unit_size_px[1] % self.patch_size_px[1] == 0
        self.local_shape_mu = (
            self.mask_unit_size_px[0] // self.patch_size_px[0],
            self.mask_unit_size_px[1] // self.patch_size_px[1],
        )
        self.global_shape_mu = (
            self.n_lats_px // self.mask_unit_size_px[0],
            self.n_lons_px // self.mask_unit_size_px[1],
        )

        assert input_scalers_mu.shape == (in_channels,)
        assert input_scalers_sigma.shape == (in_channels,)
        assert static_input_scalers_mu.shape == (in_channels_static,)
        assert static_input_scalers_sigma.shape == (in_channels_static,)
        assert output_scalers.shape == (in_channels,)
        # Input shape [batch, time, parameter, lat, lon]
        self.input_scalers_mu = nn.Parameter(
            input_scalers_mu.reshape(1, 1, -1, 1, 1), requires_grad=False
        )
        self.input_scalers_sigma = nn.Parameter(
            input_scalers_sigma.reshape(1, 1, -1, 1, 1), requires_grad=False
        )
        self.input_scalers_epsilon = input_scalers_epsilon
        # Static inputs shape [batch, parameter, lat, lon]
        self.static_input_scalers_mu = nn.Parameter(
            static_input_scalers_mu.reshape(1, -1, 1, 1), requires_grad=False
        )
        self.static_input_scalers_sigma = nn.Parameter(
            static_input_scalers_sigma.reshape(1, -1, 1, 1), requires_grad=False
        )
        self.static_input_scalers_epsilon = static_input_scalers_epsilon
        # Output shape [batch, parameter, lat, lon]
        self.output_scalers = nn.Parameter(output_scalers.reshape(1, -1, 1, 1), requires_grad=False)

        self.patch_embedding = PatchEmbed(
            patch_size=patch_size_px, channels=in_channels * input_size_time, embed_dim=embed_dim
        )

        if self.residual == 'climate':
            self.patch_embedding_static = PatchEmbed(
                patch_size=patch_size_px,
                channels=in_channels + in_channels_static,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embedding_static = PatchEmbed(
                patch_size=patch_size_px, channels=in_channels_static, embed_dim=embed_dim
            )

        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, self.embed_dim))
        self._nglobal_mu = np.prod(self.global_shape_mu)
        self._global_idx = torch.arange(self._nglobal_mu)

        self._nlocal_mu = np.prod(self.local_shape_mu)
        self._local_idx = torch.arange(self._nlocal_mu)

        self.encoder = HieraMaxViTEncoderDecoder(
            embed_dim=embed_dim,
            n_blocks=n_blocks_encoder,
            mlp_multiplier=mlp_multiplier,
            n_heads=n_heads,
            dropout=dropout,
            drop_path=drop_path,
        )

        if n_blocks_decoder == 0:
            return

        self.decoder = HieraMaxViTEncoderDecoder(
            embed_dim=embed_dim,
            n_blocks=n_blocks_decoder,
            mlp_multiplier=mlp_multiplier,
            n_heads=n_heads,
            dropout=dropout,
            drop_path=drop_path,
        )

        self.unembed = nn.Linear(
            self.embed_dim,
            self.in_channels * self.patch_size_px[0] * self.patch_size_px[1],
            bias=True,
        )

        self.masking_mode = masking_mode.lower()
        match self.masking_mode:
            case "local":
                self.generate_mask = self._gen_mask_local
            case "global":
                self.generate_mask = self._gen_mask_global
            case "both":
                self._mask_both_local: bool = True
                self.generate_mask = self._gen_mask_both
            case _:
                raise ValueError(f"Masking mode '{masking_mode}' not supported")

    def swap_masking(self) -> None:
        self._mask_both_local = not self._mask_both_local

    @property
    def n_masked_global(self):
        return int(self.mask_ratio_inputs * np.prod(self.global_shape_mu))

    @property
    def n_masked_local(self):
        return int(self.mask_ratio_inputs * np.prod(self.local_shape_mu))

    @staticmethod
    def _shuffle_along_axis(a, axis):
        # https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
        idx = torch.argsort(input=torch.rand(*a.shape), dim=axis)
        return torch.gather(a, dim=axis, index=idx)

    def _gen_mask_local(self, sizes: tuple[int]) -> tuple[torch.Tensor]:
        '''
        Args:
            batch_size: Number of elements in batch
        Returns:
            Tuple of torch tensors. [indices masked, indices unmasked].
            Each of these is a tensor of shape (batch, global sequene)
        '''
        # We identifies which indices (values) should be masked

        maskable_indices = self._local_idx.view(1, -1).expand(*sizes[:2], -1)

        maskable_indices = self._shuffle_along_axis(maskable_indices, 2)

        # `...` cannot be jit'd :-(
        indices_masked = maskable_indices[:,:, :self.n_masked_local]
        indices_unmasked = maskable_indices[:,:, self.n_masked_local:]

        return indices_masked, indices_unmasked

    def _gen_mask_global(self, sizes: tuple[int]) -> tuple[torch.Tensor]:
        '''
        Args:
            batch_size: Number of elements in batch
        Returns:
            Tuple of torch tensors. [indices masked, indices unmasked].
            Each of these is a tensor of shape (batch, global sequene)
        '''
        # We identifies which indices (values) should be masked

        maskable_indices = self._global_idx.view(1, -1).expand(*sizes[:1], -1)

        maskable_indices = self._shuffle_along_axis(maskable_indices, 1)

        indices_masked = maskable_indices[:, : self.n_masked_global]
        indices_unmasked = maskable_indices[:, self.n_masked_global:]

        return indices_masked, indices_unmasked

    def _gen_mask_both(self, sizes: tuple[int]) -> tuple[torch.Tensor]:
        if self._mask_both_local:
            return self._gen_mask_local(sizes)
        else:
            return self._gen_mask_global(sizes)

    @staticmethod
    def reconstruct_batch(
        idx_masked: torch.Tensor,
        idx_unmasked: torch.Tensor,
        data_masked: torch.Tensor,
        data_unmasked: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Reconstructs a tensor along the mask unit dimension. Batched version.

        Args:
            idx_masked: Tensor of shape `batch, mask unit sequence`.
            idx_unmasked: Tensor of shape `batch, mask unit sequence`.
            data_masked: Tensor of shape `batch, mask unit sequence, ...`. Should have same size along mask unit sequence dimension as idx_masked. Dimensions beyond the first two, marked here as ... will typically be `local_sequence, channel` or `channel, lat, lon`. These dimensions should agree with data_unmasked.
            data_unmasked: Tensor of shape `batch, mask unit sequence, ...`. Should have same size along mask unit sequence dimension as idx_unmasked. Dimensions beyond the first two, marked here as ... will typically be `local_sequence, channel` or `channel, lat, lon`. These dimensions should agree with data_masked.
        Returns:
            Tensor of same shape as inputs data_masked and data_unmasked. I.e. `batch, mask unit sequence, ...`.
            Index for the total data composed of the masked and the unmasked part
        '''
        dim: int = idx_masked.ndim

        idx_total = torch.argsort(torch.cat([idx_masked, idx_unmasked], dim=-1), dim=-1)
        idx_total = idx_total.view(*idx_total.shape, *[1]*(data_unmasked.ndim - dim))
        idx_total = idx_total.expand(*idx_total.shape[:dim], *data_unmasked.shape[dim:])

        data = torch.cat([data_masked, data_unmasked], dim=dim - 1)
        data = torch.gather(data, dim=dim - 1, index=idx_total)

        return data, idx_total

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        '''
        Args:
            batch: Dictionary containing the keys 'x', 'y', 'lead_time' and 'static'. The associated torch tensors have the following shapes:
                x: Tensor of shape [batch, time, parameter, lat, lon]
                y: Tensor of shape [batch, parameter, lat, lon]
                static: Tensor of shape [batch, channel_static, lat, lon]
                climate: Optional tensor of shape [batch, parameter, lat, lon]
                lead_time: Tensor of shape [batch]. Or none.
        Returns:
            Tensor of shape [batch, parameter, lat, lon].
        '''
        assert batch['x'].shape[2] == self.in_channels
        assert batch['x'].shape[3] == self.n_lats_px
        assert batch['x'].shape[4] == self.n_lons_px
        assert batch['y'].shape[1] == self.in_channels
        assert batch['y'].shape[2] == self.n_lats_px
        assert batch['y'].shape[3] == self.n_lons_px
        assert batch['static'].shape[1] == self.in_channels_static
        assert batch['static'].shape[2] == self.n_lats_px
        assert batch['static'].shape[3] == self.n_lons_px

        x_rescaled = (batch['x'] - self.input_scalers_mu) / (
            self.input_scalers_sigma + self.input_scalers_epsilon
        )
        batch_size = x_rescaled.shape[0]

        x_static = (batch['static'] - self.static_input_scalers_mu) / (
            self.static_input_scalers_sigma + self.static_input_scalers_epsilon
        )

        if self.residual == 'temporal':
            # We create a residual of same shape as y
            index = torch.where(batch['lead_time'] > 0, batch['x'].shape[1] - 1, 0)
            index = index.reshape(-1, 1, 1, 1, 1)
            index = index.expand(batch_size, 1, *batch['x'].shape[2:])
            x_hat = torch.gather(batch['x'], dim=1, index=index)
            x_hat = x_hat.squeeze(1)
            assert (
                batch['y'].shape == x_hat.shape
            ), f'Shapes {batch["y"].shape} and {x_hat.shape} do not agree.'
        elif self.residual == 'climate':
            climate_scaled = (batch['climate'] - self.input_scalers_mu.view(1, -1, 1, 1)) / (
                self.input_scalers_sigma.view(1, -1, 1, 1) + self.input_scalers_epsilon
            )

        # [batch, time, parameter, lat, lon] -> [batch, time x parameter, lat, lon]
        x_rescaled = x_rescaled.flatten(1, 2)

        x_embedded = self.patch_embedding(x_rescaled)
        assert x_embedded.shape[1] == self.embed_dim

        if self.residual == 'climate':
            static_embedded = self.patch_embedding_static(
                torch.cat((x_static, climate_scaled), dim=1)
            )
        else:
            static_embedded = self.patch_embedding_static(x_static)
        assert static_embedded.shape[1] == self.embed_dim

        # (batch, embed, lat//patch_size, lon//patch_size) -> (batch, global seq, local seq, embed)
        x_embedded = (
            x_embedded.reshape(
                batch_size,
                self.embed_dim,
                self.global_shape_mu[0],
                self.local_shape_mu[0],
                self.global_shape_mu[1],
                self.local_shape_mu[1],
            )
            .permute(0, 2, 4, 3, 5, 1)
            .flatten(3, 4)
            .flatten(1, 2)
        )

        static_embedded = (
            static_embedded.reshape(
                batch_size,
                self.embed_dim,
                self.global_shape_mu[0],
                self.local_shape_mu[0],
                self.global_shape_mu[1],
                self.local_shape_mu[1],
            )
            .permute(0, 2, 4, 3, 5, 1)
            .flatten(3, 4)
            .flatten(1, 2)
        )
        tokens = x_embedded + static_embedded

        # Now we generate masks based on masking_mode
        indices_masked, indices_unmasked = self.generate_mask((batch_size, self._nglobal_mu))
        indices_masked = indices_masked.to(device=tokens.device)
        indices_unmasked = indices_unmasked.to(device=tokens.device)
        maskdim: int = indices_masked.ndim

        # Unmasking
        unmask_view = (*indices_unmasked.shape, *[1]*(tokens.ndim - maskdim))
        unmasked = torch.gather(
            tokens,
            dim=maskdim - 1,
            index=indices_unmasked.view(*unmask_view).expand(
                *indices_unmasked.shape, *tokens.shape[maskdim:]
            ),
        )

        # Encoder
        x_encoded, attn_weights_encoder, layer_deltas_encoder = self.encoder(unmasked, batch['lead_time'])

        # Generate and position encode the mask tokens
        # (1, 1, 1, embed_dim) -> (batch, global_seq_masked, local seq, embed_dim)
        mask_view = (*indices_masked.shape, *[1]*(tokens.ndim - maskdim))
        masking = self.mask_token.repeat(*static_embedded.shape[:3], 1)
        masked = masking + static_embedded
        masked = torch.gather(
            masked,
            dim=maskdim - 1,
            index=indices_masked.view(*mask_view).expand(
                *indices_masked.shape, *tokens.shape[maskdim:]
            ),
        )

        recon, _ = self.reconstruct_batch(indices_masked, indices_unmasked, masked, x_encoded)

        x_decoded, attn_weights_decoder, layer_deltas_decoder = self.decoder(recon, batch['lead_time'])

        # Output: (batch, global sequence, local sequence, in_channels * patch_size[0] * patch_size[1])
        x_out = self.unembed(x_decoded)

        # Reshape to (batch, global_lat, global_lon, local_lat, local_lon, in_channels * patch_size[0] * patch_size[1])
        assert x_out.shape[0] == batch_size
        assert x_out.shape[1] == self.global_shape_mu[0] * self.global_shape_mu[1]
        assert x_out.shape[2] == self.local_shape_mu[0] * self.local_shape_mu[1]
        assert x_out.shape[3] == self.in_channels * self.patch_size_px[0] * self.patch_size_px[1]
        x_out = x_out.reshape(
            batch_size,
            self.global_shape_mu[0],
            self.global_shape_mu[1],
            self.local_shape_mu[0],
            self.local_shape_mu[1],
            self.in_channels * self.patch_size_px[0] * self.patch_size_px[1]
        )
        # Permute to (batch, in_channels * patch_size[0] * patch_size[1], global_lat, local_lat, global_lon, local_lon)
        x_out = x_out.permute(0, 5, 1, 3, 2, 4)
        # Reshape to (batch, in_channels * patch_size[0] * patch_size[1], lat // patch_size[0], lon // patch_size[1])
        x_out = x_out.flatten(4, 5).flatten(2, 3)
        # Pixel shuffle to (batch, in_channels, lat, lon)
        if self.patch_size_px[0] != self.patch_size_px[1]:
            raise NotImplementedError('Current pixel shuffle implementation assumes same patch size along both dimensions.')
        x_out = F.pixel_shuffle(x_out, self.patch_size_px[0])

        if self.residual == 'temporal':
            x_out = self.output_scalers * x_out + x_hat
        elif self.residual == 'climate':
            x_out = self.output_scalers * x_out + batch['climate']
        elif self.residual == 'none':
            x_out = self.output_scalers * x_out + self.input_scalers_mu.reshape(1, -1, 1, 1)

        return x_out, attn_weights_encoder, layer_deltas_encoder, attn_weights_decoder, layer_deltas_decoder

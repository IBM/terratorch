# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

import itertools
import math
from collections import OrderedDict
from functools import partial

import torch
from einops import repeat
from torch import nn
from torch.distributions.dirichlet import Dirichlet
import torch.nn.functional as F
from .multimae_utils import Block, trunc_normal_


class MultiMAE(nn.Module):
    """MultiMAE: Multi-task Multi-modal Masked Autoencoder
    This module performs masking in its forward pass.
    The MultiViT module defined below inherits from this module and performs a regular forward pass,
    and should be used instead for downstream tasks


    :param input_adapters: Dictionary of task -> input adapters
    :param output_adapters: Optional dictionary of task -> output adapters

    :param num_global_tokens: Number of additional global tokens to add (like cls tokens), default is 1
    :param dim_tokens: Dimension of encoder tokens
    :param depth: Depth of encoder
    :param num_heads: Number of attention heads
    :param mlp_ratio: MLP hidden dim ratio
    :param qkv_bias: Set to False to disable bias
    :param drop_rate: Dropout after MLPs and Attention
    :param attn_drop_rate: Attention matrix drop rate
    :param drop_path_rate: DropPath drop rate
    :param norm_layer: Type of normalization layer
    :param fp32_output_adapters: List of task identifiers to force output adapters to
    run with mixed precision turned off for stability reasons.
    """

    default_norm_layer = partial(nn.LayerNorm, eps=1e-6)

    def __init__(
        self,
        input_adapters: dict[str, nn.Module],
        output_adapters: dict[str, nn.Module] | None,
        loss_functions: dict[str, nn.Module] | None,
        num_global_tokens: int = 1,
        dim_tokens: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = default_norm_layer,
        fp32_output_adapters: list[str] | None = None,
        num_input_tokens: int = 128,
        merge_method: str = None,
        **kwargs,
    ):
        super().__init__()

        # Initialize input and output adapters
        for adapter in input_adapters.values():
            adapter.init(dim_tokens=dim_tokens)
        self.input_adapters = nn.ModuleDict(input_adapters)
        if output_adapters is not None:
            for adapter in output_adapters.values():
                adapter.init(dim_tokens_enc=dim_tokens)
            self.output_adapters = nn.ModuleDict(output_adapters)
        else:
            self.output_adapters = None
        self.fp32_output_adapters = fp32_output_adapters or []
        self.loss_functions = loss_functions
        self.num_input_tokens = num_input_tokens

        # Additional learnable tokens that can be used by encoder to process/store global information
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, dim_tokens))
        trunc_normal_(self.global_tokens, std=0.02)

        if merge_method == 'concat':  # TODO: Move prepare/concat to this model forward?
            embed_factor = len(input_adapters)
        else:
            embed_factor = 1
        self.merge_method = merge_method
        assert merge_method in ['mean', 'max', 'concat', None], "merge_method must be one of mean, max, concat, None."
        self.out_channels = [int(dim_tokens) * embed_factor] * depth

        # Transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.layers: nn.ModuleList = nn.ModuleList(
            Block(
                dim=dim_tokens,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        )

        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif "kv" in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)

            if isinstance(m, nn.Conv2d):
                if ".proj" in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = {"global_tokens"}

        for task, adapter in self.input_adapters.items():
            if hasattr(adapter, "no_weight_decay"):
                to_skip = adapter.no_weight_decay()
                to_skip = {f"input_adapters.{task}.{name}" for name in to_skip}
                no_wd_set = no_wd_set | to_skip

        for task, adapter in self.output_adapters.items():
            if hasattr(adapter, "no_weight_decay"):
                to_skip = adapter.no_weight_decay()
                to_skip = {f"output_adapters.{task}.{name}" for name in to_skip}
                no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def sample_alphas(self, B: int, n_tasks: int, alphas: float = 1.0, eps: float = 1e-5):
        """
        Sample alphas for Dirichlet sampling such that tasks are first uniformly chosen and then Dirichlet sampling
        is performed over the chosen ones.

        :param B: Batch size
        :param n_tasks: Number of input tasks
        :param alphas: Float or list to multiply task choices {0,1} by
        :param eps: Small constant since Dirichlet alphas need to be positive
        """
        valid_task_choices = torch.Tensor([list(i) for i in itertools.product([0, 1], repeat=n_tasks)][1:])
        rand_per_sample_choice = torch.randint(0, len(valid_task_choices), (B,))
        alphas_tensor = torch.index_select(valid_task_choices, 0, rand_per_sample_choice)
        alphas_tensor = alphas_tensor * torch.tensor(alphas) + eps
        return alphas_tensor

    def generate_random_masks(
        self,
        input_tokens: dict[str, torch.Tensor],
        num_input_tokens: int,
        alphas: float | list[float] = 1.0,
        sample_tasks_uniformly: bool = False,
    ):
        """
        Sample a total of num_input_tokens from different tasks using Dirichlet sampling.

        :param input_tokens: Dictionary of tensors to sample num_input_tokens from
        :param num_input_tokens: Number of tokens to select
        :param alphas: Dirichlet distribution parameter alpha. Lower alpha = harder,
            less uniform sampling. Can be float or list of floats.
        :param sample_tasks_uniformly: Set to True to first sample 1-n_tasks uniformly at random
            for each sample in the batch. Dirichlet sampling is then done over selected subsets.
        """
        B = list(input_tokens.values())[0].shape[0]
        device = next(iter(input_tokens.values())).device

        alphas = [alphas] * len(input_tokens) if isinstance(alphas, float) else alphas
        if sample_tasks_uniformly:
            alphas = self.sample_alphas(B, len(input_tokens), alphas=alphas)
            task_sampling_dist = Dirichlet(alphas).sample().to(device)
        else:
            task_sampling_dist = Dirichlet(torch.Tensor(alphas)).sample((B,)).to(device)

        samples_per_task = (task_sampling_dist * num_input_tokens).round().long()

        task_masks = []
        num_tokens_per_task = [task_tokens.shape[1] for task_tokens in input_tokens.values()]
        for i, num_tokens in enumerate(num_tokens_per_task):
            # Use noise to shuffle arange
            noise = torch.rand(B, num_tokens, device=device)  # noise in [0, 1]
            ids_arange_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            mask = torch.arange(num_tokens, device=device).unsqueeze(0).expand(B, -1)
            mask = torch.gather(mask, dim=1, index=ids_arange_shuffle)
            # 0 is keep (unmasked), 1 is remove (masked)
            mask = torch.where(mask < samples_per_task[:, i].unsqueeze(1), 0, 1)
            task_masks.append(mask)

        mask_all = torch.cat(task_masks, dim=1)
        ids_shuffle = torch.argsort(mask_all + torch.rand_like(mask_all.float()), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_input_tokens]

        # Update binary mask to adjust for task rounding
        mask_all = torch.ones_like(mask_all)
        mask_all[:, :num_input_tokens] = 0
        # Unshuffle to get the binary mask
        mask_all = torch.gather(mask_all, dim=1, index=ids_restore)
        # Split to get task masks
        task_masks = torch.split(mask_all, num_tokens_per_task, dim=1)
        # Convert to dict
        task_masks = dict(zip(input_tokens.keys(), task_masks, strict=False))

        return task_masks, ids_keep, ids_restore

    @staticmethod
    def make_mask(
        N_H,
        N_W,
        xy_idxs,
        full_tasks=[],
        indicate_visible=True,
        flatten=True,
        device="cuda",
    ):
        """
        Creates masks for each task, given lists of un-masked x,y coordinates.
        """
        xy_idxs = {k: torch.LongTensor(v) for k, v in xy_idxs.items()}

        task_masks = {k: torch.ones(N_H, N_W).to(device) for k in xy_idxs.keys()}

        for k in xy_idxs.keys():
            if len(xy_idxs[k]) > 0:
                task_masks[k][xy_idxs[k][:, 1], xy_idxs[k][:, 0]] = 0

        for task in full_tasks:
            task_masks[task][:] = 0

        if not indicate_visible:
            task_masks = {k: 1 - v for k, v in task_masks.items()}

        if flatten:
            task_masks = {k: v.flatten().unsqueeze(0) for k, v in task_masks.items()}

        return task_masks

    def generate_input_info(self, input_task_tokens, image_size):
        input_info = OrderedDict()
        i = 0
        input_info["tasks"] = {}
        for domain, tensor in input_task_tokens.items():
            num_tokens = tensor.shape[1]
            d = {
                "num_tokens": num_tokens,
                "has_2d_posemb": True,  # TODO: Modify when adding non-2D tasks
                "start_idx": i,
                "end_idx": i + num_tokens,
            }
            i += num_tokens
            input_info["tasks"][domain] = d

        input_info["image_size"] = image_size
        input_info["num_task_tokens"] = i
        input_info["num_global_tokens"] = self.num_global_tokens

        return input_info

    def forward(
        self,
        x: dict[str, torch.Tensor] | torch.Tensor,
        mask_inputs: bool = True,
        task_masks: dict[str, torch.Tensor] = None,
        num_input_tokens: int | None = None,
        alphas: float | list[float] = 0.5,
        sample_tasks_uniformly: bool = False,
    ):
        """
        Forward pass through input adapters, transformer encoder and output adapters.
        If specified, will randomly drop input tokens.

        :param x: Input tensor or dictionary of tensors
        :param mask_inputs: Set to True to enable random masking of input patches
        :param task_masks: Optional dictionary of task->mask pairs.
        :param num_input_tokens: Number of tokens to randomly select for encoder.
            Only used if mask_inputs is True.
        :param alphas: Dirichlet distribution parameter alpha for task sampling.
            Higher alpha = harder, less uniform sampling. Can be float or list of floats.
        :param sample_tasks_uniformly: Set to True if tasks should be uniformly presampled,
            before Dirichlet sampling decides share of masked tokens between them.
        """

        ## Processing input modalities
        # If input x is a Tensor, assume it's RGB
        x = {"rgb": x} if isinstance(x, torch.Tensor) else x

        # Need image size for tokens->image reconstruction
        # Assuming same image size for all modalities
        shape = list(x.values())[0].shape
        B = shape[0]
        H, W = shape[-2:]

        # Encode selected inputs to tokens
        input_task_tokens = {
            domain: self.input_adapters[domain](tensor)
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }

        input_info = self.generate_input_info(input_task_tokens=input_task_tokens, image_size=(H, W))

        # Select random subset of tokens from the chosen input tasks and concatenate them
        if mask_inputs:
            num_input_tokens = num_input_tokens or self.num_input_tokens
        else:
            num_input_tokens = sum([tensor.shape[1] for tensor in input_task_tokens.values()])

        # Generating masks
        if task_masks is None:
            task_masks, ids_keep, ids_restore = self.generate_random_masks(
                input_task_tokens, num_input_tokens, alphas=alphas, sample_tasks_uniformly=sample_tasks_uniformly
            )
        else:
            mask_all = torch.cat([task_masks[task] for task in input_task_tokens.keys()], dim=1)
            ids_shuffle = torch.argsort(mask_all, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, : (mask_all == 0).sum()]

        input_tokens = torch.cat([task_tokens for task_tokens in input_task_tokens.values()], dim=1)

        # Apply mask
        input_tokens = torch.gather(input_tokens, dim=1,
                                    index=ids_keep.unsqueeze(-1).repeat(1, 1, input_tokens.shape[2]))

        # Add global tokens to input tokens
        global_tokens = repeat(self.global_tokens, "() n d -> b n d", b=B)
        input_tokens = torch.cat([input_tokens, global_tokens], dim=1)

        # Transformer forward pass
        outputs = []
        encoder_tokens = input_tokens
        for layer in self.layers:
            encoder_tokens = layer(encoder_tokens)
            outputs.append(encoder_tokens)

        # Output decoders
        if self.output_adapters is None:
            return outputs, task_masks

        # TODO: Add target masking
        # Decode tokens for each task using task-specific output adapters
        preds = {
            domain: self.output_adapters[domain](
                encoder_tokens=encoder_tokens,
                input_info=input_info,
                ids_keep=ids_keep,
                ids_restore=ids_restore,
            )
            for domain in self.output_adapters
            if domain not in self.fp32_output_adapters
        }
        # Force running selected output adapters in fp32 mode
        with torch.amp.autocast('cuda', enabled=False):
            for domain in self.fp32_output_adapters:
                if domain not in self.output_adapters:
                    continue
                preds[domain] = self.output_adapters[domain](
                    encoder_tokens=encoder_tokens.float(),
                    input_info=input_info,
                    ids_keep=ids_keep,
                    ids_restore=ids_restore,
                )

        loss = {f'{domain}_loss': self.loss_functions[domain](pred, x[domain], task_masks[domain])
                for domain, pred in preds.items()}

        loss['loss'] = torch.stack(list(loss.values())).sum()

        # Convert token masks to pixel masks
        for key, mask in task_masks.items():
            # TODO: Assumes squared inputs
            N_sqrt = int(mask.shape[1] ** 0.5)
            mask = mask.view(B, N_sqrt, N_sqrt)
            task_masks[key] = F.interpolate(mask.unsqueeze(1).to(torch.uint8), size=(H, W), mode='nearest').squeeze(1)

        return loss, preds, task_masks


class MultiViT(MultiMAE):
    """MultiViT: Multi-modal Vision Transformer
    This is MultiMAE without masking and with a simplified / faster forward pass


    :param input_adapters: Dictionary of task -> input adapters
    :param output_adapters: Optional dictionary of task -> output adapters

    :param num_global_tokens: Number of additional global tokens to add (like cls tokens), default is 1
    :param dim_tokens: Dimension of encoder tokens
    :param depth: Depth of encoder
    :param num_heads: Number of attention heads
    :param mlp_ratio: MLP hidden dim ratio
    :param qkv_bias: Set to False to disable bias
    :param drop_rate: Dropout after MLPs and Attention
    :param attn_drop_rate: Attention matrix drop rate
    :param drop_path_rate: DropPath drop rate
    :param norm_layer: Type of normalization layer
    """

    def process_input(self, x):
        # If input x is a Tensor, assume it's RGB
        x = {"rgb": x} if isinstance(x, torch.Tensor) else x
        # Need image size for tokens->image reconstruction
        if "rgb" in x:
            B, _, H, W = x["rgb"].shape
        elif "semseg" in x:
            B, H, W = x["semseg"].shape
            H *= self.input_adapters["semseg"].stride_level
            W *= self.input_adapters["semseg"].stride_level
        else:
            # Assuming same shape for all inputs
            B, _, H, W = list(x.values())[0].shape

        # Encode selected inputs to tokens
        input_task_tokens = {
            domain: self.input_adapters[domain](tensor)
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }

        input_info = self.generate_input_info(input_task_tokens=input_task_tokens, image_size=(H, W))
        input_tokens = torch.cat(list(input_task_tokens.values()), dim=1)

        # Add global tokens to input tokens
        global_tokens = repeat(self.global_tokens, "() n d -> b n d", b=B)
        input_tokens = torch.cat([input_tokens, global_tokens], dim=1)
        return input_tokens, input_info

    def process_output(self, x, input_info, method):
        # TODO: Make it generalizable
        num_tokens = list(input_info['tasks'].values())[0]['num_tokens']
        assert any(info['num_tokens'] == num_tokens for info in list(input_info['tasks'].values())), \
            "Current code only supports modalities with the same number of tokens"
        def _unstack_image_modalities(x):
            x = torch.split(x, num_tokens, dim=1)  # Split tokens by modality
            x = torch.stack(x, dim=1)  # (B, M, N, D)
            return x

        # Merge tokens from different modalities
        if self.merge_method == 'mean':
            x = _unstack_image_modalities(x)
            x = x.mean(dim=1)

        elif self.merge_method == 'max':
            x = _unstack_image_modalities(x)
            x = x.max(dim=1)[0]

        elif self.merge_method == 'concat':
            # TODO: Handle missing modalities with a learnable self.missing_token. Currently expects all modalities.
            assert len(input_info['tasks']) == len(self.input_adapters), "Method concat expects all modalities as input"
            x = _unstack_image_modalities(x)
            x = torch.cat(x.unbind(dim=1), dim=-1)

        elif self.merge_method is None:
            pass  # Do nothing
        else:
            raise NotImplementedError(f'Merging method {self.merge_method} is not implemented. '
                                      f'Select one of mean, max or concat.')

        return x


    def forward(
        self,
        x: dict[str, torch.Tensor] | torch.Tensor,
    ):
        """
        Forward pass through input adapters, transformer encoder and output adapters.

        :param x: Input tensor or dictionary of tensors
        :param return_all_layers: Set to True to return all transformer layers
        """

        x, input_info = self.process_input(x)

        out = []
        for block in self.layers:
            x = block(x)
            out.append(x)

        # Drop global token
        global_token = [tokens[:, :input_info['num_global_tokens']] for tokens in out]
        out = [tokens[:, input_info['num_global_tokens']:] for tokens in out]

        out = [self.process_output(x, input_info, self.merge_method) for x in out]

        return out

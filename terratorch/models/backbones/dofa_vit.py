# reference torchgeo https://torchgeo.readthedocs.io/en/latest/_modules/torchgeo/models/dofa.html#DOFA
import torch
import torch.nn.functional as F
import logging
from collections.abc import Callable
from functools import partial
import huggingface_hub
import torch.nn as nn
from typing import List
import huggingface_hub
from torchvision.models._api import Weights, WeightsEnum
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
import math
from timm.models.vision_transformer import VisionTransformer
import pdb

waves_list= {
  "COASTAL_AEROSOL": 0.44,
  "BLUE": 0.49,
  "GREEN": 0.56,
  "RED": 0.665,
  "RED_EDGE_1": 0.705,
  "RED_EDGE_2": 0.74, 
  "RED_EDGE_3": 0.783,
  "NIR_BROAD": 0.832,
  "NIR_NARROW": 0.864,
  "WATER_VAPOR": 0.945,
  "CIRRUS": 1.373,
  "SWIR_1": 1.61,
  "SWIR_2": 2.20,
  "THEMRAL_INFRARED_1": 10.90,
  "THEMRAL_INFRARED_12": 12.00, 
  "VV": 5.405,
  "VH": 5.405,
  "ASC_VV": 5.405,
  "ASC_VH": 5.405,
  "DSC_VV": 5.405,
  "DSC_VH": 5.405,
  "VV-VH": 5.405
}


def resize(input: torch.Tensor,
           size: tuple[int, int] | None = None,
           scale_factor: float | None = None,
           mode: str = 'nearest',
           align_corners: bool | None = None,
           warning: bool = True) -> torch.Tensor:
    """Resize input tensor with alignment warning check.
    
    Args:
        input: Input tensor of shape [B, C, H, W]
        size: Target output size (H, W)
        scale_factor: Multiplier for spatial size
        mode: Interpolation mode ('bilinear', 'bicubic'.)
        align_corners: If True, aligns corners for non-nearest modes
        warning: If True, warns about potential alignment issues
    
    Returns:
        Resized tensor of shape [B, C, H_new, W_new]
    """
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
    """Resize pos_embed weights.
    Resize pos_embed using bicubic interpolate method.
    Args:
        pos_embed (torch.Tensor): Position embedding weights.
        input_shpae (tuple): Tuple for (downsampled input image height,
            downsampled input image width).
        pos_shape (tuple): The resolution of downsampled origin training
            image.
        mode (str): Algorithm used for upsampling:
            ``'bilinear'`` | ``'bicubic'`` . Default: ``'bilinear'``
    Return:
        torch.Tensor: The resized pos_embed of shape [B, L_new, C]
    """
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
    pos_h, pos_w = pos_shape
    cls_token_weight = pos_embed[:, 0]
    pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
    pos_embed_weight = pos_embed_weight.reshape(
        1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
    pos_embed_weight = resize(
        pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
    cls_token_weight = cls_token_weight.unsqueeze(1)
    pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
    pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
    return pos_embed


from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block

def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    old_shape = pos
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class TransformerWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1):
        super(TransformerWeightGenerator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x):
        # x should have shape [seq_len, batch, input_dim]
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(
            transformer_output[-1]
        )  # Using the last output to generate bias
        return weights, bias

class Basic1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        conv = nn.Linear(in_channels, out_channels, bias)
        self.conv = nn.Sequential(
            conv,
        )
        if not bias:
            self.conv.add_module("ln", nn.LayerNorm(out_channels))
        self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class FCResLayer(nn.Module):
    def __init__(self, linear_size=128):
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        # y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


class Dynamic_MLP_OFA(nn.Module):
    """
    Input: channels of wavelength (normalized): List -> List
           kernel size of the depth-wise convolution: kernel_size, default 3x3
           wv_planes
           inplanes
    """

    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024, convert_patch_14_to_16=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1
        self.convert_patch_14_to_16 = convert_patch_14_to_16

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)
        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """
        initialize the base weights and dynamic mlp weights
        """
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, wvs):
        inplanes = wvs.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)  # 3x3x3
        # bias = None

        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        if self.convert_patch_14_to_16:
            assert self.kernel_size == 14
            self.new_kernel_size = 16
            weights = torch.nn.functional.interpolate(
                weights,
                size=(16, 16),
                mode='bicubic',
                align_corners=False
            )
            dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=self.new_kernel_size, padding=1, dilation=1
            )
        else:
            dynamic_out = F.conv2d(
                img_feat, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
            )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves

class OFAViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        drop_rate=0.0,
        out_indices=None,
        drop_path_rate=0.0,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        wv_planes=128,
        num_classes=45,
        global_pool=True,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.wv_planes = wv_planes
        self.out_indices = out_indices

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim
        )
        self.img_size = img_size
        if isinstance(img_size, tuple):
            self.img_size = self.img_size[0]

        self.num_patches = (self.img_size // patch_size) ** 2
        self.patch_embed.num_patches = self.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # ---------------------------------------------------------------------------
        # prompt setting
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward_features(self, x, wave_list):
        # embed patches
        wavelist = torch.tensor(wave_list, device=x.device).float()
        self.waves = wavelist
        # TODO #1 how to convert coordinates to higher dimension
        x, _ = self.patch_embed(x, self.waves)

        hw = self.img_size // self.patch_embed.kernel_size
        hw_shape = (hw, hw)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        out_features = []

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                out_features.append(x)

        return out_features

    def forward(self, x, wave_list):
        x = self.forward_features(x, wave_list)
        return x


class DOFAViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=14,
        drop_rate=0.0,
        out_indices=None,
        drop_path_rate=0.0,
        embed_dim=768,
        depth=24,
        num_heads=16,
        wv_planes=128,
        num_classes=45,
        global_pool=True,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        convert_patch_14_to_16=False,
    ):
        super().__init__()

        self.wv_planes = wv_planes
        self.out_indices = out_indices

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=128, inter_dim=128, kernel_size=14, embed_dim=embed_dim, convert_patch_14_to_16=convert_patch_14_to_16
        )
        self.img_size = img_size
        if isinstance(img_size, tuple):
            self.img_size = self.img_size[0]

        self.num_patches = (self.img_size // patch_size) ** 2
        self.patch_embed.num_patches = self.num_patches
        model_args = dict(patch_size=patch_size, embed_dim=embed_dim, depth=depth, drop_path_rate=0.1,
                num_heads=num_heads, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        self.model = VisionTransformer(**model_args)

        del self.model.patch_embed.proj
        self.dynamic_img_size = True
        self.waves = None
        self.norm = norm_layer(embed_dim)

    def forward_features(self, x, wave_list):
        # embed patches
        wavelist = torch.tensor(wave_list, device=x.device, requires_grad=False).float()
        self.waves = wavelist
        x, _ = self.patch_embed(x, self.waves)
        B,HW,C = x.shape
        hw = int(math.sqrt(HW))
        hw_shape = (hw, hw)
        if self.dynamic_img_size:
            x = x.view(B,hw,hw,C)

        x = self.model._pos_embed(x)
        # masking: length -> length * mask_ratio
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        out_features = []

        # apply Transformer blocks
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.out_indices:
                out_features.append(x)

        x = self.model.norm(x)
        return out_features

    def forward(self, x, wave_list):
        x = self.forward_features(x, wave_list)
        return x


def vit_base_patch14(**kwargs):
    model = DOFAViT(
        out_indices=[4, 6, 10, 11],
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch14(**kwargs):
    model = DOFAViT(
        out_indices=[5, 9, 15, 21],
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def vit_base_patch16(**kwargs):
    model = OFAViT(
        out_indices=[4, 6, 10, 11],
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = OFAViT(
        out_indices=[5, 11, 17, 22],
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

class DOFAEncoderWrapper(nn.Module):

    """
    A wrapper for DOFA models from torchgeo to return only the forward pass of the encoder 
    Attributes:
        dofa_model (DOFA): The instantiated dofa model
    Methods:
        forward(x: List[torch.Tensor], wavelengths: list[float]) -> torch.Tensor:
            Forward pass for embeddings with specified indices.
    """

    def __init__(self, dofa_model, wavelengths, weights=None, out_indices=None) -> None:
        """
        Args:
            dofa_model (DOFA): The decoder module to be wrapped.
            weights ()
        """
        super().__init__()
        self.dofa_model = dofa_model
        self.weights = weights
        self.wavelengths = wavelengths

        self.out_indices = out_indices if out_indices else [-1]
        self.out_channels = [self.dofa_model.patch_embed.embed_dim] * len(self.out_indices)
        self.dofa_model.out_indices = self.out_indices

    def forward(self, x: List[torch.Tensor], **kwargs) -> torch.Tensor:
        N,C,oh,ow = x.shape
        wavelist = torch.tensor(self.wavelengths, device=x.device).float()
        outs = self.dofa_model.forward_features(x, wavelist)

        return tuple(outs)

def get_wavelenghts(model_bands: list[str]) -> list[float]:
    """Extract wavelength values for given spectral bands.
    
    Args:
        model_bands: List of band names (e.g., ['RED', 'NIR', 'SWIR_1'])
    
    Returns:
        List of corresponding wavelength values in micrometers
    """
    wavelengths = [waves_list[x.split('.')[-1]] for x in model_bands]
    return wavelengths


@TERRATORCH_BACKBONE_REGISTRY.register
def dofav1_base_patch16_224(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: str = None, out_indices: list | None = None, pos_interpolation_mode: str = 'bilinear', **kwargs):
    model = vit_base_patch16(**kwargs)
    input_size = kwargs["img_size"] if "img_size" in kwargs else 224
    if pretrained:
        model = load_dofa_weights(model, pos_interpolation_mode, ckpt_data, weights, input_size)
    wavelengths = get_wavelenghts(model_bands)
    
    return DOFAEncoderWrapper(model, wavelengths, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def dofav1_large_patch16_224(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: str = None, out_indices: list | None = None, pos_interpolation_mode: str = 'bilinear', **kwargs):
    model = vit_large_patch16(**kwargs)
    input_size = kwargs["img_size"] if "img_size" in kwargs else 224
    if pretrained:
        model = load_dofa_weights(model, pos_interpolation_mode, ckpt_data, weights, input_size)
    wavelengths = get_wavelenghts(model_bands)
    
    return DOFAEncoderWrapper(model, wavelengths, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def dofav2_base_patch14_224(model_bands, pretrained = False, ckpt_data: str | None = None,  weights: Weights | None = None, out_indices: list | None = None, pos_interpolation_mode: str = 'bilinear', convert_patch_14_to_16: bool = False, **kwargs):
    kwargs["convert_patch_14_to_16"] = convert_patch_14_to_16
    model = vit_base_patch14(**kwargs)
    input_size = kwargs["img_size"] if "img_size" in kwargs else 224
    if pretrained:
        model = load_dofa_weights(model, pos_interpolation_mode, ckpt_data, weights, input_size)
    wavelengths = get_wavelenghts(model_bands)
    
    return DOFAEncoderWrapper(model, wavelengths, weights, out_indices)

@TERRATORCH_BACKBONE_REGISTRY.register
def dofav2_large_patch14_224(model_bands, pretrained = False, ckpt_data: str | None = None, weights: Weights | None = None, out_indices: list | None = None, pos_interpolation_mode: str = 'bilinear', convert_patch_14_to_16: bool = False, **kwargs):
    """DOFA v2 large model with patch size 14x14 and 224x224 input resolution.
    
    Args:
        model_bands: List of spectral bands to use
        pretrained: Whether to load pretrained weights
        ckpt_data: Path or URL to checkpoint data
        weights: Pretrained weights enum
        out_indices: Indices of transformer blocks to output features from
        **kwargs: Additional arguments passed to the model constructor
        
    Returns:
        DOFAEncoderWrapper: Wrapped DOFA model
    """
    # For v2, we're using the same base model but with patch size 14
    kwargs["convert_patch_14_to_16"] = convert_patch_14_to_16
    model = vit_large_patch14(**kwargs)
    convert_patch_14_to_16 = False
    input_size = kwargs.get("img_size", 224)
    if pretrained:
        model = load_dofa_weights(model, pos_interpolation_mode, ckpt_data, weights, input_size, patch_size=14)
    wavelengths = get_wavelenghts(model_bands)
    
    return DOFAEncoderWrapper(model, wavelengths, weights, out_indices)

def load_dofa_weights(model: nn.Module, mode: str, ckpt_data: str | None = None,  weights: Weights | None = None, input_size: int = 224, patch_size: int = 16) -> nn.Module:
    state_dict = model.state_dict()
    print("Loading weights")
    if ckpt_data is not None:
        if ckpt_data.find("https://hf.co/") > -1:
            repo_id = ckpt_data.split("/resolve/")[0].replace("https://hf.co/", '')
            filename = ckpt_data.split("/")[-1]
            ckpt_data = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint_model = torch.load(ckpt_data, map_location="cpu", weights_only=True)

        for k in ["head.weight", "head.bias"]:
                if (
                    k in checkpoint_model
                    and checkpoint_model[k].shape != state_dict[k].shape
                ):
                    logging.info(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
        if input_size != 224 and patch_size==16:
            if (
                "pos_embed" in checkpoint_model
                and checkpoint_model["pos_embed"].shape != state_dict["pos_embed"].shape
            ):
                logging.info("Resizing pos_embed from pretrained checkpoint")
                h, w = input_size, input_size
                pos_size = int(math.sqrt(checkpoint_model['pos_embed'].shape[1] - 1))
                checkpoint_model["pos_embed"] = resize_pos_embed(pos_embed=checkpoint_model['pos_embed'],input_shpae=(h // patch_size, w // patch_size), pos_shape=(pos_size, pos_size), mode=mode)
    
        msg = model.load_state_dict(checkpoint_model, strict=False)
    
        logging.info(msg)
    else:
        if weights is not None:
            
            checkpoint_model = weights.get_state_dict(progress=True)
            allowed_missing_keys =  {'fc_norm.weight', 'fc_norm.bias', 'head.weight', 'head.bias'}
            if input_size != 224:
                if (
                    "pos_embed" in checkpoint_model
                    and checkpoint_model["pos_embed"].shape != state_dict["pos_embed"].shape
                ):
                    logging.info("Removing key pos_embed from pretrained checkpoint")
                    del checkpoint_model["pos_embed"]
                    allowed_missing_keys.add('pos_embed')
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
            logging.info(f"Weights loaded.")
            # Both fc_norm and head are generated dynamically
            assert set(missing_keys) <= allowed_missing_keys
            assert not unexpected_keys
        else:
            print("No weights to load.")
            
    return model
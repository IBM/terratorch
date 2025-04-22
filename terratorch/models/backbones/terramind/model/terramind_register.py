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

import torch
import logging
from torch import nn
from functools import partial
from .terramind import TerraMind
from .terramind_vit import TerraMindViT
from .terramind_tim import TerraMindTiM
from .terramind_generation import TerraMindGeneration
from .tm_utils import LayerNorm
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY, TERRATORCH_FULL_MODEL_REGISTRY
from huggingface_hub import hf_hub_download

logger = logging.getLogger('terramind')

# Model definitions
__all__ = [
    # pre-trained models
    'terramind_v01_base',
    'terramind_v1_base',
    'terramind_v1_large',
    'terramind_v1_base_tim',
    'terramind_v1_large_tim',
    'terramind_v1_base_mae',
    'terramind_v1_large_mae',
    'terramind_v1_base_generate',
    'terramind_v1_large_generate',
]

pretrained_weights = {
        "terramind_v01_base": {
            "hf_hub_id": "FAST-EO/TerraMind-0.1-base",
            "hf_hub_filename": "TerraMind_v01_base.pt",
        },
        "terramind_v1_base": {
            "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-base",
            "hf_hub_filename": "TerraMind_v1_base.pt",
        },
        "terramind_v1_large": {
            "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-large",
            "hf_hub_filename": "TerraMind_v1_large.pt",
        },
    }

PRETRAINED_BANDS = {
    'untok_sen2l2a@224': [
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "WATER_VAPOR",
        "SWIR_1",
        "SWIR_2",
    ],
    'untok_sen2l1c@224': [
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "WATER_VAPOR",
        "CIRRUS",
        "SWIR_1",
        "SWIR_2",
    ],
    'untok_sen2rgb@224': ["RED", "GREEN", "BLUE"],
    'untok_sen1grd@224': ["VV", "VH"],
    'untok_sen1rtc@224': ["VV", "VH"],
    'untok_dem@224': ["DEM"],
}


v01_pretraining_mean = {
    'untok_sen2l2a@224': [794.311,  925.161, 1183.128, 1338.041, 1667.254, 2233.633, 2460.96 , 2555.569, 2619.542, 2703.298, 2406.497, 1841.645],
    'tok_sen2l2a@224': [794.311,  925.161, 1183.128, 1338.041, 1667.254, 2233.633, 2460.96 , 2555.569, 2619.542, 2703.298, 2406.497, 1841.645],
    'tok_sen1grd@224': [-12.599, -20.293],
    'tok_lulc@224': [0],
    'tok_dem@224': [435.726],
}

v01_pretraining_std = {
    'untok_sen2l2a@224': [1164.883, 1205.586, 1223.713, 1399.638, 1403.298, 1378.513, 1434.924, 1491.141, 1454.089, 1660.395, 1473.248, 1365.080],
    'tok_sen2l2a@224': [1164.883, 1205.586, 1223.713, 1399.638, 1403.298, 1378.513, 1434.924, 1491.141, 1454.089, 1660.395, 1473.248, 1365.080],
    'tok_sen1grd@224': [5.195, 5.890],
    'tok_lulc@224': [1],
    'tok_dem@224': [560.326],
}

v1_pretraining_mean = {
    'untok_sen2l2a@224': [1390.458, 1503.317, 1718.197, 1853.91, 2199.1, 2779.975, 2987.011, 3083.234, 3132.22, 3162.988, 2424.884, 1857.648],
    'untok_sen2l1c@224': [2357.089, 2137.385, 2018.788, 2082.986, 2295.651, 2854.537, 3122.849, 3040.56, 3306.481, 1473.847,  506.07, 2472.825, 1838.929],
    'untok_sen2rgb@224': [87.271, 80.931, 66.667],
    'untok_sen1grd@224': [-12.599, -20.293],
    'untok_sen1rtc@224': [-10.93, -17.329],
    'untok_dem@224': [670.665],
    'tok_sen1grd@224': [-12.599, -20.293],
    'tok_sen1rtc@224': [-10.93, -17.329],
    'tok_sen2l2a@224': [1390.458, 1503.317, 1718.197, 1853.91, 2199.1, 2779.975, 2987.011, 3083.234, 3132.22, 3162.988, 2424.884, 1857.648],
    'tok_lulc@224': [0],
    'tok_dem@224': [670.665],
    'tok_ndvi@224': [0.327],
}

v1_pretraining_std = {
    'untok_sen2l2a@224': [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926, 1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311],
    'untok_sen2l1c@224': [1624.683, 1675.806, 1557.708, 1833.702, 1823.738, 1733.977, 1732.131, 1679.732, 1727.26, 1024.687, 442.165, 1331.411, 1160.419],
    'untok_sen2rgb@224': [58.767, 47.663, 42.631],
    'untok_sen1grd@224': [5.195, 5.890],
    'untok_sen1rtc@224': [4.391, 4.459],
    'untok_dem@224': [951.272],
    'tok_sen2l2a@224': [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926, 1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311],
    'tok_sen1grd@224': [5.195, 5.890],
    'tok_sen1rtc@224': [4.391, 4.459],
    'tok_lulc@224': [1],
    'tok_dem@224': [951.272],
    'tok_ndvi@224': [0.322],
}



def select_modality_patch_embed_weights(model: TerraMindViT, bands: dict[str, list], pretrained_bands: dict[str, list]):
    """
    Update patch embeddings weights for each provided modality by selecting the pretrained weights for each band.
    Args:
         model (TerraMindViT): model
         bands (dict[str, list]): Bands with format {<modality>: [<band names>]}
         pretrained_bands (dict[str, list]): Pretrained bands of the model with format {<modality>: [<band names>]}
    """
    # Update modality names to match model layer names
    bands = {model.mod_name_mapping[k]: v for k, v in bands.items()}
    for mod, mod_bands in bands.items():
        if mod not in pretrained_bands:
            logger.info(f"Cannot load band weights for modality {mod}, not found in pretrained bands.")
            continue

        pixel_count = model.encoder_embeddings[mod].patch_size[0] * model.encoder_embeddings[mod].patch_size[1]

        pretrained_weight = model.encoder_embeddings[mod].proj.weight.clone()
        # Init new projection layer with updated number of channels
        model.encoder_embeddings[mod].proj = nn.Linear(
            pixel_count * len(mod_bands),
            model.encoder_embeddings[mod].dim_tokens,
            bias=False
        )
        temp_weight = model.encoder_embeddings[mod].proj.weight.clone()

        # Reshape to [dim, pixel, band]
        temp_weight = temp_weight.view(temp_weight.shape[0], pixel_count, -1)
        pretrained_weight = pretrained_weight.view(pretrained_weight.shape[0], pixel_count, -1)

        # Copy weights of bands
        for index, band in enumerate(mod_bands):
            if band in pretrained_bands[mod]:
                logging.info(f"Loaded weights for {band} in position {index} of patch embed")
                pretrained_index = pretrained_bands[mod].index(band)
                temp_weight[..., index] = pretrained_weight[..., pretrained_index]

        # Update model weights
        model.encoder_embeddings[mod].proj.weight = nn.Parameter(temp_weight.view(temp_weight.shape[0], -1))

    return model


def checkpoint_filter_fn(state_dict, model: TerraMindViT | TerraMind) -> dict:
    """Manually filter pre-trained weights for TerraMind to enable strict weight loading."""

    model_state_dict = model.state_dict()
    clean_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                clean_dict[k] = v
            else:
                logger.warning(f"Shape for {k} ({list(v.shape)}) does not match model weights "
                               f"({list(model_state_dict[k].shape)}), skipping weights.")

    missing_params = set(model_state_dict.keys()) - set(clean_dict.keys())
    for k in missing_params:
        logger.warning(f"Weights for {k} are missing in state dict, using random initialization.")
        clean_dict[k] = model_state_dict[k]

    state_dict = clean_dict

    return state_dict


def checkpoint_filter_fn_tim(state_dict, model: TerraMindTiM) -> dict:
    """Manually filter pre-trained weights for TerraMind ViT to enable strict weight loading."""

    model_state_dict = model.state_dict()
    clean_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                clean_dict[k] = v
            else:
                logger.warning(f"Shape for {k} ({list(v.shape)}) does not match model weights "
                               f"({list(model_state_dict[k].shape)}), skipping weights.")
        if 'sampler.model.' + k in model_state_dict:
            # Copy weights for MAE model for TiM
            mae_k = 'sampler.model.' + k
            if v.shape == model_state_dict[mae_k].shape:
                clean_dict[mae_k] = v
            else:
                raise ValueError(f"Shape for {k} ({list(v.shape)}) does not match MAE model weights "
                                 f"({list(model_state_dict[mae_k].shape)}). Cannot run chain of thoughts without MAE.")

    missing_params = set(model_state_dict.keys()) - set(clean_dict.keys())
    for k in missing_params:
        if k.startswith('sampler.model.'):
            raise ValueError(f"Weights for {k} are missing in state dict, cannot run chain of thoughts without MAE.")
        logger.warning(f"Weights for {k} are missing in state dict, using random initialization.")
        clean_dict[k] = model_state_dict[k]

    state_dict = clean_dict

    return state_dict


def checkpoint_filter_fn_generate(state_dict, model: TerraMindGeneration) -> dict:
    """Manually filter pre-trained weights for TerraMind to enable strict weight loading."""

    model_state_dict = model.state_dict()
    clean_dict = {}
    for k, v in state_dict.items():
        mae_k = 'sampler.model.' + k
        if mae_k in model_state_dict:
            if v.shape == model_state_dict[mae_k].shape:
                clean_dict[mae_k] = v
            else:
                logger.warning(f"Shape for {k} ({list(v.shape)}) does not match model weights "
                               f"({list(model_state_dict[mae_k].shape)}), skipping weights.")

    missing_params = set(model_state_dict.keys()) - set(clean_dict.keys())
    for k in missing_params:
        if not k.startswith('tokenizer'):
            # No warning because tokenizer weights are loaded separately
            logger.warning(f"Weights for {k} are missing in state dict, using random initialization.")
        clean_dict[k] = model_state_dict[k]

    state_dict = clean_dict

    return state_dict


def build_terrammind_vit(
        variant: str = None,
        pretrained: bool = False,
        ckpt_path: str | None = None,
        bands: dict[str, list] | None = None,
        pretrained_bands: dict[str, list] | None = None,
        **kwargs):

    model = TerraMindViT(**kwargs)

    if ckpt_path is not None:
        # Load model from checkpoint
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        loaded_keys = model.load_state_dict(state_dict, strict=False)
        if loaded_keys.missing_keys:
            logger.warning(f"Missing keys in ckpt_path {ckpt_path}: {loaded_keys.missing_keys}")
        if loaded_keys.unexpected_keys:
            logger.warning(f"Missing keys in ckpt_path {ckpt_path}: {loaded_keys.missing_keys}")

    elif pretrained:
        # Load model from Hugging Face
        state_dict_file = hf_hub_download(repo_id=pretrained_weights[variant]['hf_hub_id'],
                                          filename=pretrained_weights[variant]['hf_hub_filename'])
        state_dict = torch.load(state_dict_file, map_location="cpu", weights_only=True)
        state_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(state_dict, strict=True)

    if bands is not None:
        model = select_modality_patch_embed_weights(model, bands, pretrained_bands)

    return model


def build_terrammind_mae(
        variant: str = None,
        pretrained: bool = False,
        ckpt_path: str | None = None,
        **kwargs):

    model = TerraMind(**kwargs)

    if ckpt_path is not None:
        # Load model from checkpoint
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        loaded_keys = model.load_state_dict(state_dict, strict=False)
        if loaded_keys.missing_keys:
            logger.warning(f"Missing keys in ckpt_path {ckpt_path}: {loaded_keys.missing_keys}")
        if loaded_keys.unexpected_keys:
            logger.warning(f"Missing keys in ckpt_path {ckpt_path}: {loaded_keys.missing_keys}")

    elif pretrained:
        # Load model from Hugging Face
        state_dict_file = hf_hub_download(repo_id=pretrained_weights[variant]['hf_hub_id'],
                                          filename=pretrained_weights[variant]['hf_hub_filename'])
        state_dict = torch.load(state_dict_file, map_location="cpu", weights_only=True)
        state_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(state_dict, strict=True)

    return model


def build_terrammind_tim(
        variant: str = None,
        pretrained: bool = False,
        ckpt_path: str | None = None,
        bands: dict[str, list] | None = None,
        pretrained_bands: dict[str, list] | None = None,
        **kwargs):

    model = TerraMindTiM(**kwargs)

    if ckpt_path is not None:
        # Load model from checkpoint
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        loaded_keys = model.load_state_dict(state_dict, strict=False)
        if loaded_keys.missing_keys:
            logger.warning(f"Missing keys in ckpt_path {ckpt_path}: {loaded_keys.missing_keys}")
        if loaded_keys.unexpected_keys:
            logger.warning(f"Missing keys in ckpt_path {ckpt_path}: {loaded_keys.missing_keys}")

    elif pretrained:
        # Load model from Hugging Face
        state_dict_file = hf_hub_download(repo_id=pretrained_weights[variant]['hf_hub_id'],
                                          filename=pretrained_weights[variant]['hf_hub_filename'])
        state_dict = torch.load(state_dict_file, map_location="cpu", weights_only=True)
        state_dict = checkpoint_filter_fn_tim(state_dict, model)
        model.load_state_dict(state_dict, strict=True)

    if bands is not None:
        raise NotImplementedError('Bands cannot be adapted because the MAE model for TiM is not trained.')
        # TODO: Test if possible for TiM model, maybe with a subset of input modalities for TiM.
        model = select_modality_patch_embed_weights(model, bands, pretrained_bands)

    return model


def build_terrammind_generate(
        variant: str = None,
        pretrained: bool = False,
        ckpt_path: str | None = None,
        **kwargs):

    model = TerraMindGeneration(pretrained=pretrained, **kwargs)

    if ckpt_path is not None:
        # Load model from checkpoint
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        loaded_keys = model.load_state_dict(state_dict, strict=False)
        if loaded_keys.missing_keys:
            logger.warning(f"Missing keys in ckpt_path {ckpt_path}: {loaded_keys.missing_keys}")
        if loaded_keys.unexpected_keys:
            logger.warning(f"Missing keys in ckpt_path {ckpt_path}: {loaded_keys.missing_keys}")

    elif pretrained:
        # Load model from Hugging Face
        state_dict_file = hf_hub_download(repo_id=pretrained_weights[variant]['hf_hub_id'],
                                          filename=pretrained_weights[variant]['hf_hub_filename'])
        state_dict = torch.load(state_dict_file, map_location="cpu", weights_only=True)
        state_dict = checkpoint_filter_fn_generate(state_dict, model)
        model.load_state_dict(state_dict, strict=True)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def terramind_v1_base(**kwargs):
    model = build_terrammind_vit(
        variant='terramind_v1_base',
        encoder_depth=12,
        dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        pretrained_bands=PRETRAINED_BANDS,
        **kwargs
    )
    return model

      
@TERRATORCH_BACKBONE_REGISTRY.register
def terramind_v1_base_tim(**kwargs):
    model = build_terrammind_tim(
        variant='terramind_v1_base',
        encoder_depth=12,
        decoder_depth=12,
        dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        pretrained_bands=PRETRAINED_BANDS,
        **kwargs
    )
    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def terramind_v01_base(**kwargs):
    model = build_terrammind_vit(
        variant='terramind_v01_base',
        encoder_depth=12,
        dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        pretrained_bands={'untok_sen2l2a@224': PRETRAINED_BANDS['untok_sen2l2a@224']},
        **kwargs
    )
    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def terramind_v1_large(**kwargs):
    model = build_terrammind_vit(
        variant='terramind_v1_large',
        encoder_depth=24,
        dim=1024,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        pretrained_bands=PRETRAINED_BANDS,
        **kwargs
    )
    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def terramind_v1_large_tim(**kwargs):
    model = build_terrammind_tim(
        variant='terramind_v1_large',
        encoder_depth=24,
        decoder_depth=24,
        dim=1024,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        pretrained_bands=PRETRAINED_BANDS,
        **kwargs
    )
    return model


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terramind_v1_base_mae(**kwargs):
    model = build_terrammind_mae(
        variant='terramind_v1_base',
        encoder_depth=12,
        decoder_depth=12,
        dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        pretrained_bands=PRETRAINED_BANDS,
        **kwargs
    )
    return model


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terramind_v1_large_mae(**kwargs):
    model = build_terrammind_mae(
        variant='terramind_v1_large',
        encoder_depth=24,
        decoder_depth=24,
        dim=1024,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        pretrained_bands=PRETRAINED_BANDS,
        **kwargs
    )
    return model


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terramind_v01_base_generate(**kwargs):
    model = build_terrammind_generate(
        variant='terramind_v01_base',
        encoder_depth=12,
        decoder_depth=12,
        dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        pretraining_mean=v01_pretraining_mean,
        pretraining_std=v01_pretraining_std,
        **kwargs
    )
    return model


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terramind_v1_base_generate(**kwargs):
    model = build_terrammind_generate(
        variant='terramind_v1_base',
        encoder_depth=12,
        decoder_depth=12,
        dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        pretraining_mean=v1_pretraining_mean,
        pretraining_std=v1_pretraining_std,
        **kwargs
    )
    return model


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terramind_v1_large_generate(**kwargs):
    model = build_terrammind_generate(
        variant='terramind_v1_large',
        encoder_depth=24,
        decoder_depth=24,
        dim=1024,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        pretraining_mean=v1_pretraining_mean,
        pretraining_std=v1_pretraining_std,
        **kwargs
    )
    return model

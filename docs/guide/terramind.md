# TerraMind

The TerraMind models are fully integrated into TerraTorch and support standard fine-tuning, TiM tunig and generations.
The Tokenizers are registered in the `FULL_MODEL_REGISTRY` and can be used in python scripts for encoding or reconstructions.

More information at https://ibm.github.io/terramind/.

If you have an problem, please create an issue in our [GitHub repo](https://github.com/IBM/terramind).

## Model versions

Here is an overview of all available 

TerraMind 1.0 backbones (`BACKBONE_REGISTRY`):
```text
terramind_v1_base
terramind_v1_large
terramind_v1_base_tim
terramind_v1_large_tim
```

TerraMind 1.0 generative models (`FULL_MODEL_REGISTRY`):
```text
terramind_v1_base_generate
terramind_v1_large_generate
```

TerraMind 1.0 Tokenziers (`FULL_MODEL_REGISTRY`):
```text
terramind_v1_tokenizer_s2l2a
terramind_v1_tokenizer_s1rtc
terramind_v1_tokenizer_s1grd
terramind_v1_tokenizer_dem
terramind_v1_tokenizer_lulc
terramind_v1_tokenizer_ndvi
```

The backbones support the following pre-trained modalities as raw inputs: `S2L1C`, `S2L2A`, `RGB`, `S1GRD`, `S1RTC`, `DEM`, and `Coordinates`.
The RGB patch embedding was pre-trained on S2 RGB inputs (0-255).

Additionally, the following inputs can be used via Tokenizers (not recommended for fine-tuning, see section [new modalities](#new-modalities)): `LULC` and `NDVI`.

The following modalities can be used as `tim_modalities` or `output_modalities`:  `S2L2A`, `S1GRD`, `S1RTC`, `DEM`, `LULC`, `NDVI`, and `Coordinates`.

!!! note
    Support for coordinates is added in `terratorch==1.1`. Make sure to use the most recent version or install from `main`.


??? note "Experimental models (v0.1)"
    TerraMind v0.1 models (refert to TM-B-single in the paper):
    ```yaml
    terramind_v01_base
    terramind_v01_base_tim
    terramind_v01_base_generate
    ```
    This model is experimental and not publicly accessible. It supports only `S2L2A` as raw input modality and `Captions` as input and output modality.

## Usage

### Fine-tuning

TerraMind requires the additional parameter `modalities` to define the input types. You can pass it to `BACKBONE_FACTORY` or define it as `model_args`.
Additionally, you can define how the encoder output of multiple modalities are merged. By default, averages over all image modalities of each patch.  

```python
from terratorch.registry import BACKBONE_REGISTRY

model = BACKBONE_REGISTRY.build("terramind_v1_base", pretrained=True, modalities=["S2L2A", "S1GRD"])
```

```yaml
    model_factory: EncoderDecoderFactory
    model_args:
      backbone: terramind_v1_base
      backbone_pretrained: True
      backbone_modalities:  # List your raw inputs
        - S2L2A
        - S1GRD
      backbone_merge_method: mean  # mean (default), max, and concat working in TerraTorch fine-tuning (for custom scripts: dict, None).
```

The output of the backbone is a list of tensors, specically, the outputs of each model layer. With the shape \[batch, token, embedding]. Note that Terramind v1 does not use a CLS token.
If you want to combine the model with a hierarchical decoder, such as UNetDecoder or UperNetDecoder, use the following necks:

```yaml
necks:
  - name: ReshapeTokensToImage
    remove_cls_token: False  # Need to be False because of missing CLS token in TerraMind
  - name: SelectIndices  # Select layers used in the UNet decoder
    indices: [2, 5, 8, 11]  # Base model
#            indices: [5, 11, 17, 23]  # Large model
  - name: LearnedInterpolateToPyramidal  # Upscale outputs to UNet input size
```

You can find an example for fine-tuning TerraMind with multi-modal inputs in this [notebook](https://github.com/IBM/terramind/blob/main/notebooks/terramind_v1_base_sen1floods11.ipynb) and this [config](https://github.com/IBM/terramind/blob/main/configs/terramind_v1_base_sen1floods11.yaml) file.

### Subset of input bands

If you only want to use a subset of the original bands using in pre-training, you can specify them in a dict with the `bands` parameter. 
The dict key is the name of the modality and the values are the name of the bands you want to use. 
If you have new unseen bands, give them a new name and TerraTorch initalizes the patch embedding of these channels with new random weights.
If you don't specify the bands of a modality, all bands are expected. 

Here is an example, where we want to use the Sentinel-2 embeddings for Landsat-8 data. `S2L1C` still needs to be part of the modality name so that TerraTorch knows which patch embeddings to load (the name can be quite flexible, e.g., upper or lower case works).
```yaml
backbone_modalities:
  - S2L1C_L8
  - S1GRD
backbone_bands:
  S2L1C_L8:  # Modality name has to match
    - COASTAL_AEROSOL
    - BLUE
    - GREEN
    - RED
    - NIR_NARROW
    - SWIR_1
    - SWIR_2
    - PANCHROMATIC  # New band
    - CIRRUS
    - THERMAL_1  # New band
    - THERMAL_2  # New band
```

The name of the pre-trained bands are [here](https://github.com/IBM/terratorch/blob/53768e684a50e3f7e37d654f499dcccb4373940b/terratorch/models/backbones/terramind/model/terramind_register.py#L77) specified:
```yaml
S2L1C / S2L2A:
  - COASTAL_AEROSOL
  - BLUE
  - GREEN
  - RED
  - RED_EDGE_1
  - RED_EDGE_2
  - RED_EDGE_3
  - NIR_BROAD
  - NIR_NARROW
  - WATER_VAPOR
  - CIRRUS  # Only in S2L1C
  - SWIR_1
  - SWIR_2

S1GRD / S1RTC:
  - VV
  - VH

RGB:
  - BLUE
  - GREEN
  - RED
  
DEM: 
  - DEM
```

## New modalities

You might want to use input modalities which are not used with raw inputs during pre-training. 
Therefore, you can define a new patch embedding by providing a dict as an input in the modality list, which specifies the new name and the number of input channels.
You can also use it if you want to fine-tune the model with NDVI or LULC data, which would otherwise use the TerraMind tokenizers, which increases the model size quite a lot.
Here is an example that reuses the S-2 patch embedding but initalizes a new patch emebdding for NDVI data and a completly new modality:  
```yaml
backbone_modalities:
  - S2L2A
  - NDVI: 1
  - PLANET: 6
```

Note that in our experience it is always better to reuse a patch embedding, even with other satellite (e.g. using S2 or RGB modalities for optical ones).
The model can more quickly adapt to different data scaling etc. compared to learning it from scratch.
Note that with the current implementation, you cannot reuse a specific patch embedding mulitple times. However, you could also do something like this:
```yaml
backbone_modalities:
  - S2L2A
  - NDVI: 1
  - S2L1C_PLANET # Reuse S2L1C patch embedding
backbone_bands:
  S2L1C_PLANET:  
    - BLUE
    - GREEN
    - RED
    - ...  
```

If you use a single new modality, you can also define it via: 
```yaml
    model_args:
      backbone: terramind_v1_base
      backbone_pretrained: True
      backbone_modalities: [] # Set list to None or empty list
      backbone_in_chans: 3  # Define in channels of your modality, defaults to 3.
```
The new modality is called `image` and the patch embedding will be randomly initialized. 
You can pass the data to the model as a PyTorch tensor or in a dict `{"image": tensor}`. 


## Thinking-in-Modalities (TiM)

During fine-tuning or inference, TerraMind can pause for a moment, imagine a helpful but absent layer, append the imagined tokens to its own input sequence, and then lets the fine-tuned encoder continue to improve its own performance. 
Because the imagination lives in token space, the approach avoids the heavy diffusion decoding that full image synthesis would require. 
So, TerraMind can generate any missing modality as an intermediate step — an ability we call Thinking in Modalities (TiM). 
We refer to the [paper](https://arxiv.org/pdf/2504.11171) for details.

!!! warning "Important"
    Thinking-in-Modalities only works with pre-trained modalities as inputs and with all input bands.
    The Encoder-Decoder model generating the TiM modalities is never trained to ensure efficient fine-tuning.
    Therefore, it cannot adapt to other inputs such as new modalities or subsets of the pre-trained bands.
    If this is the case for you, you cannot use the TiM model.

You can use TiM models by just adding the `_tim` suffix to the model name and specifing the `tim_modalities`.

```python
from terratorch.registry import BACKBONE_REGISTRY

model = BACKBONE_REGISTRY.build("terramind_v1_base_tim", pretrained=True, modalities=["S2L2A", "S1GRD"], tim_modalities=["LULC"])
```

```yaml
    model_factory: EncoderDecoderFactory
    model_args:
      backbone: terramind_v1_base_tim
      backbone_pretrained: True
      backbone_modalities:
        - S2L2A
      backbone_tim_modalities:
        - LULC
```

Here is a [config](https://github.com/IBM/terramind/blob/main/configs/terramind_v1_base_tim_lulc_sen1floods11.yaml) example for fine-tuning. 

## Generation

You can perform any-to-any generation with the `terramind_v1_base_generate` and `terramind_v1_large_generate` models:

```python
from terratorch.registry import BACKBONE_REGISTRY

model = BACKBONE_REGISTRY.build(
    'terramind_v1_base_generate',
    modalities=['S2L2A'],  # Input modalities.
    output_modalities=['S2L2A', 'S1GRD', 'LULC'],  # List any number of output modalities for generation.
    pretrained=True,
    standardize=True,  # Apply standardization on input and output.
)
```

You don't need to apply standardization on the input or output, if you set `standardize=True`. This avoids error from using wrong values.
Alternatively, you can find the standardization values from pre-training [here](https://github.com/IBM/terratorch/blob/53768e684a50e3f7e37d654f499dcccb4373940b/terratorch/models/backbones/terramind/model/terramind_register.py#L130).

We provide demos in [terramind_generation.ipynb](https://github.com/IBM/terramind/blob/main/notebooks/terramind_generation.ipynb) and [any_to_any_generation.ipynb](https://github.com/IBM/terramind/blob/main/notebooks/terramind_any_to_any_generation.ipynb).
You can also use TerraTorch's large-tile-inference to generate images of larger scenes, which is demonstrated in [large_tile_generation.ipynb](https://github.com/IBM/terramind/blob/main/notebooks/large_tile_generation.ipynb).

## Tokenizer

The tokenizers are integrated into the TerraTorch as well can be initialized with: 

```python
from terratorch.registry import FULL_MODEL_REGISTRY

model = FULL_MODEL_REGISTRY.build('terramind_v1_tokenizer_s2l2a', pretrained=True)
model = FULL_MODEL_REGISTRY.build('terramind_v1_tokenizer_s1rtc', pretrained=True)
model = FULL_MODEL_REGISTRY.build('terramind_v1_tokenizer_s1grd', pretrained=True)
model = FULL_MODEL_REGISTRY.build('terramind_v1_tokenizer_dem', pretrained=True)
model = FULL_MODEL_REGISTRY.build('terramind_v1_tokenizer_ndvi', pretrained=True)
model = FULL_MODEL_REGISTRY.build('terramind_v1_tokenizer_lulc', pretrained=True)
```

You can create reconstructions with:
```python
with torch.no_grad():
    # Run full reconstruction
    reconstruction = model(normalized_input)

# Alternatively run only the encoder or decoder
with torch.no_grad():
    # Tokenize
    _, _, tokens = model.encode(normalized_input)
    
    # Detokenize
    reconstruction = model.decode_tokens(tokens)
```

We provide an example in [terramind_tokenizer_reconstruction.ipynb](https://github.com/IBM/terramind/blob/main/notebooks/terramind_tokenizer_reconstruction.ipynb).


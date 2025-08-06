# Prithvi EO Models

Code examples and more details are available in the [Prithvi-EO 2.0 GitHub repo](https://github.com/NASA-IMPACT/Prithvi-EO-2.0).

---

## Model Versions

Available model names:

```text
prithvi_eo_v1_100
prithvi_eo_v2_300
prithvi_eo_v2_600
prithvi_eo_v2_300_tl
prithvi_eo_v2_600_tl
```

Models with the `_tl` suffix support additional time and location metadata inputs. See [Metadata Inputs](#metadata-inputs).

These models were pre-trained on the following bands:
`BLUE`, `GREEN`, `RED`, `NIR_NARROW`, `SWIR_1`, `SWIR_2`

You can optionally specify a subset or new band names using a list. Unknown bands will have their patch embeddings initialized with random weights.

```python
from terratorch.registry import BACKBONE_REGISTRY

model = BACKBONE_REGISTRY.build(
    "prithvi_eo_v2_300", pretrained=True,
    bands=["RED", "GREEN", "BLUE"],  # Optional: specify bands
    num_frames=1,  # Optional: number of time steps (default: 1)
)
```

```yaml
model_factory: EncoderDecoderFactory
model_args:
  backbone: prithvi_eo_v2_300_tl
  backbone_pretrained: True
```

Backbone output: list of tensors with shape `[batch, token, embedding]` (includes CLS token).

For hierarchical decoders such as UNet, use the following necks:

```yaml
necks:
  - name: ReshapeTokensToImage
  - name: SelectIndices
    # indices: [2, 5, 8, 11]  # 100M model
    indices: [5, 11, 17, 23]  # 300M model
    # indices: [7, 15, 23, 31]  # 600M model
  - name: LearnedInterpolateToPyramidal
```

Full example: [burn\_scars.yaml](https://github.com/IBM/terratorch/blob/main/examples/confs/burn_scars.yaml)

---

## Metadata Inputs

Metadata is optional and supported only by `_tl` models. During pre-training, metadata was dropped in 10% of the samples, so the model is robust to missing metadata.

Specify metadata usage in the config:

```yaml
coords_encoding:
  - time
  - location
```

During inference, pass the metadata inputs like so:

```python
output = model(
    data_tensor,
    temporal_coords=time_data,  # Shape: [B, T, 2] — year, day of year (0–364)
    location_coords=loc_data,   # Shape: [B, 2] — longitude, latitude
)
```

Metadata example using `pandas` and `torch`:

```python
date = pd.to_datetime('2024-06-15')
time_data = torch.Tensor([[[date.year, date.dayofyear - 1]]], device=device)  # [1, 1, 2]
loc_data = torch.Tensor([[47.309, 8.544]], device=device)  # [1, 2]
```

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

## Usage

You can build the backbone using the `BACKBONE_REGISTRY`.  
Optionally, specify a subset or new band names using a list. Unknown bands will have their patch embeddings initialized with random weights.
For multi-temporal task, specify the number of input frames.

```python
from terratorch.registry import BACKBONE_REGISTRY

model = BACKBONE_REGISTRY.build(
    "prithvi_eo_v2_300", pretrained=True,
    bands=["RED", "GREEN", "BLUE", "NEW"],  # Optional, specify bands
    num_frames=1,  # Optional, number of time steps (default: 1)
)
```

### Fine-tuning

Use Prithvi EO as a backbone in TerraTorch's EncoderDecoderFactory:
=== "YAML"
    ```yaml
    model:
      class_path: terratorch.tasks.SemanticSegmentationTask
      init_args:
        model_factory: EncoderDecoderFactory
        model_args:
          backbone: prithvi_eo_v2_300_tl
          backbone_pretrained: True
          backbone_bands: [RED, GREEN, BLUE, NEW]  # Optional
          backbone_num_frames: 1  # Optional
          ...
    ```

=== "Python"
    ```python
    task = terratorch.tasks.SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory", 
        model_args={
            "backbone": "prithvi_eo_v2_300_tl",
            "backbone_pretrained": True,
            "backbone_bands": ["RED", "GREEN", "BLUE", "NEW"],  # Optional
            "backbone_num_frames": 1,  # Optional
            ...
        },
        ...
    )
    ```


Backbone output: list of tensors with shape `[batch, token, embedding]` (includes CLS token).

For hierarchical decoders such as UNet, use the following necks:

=== "YAML"
    ```yaml
    model_args:    
      ...
      necks:
        - name: ReshapeTokensToImage  # Reshape 1D tokens to 2D grid
        - name: SelectIndices  # Select three intermediate layer outputs and the final one
          # indices: [2, 5, 8, 11]  # 100M model
          indices: [5, 11, 17, 23]  # 300M model
          # indices: [7, 15, 23, 31]  # 600M model
        - name: LearnedInterpolateToPyramidal  # Upscale outputs for hierarchical decoders
        ...
    ```

=== "Python"
    ```python
    model_args={
        ...        
        "necks": [
            {"name": "ReshapeTokensToImage", 
             "remove_cls_token": False}
            {"name": "SelectIndices", 
            #  "indices": [2, 5, 8, 11]}, # 100M model
            "indices": [5, 11, 17, 23]}, # 300M model
            # "indices": [7, 15, 23, 31]}, # 600M model
            {"name": "LearnedInterpolateToPyramidal"}
        ]
        ...
    } 
    ```

Full example: [burn\_scars.yaml](https://github.com/IBM/terratorch/blob/main/examples/confs/burn_scars.yaml)

---

### Metadata Inputs

Metadata is optional and supported only by `_tl` models. During pre-training, metadata was dropped in 10% of the samples, so the model is robust to missing metadata.

Specify metadata usage with:

=== "YAML"
    ```yaml
    backbone_coords_encoding:
      - time
      - location
    ```

=== "Python"
    ```python
    model_args={
        ...
        "backbone_coords_encoding": [
            "time",
            "location",
        ],
        ...
    }
    ```

=== "Backbone registry"
    ```python
    model = BACKBONE_REGISTRY.build("prithvi_eo_v2_300", pretrained=True, 
                                    coords_encoding=[
                                        "time", 
                                        "location", 
                                    ])
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

!!! warning
    Metadata is currently not supported with the generic data modules. You are required to use a custom data module and dataset class, e.g., by modifying one listed in [Datamodules](..%2Fpackage%2Fdatamodules.md). 

The TerraTorch task automatically passes all additional values in the batch dict to the model.
In your custom dataset class, add the metadata as additional values to the dict:
```python
def __getitem__(idx):
    ...
    # Load metadata from 
    date: str = '2024-06-15'  # Example for a date
    lon, lat = 47.309, 8.544  # Example for a location

    date = pd.to_datetime(date)
    time_data = torch.Tensor([[date.year, date.dayofyear - 1]])  # Shape [T, 2]
    loc_data = torch.Tensor([lon, lat])  # Shape [2]
    ...
    sample = {
        "image": data,
        "mask": mask,
        "temporal_coords": time_data,
        "location_coords": loc_data
    }
    return sample
```

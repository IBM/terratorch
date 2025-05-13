
# Inference

You can run inference with TerraTorch by providing the path to an input folder and output directory. 
You can do this directly via the CLI with:
```shell
terratorch predict -c config.yaml --ckpt_path path/to/model/checkpoint.ckpt --data.init_args.predict_data_root input/folder/ --predict_output_dir output/folder/ 
```

This approach works only for supported data modules like the TerraTorch `GenericNonGeoSegmentationDataModule`. 
Alternatively, define the parameters in the config file:

```yaml
data:
  class_path: terratorch.datamodules.GenericNonGeoSegmentationDataModule
  init_args:
    ...
    predict_data_root: path/to/input/files/
```

## Tiled inference via CLI

TerraTorch supports a tiled inference that splits up a tile into smaller patches. With this approach, you can run a model on very large tiles like a 10k x 10k pixel Sentinel-2 tile. 

Define the tiled inference parameters in the yaml config like the following:
```yaml
model:
  class_path: terratorch.tasks.SemanticSegmentationTask
  init_args:
    ...
    tiled_inference_parameters:
      h_crop: 512
      h_stride: 496
      w_crop: 512
      w_stride: 496
      batch_size: 16  # default
      average_patches: true  # default
      verbose: false # default
```

Next, you can run:
```shell
terratorch predict -c config.yaml --ckpt_path path/to/model/checkpoint.ckpt --data.init_args.predict_data_root input/folder/ --predict_output_dir output/folder/
```

PyTorch Lightning load each input tile to the GPU. This can result in out-of-memory errors with very large tiles like 100k x 100k pixels. In this case, use python to run the tiled inference.

## Tiled inference via Python

You can use TerraTorch to run tiled inference in a python script like the following:

```python
import torch
from terratorch.tasks import SemanticSegmentationTask
from terratorch.tasks.tiled_inference import TiledInferenceParameters, tiled_inference

# Init an TerraTorch task, e.g. for semantic segmentation
model = SemanticSegmentationTask.load_from_checkpoint(
    ckpt_path,  # Pass the checkpoint path
    model_factory="EncoderDecoderFactory",
    model_args=model_args,  # Pass your model args
)


tiled_inference_parameters = TiledInferenceParameters(
    h_crop=256, 
    h_stride=240, 
    w_crop=256, 
    w_stride=240, 
    average_patches=True, 
    batch_size=16, 
    verbose=True
)

# Apply your standardization values to the input tile
input = (input - means[:, None, None]) / stds[:, None, None]
# Create input tensor with shape [B, C, H, W] on CPU
input = torch.tensor(input, dtype=torch.float, device='cpu').unsqueeze(0)

# Inference wrapper for TerraTorch task model
def model_forward(x,  **kwargs):
    return model(x, **kwargs).output

# Run tiled inference (data is loaded automatically to GPU)
pred = tiled_inference(model_forward, input, num_classes, tiled_inference_parameters)

# Remove batch dim and compute segmentation map
pred = pred.squeeze(0).argmax(dim=0)
```

The example assumes a numpy array as input and a `TerraTorch.task.SemanticSegmentationTask` as model. 
You can easily modify the script by setting `num_classes=1` for regression tasks or using a custom PyTorch model instead of `model_forward` for `tiled_inference`.

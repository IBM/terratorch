# Quick start

## Setup

Let's start by setting up an environment and installing TerraTorch.

!!! Tip
    You can quickly setup a new virtual environment by running:
    ```shell
    python -m venv venv  # using python 3.10 or newer
    pip install --upgrade pip
    pip install terratorch
    ```

### Configuring the environment

TerraTorch is currently tested for Python in `3.10 <= Python <= 3.13`. 

GDAL is required  to read and write TIFF images. It is usually easy to install in Unix/Linux systems, but if it is not your case 
we recommend using a conda environment and installing it with `conda install -c conda-forge gdal`. 

### Installing TerraTorch
For a stable point-release, use `pip install terratorch`.
If you prefer to get the most recent version of the main branch, install the library with `pip install git+https://github.com/IBM/terratorch.git`.

To install as a developer (e.g., to extend the library), fork the [repo](https://github.com/IBM/terratorch) and clone your repo with `git clone https://github.com/<your_username>/terratorch.git`. Then run `pip install -e .`.
We welcome contributions from the community and provide some [guidelines](../about/contributing.md) for you.

## Creating Backbones

You can interact with the library at several levels of abstraction. Each deeper level of abstraction trades off some amount of flexibility for ease of use and configuration.
In the simplest case, we might only want access a backbone and code all the rest ourselves. In this case, we can simply use the library as a backbone factory:

```python title="Instantiating a Prithvi backbone"
from terratorch import BACKBONE_REGISTRY

# Find available Prithvi models
print([model_name for model_name in BACKBONE_REGISTRY if "terratorch_prithvi" in model_name])
>>> ['terratorch_prithvi_eo_tiny', 'terratorch_prithvi_eo_v1_100', 'terratorch_prithvi_eo_v2_300', 'terratorch_prithvi_eo_v2_600', 'terratorch_prithvi_eo_v2_300_tl', 'terratorch_prithvi_eo_v2_600_tl']

# Show all models with list(BACKBONE_REGISTRY)

# check a model is in the registry
"terratorch_prithvi_eo_v2_300" in BACKBONE_REGISTRY
>>> True

# without the prefix, all internal registries will be searched until the first match is found
"prithvi_eo_v1_100" in BACKBONE_REGISTRY
>>> True

# instantiate your desired model
# the backbone registry prefix (e.g., `terratorch` or `timm`) is optional
# in this case, the underlying registry is terratorch.
model = BACKBONE_REGISTRY.build("prithvi_eo_v1_100", pretrained=True)

# instantiate your model with more options, for instance, passing input bands or weights from your own file
model = BACKBONE_REGISTRY.build(
    "prithvi_eo_v2_300", bands=["RED", "GREEN", "BLUE"], num_frames=1, pretrained=True, ckpt_path='path/to/model.pt'
)

# Rest of your PyTorch / PyTorchLightning code
...
```

Internally, TerraTorch maintains several registries for components such as backbones or decoders. The top-level `BACKBONE_REGISTRY` collects all of them.

The name passed to `build` is used to find the appropriate model constructor, which will be the first model from the first registry found with that name.

To explicitly determine the registry that will build the model, you may prepend a prefix such as `timm_` to the model name. In this case, the `timm` model registry will be exclusively searched for the model.

## Creating a full model

We also provide a model factory for a task-specific model that combines a backbone with a decoder:

```python title="Building a full model, with task-specific decoder"
from terratorch.models import EncoderDecoderFactory

model_factory = EncoderDecoderFactory()

# Let's build a segmentation model
# Parameters prefixed with backbone_ get passed to the backbone
# Parameters prefixed with decoder_ get passed to the decoder
# Parameters prefixed with head_ get passed to the head

model = model_factory.build_model(
    task="segmentation",
    backbone="prithvi_eo_v2_300",
    backbone_pretrained=True,
    backbone_bands=[
        "BLUE",
        "GREEN",
        "RED",
        "NIR_NARROW",
        "SWIR_1",
        "SWIR_2",
    ],
    necks=[
        {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
        {"name": "ReshapeTokensToImage"},
        {"name": "LearnedInterpolateToPyramidal"}
    ],
    decoder="UNetDecoder",
    decoder_channels=[512, 256, 128, 64],
    head_dropout=0.1,
    num_classes=4,
)

# Rest of your PyTorch / PyTorchLightning code
...
```

You might wonder what `necks` are. Different model architectures like CNNs or ViTs return outputs in different formats while different decoders expect other input shapes. 
Therefore, we use necks to reshape the backbone output into the correct format for the decoder input.
In this case, we select intermediate model outputs for the UNet, reshape the 1D tokens of Prithvi into a 2D grid, and upscale the intermediate outputs as the UNet decoder expects hierarchical inputs.  
You can simply use a backbone-neck-decoder combination from one of the provided examples or check the [necks](../package/necks.md) page in the user guide for more details.

## Training with Lightning Tasks

At the highest level of abstraction, you can directly obtain a LightningModule ready to be trained.
We simply need to pass the model factory and the model arguments to the task.
Passed to a Lightning Trainer, the task executes the training, validation, and testing steps. 

```python title="Building a full pixel-wise regression task"
from terratorch.tasks import PixelwiseRegressionTask

model_args = dict(
  backbone="prithvi_eo_v2_300",
  backbone_pretrained=True,
  backbone_num_frames=1,
  backbone_bands=[
      "BLUE",
      "GREEN",
      "RED",
      "NIR_NARROW",
      "SWIR_1",
      "SWIR_2",
  ],
    necks=[
        {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
        {"name": "ReshapeTokensToImage"},
        {"name": "LearnedInterpolateToPyramidal"}
    ],
    decoder="UNetDecoder",
    decoder_channels=[512, 256, 128, 64],
    head_dropout=0.1,
)

task = PixelwiseRegressionTask(
    model_factory="EncoderDecoderFactory",
    model_args=model_args,
    loss="rmse",
    lr=1e-4,
    ignore_index=-1,
    optimizer="AdamW",
    optimizer_hparams={"weight_decay": 0.05},
)

# Pass this LightningModule to a Lightning Trainer, together with some LightningDataModule
...
```


Alternatively, all the process can be summarized in configuration files written in YAML format, as seen below.

```yaml title="Configuration file for a semantic segmentation task"
# lightning.pytorch==2.1.1
seed_everything: 0
trainer:
  accelerator: auto  # Lightning automatically selects all available GPUs
  strategy: auto
  devices: auto
  num_nodes: 1
  precision: 16-mixed  # Using half precision speeds up the training
  logger: True  # Lightning uses a Tensorboard logger by default
  callbacks:  # Callbacks are additional steps executed by lightning. 
    - class_path: RichProgressBar
    - class_path: LearningRateMonitor
      init_args:
        logging_interval: epoch
  max_epochs: 100
  log_every_n_steps: 5
  enable_checkpointing: true  # Defaults to true. TerraTorch automatically adds a Checkpoint callback to save the model
  default_root_dir: output/prithvi/experiment  # Define your output folder

data:
  # Define your data module. You can also use one of TerraTorch's generic data modules 
  class_path: terratorch.datamodules.sen1floods11.Sen1Floods11NonGeoDataModule
  init_args:
    batch_size: 16
    num_workers: 8
  dict_kwargs:
    data_root: <path_to_data_root>
    bands:
      - 1
      - 2
      - 3
      - 8
      - 11
      - 12

model:
  class_path: terratorch.tasks.SemanticSegmentationTask
  init_args:
    model_factory: EncoderDecoderFactory
    model_args:
      backbone: prithvi_eo_v2_300
      backbone_img_size: 512
      backbone_pretrained: True
      backbone_bands:
        - BLUE
        - GREEN
        - RED
        - NIR_NARROW
        - SWIR_1
        - SWIR_2
      necks:
        - name: SelectIndices
          indices: [5, 11, 17, 23]
        - name: ReshapeTokensToImage
        - name: LearnedInterpolateToPyramidal
      decoder: UNetDecoder
      decoder_channels: [512, 256, 128, 64]
      head_channel_list: [256]  # Pass a list for an MLP head
      head_dropout: 0.1
      num_classes: 2
    loss: dice
    ignore_index: -1
    freeze_backbone: false  # Full fine-tuning

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 1.e-4
lr_scheduler:
  class_path: ReduceLROnPlateau
  init_args:
    monitor: val/loss
```

To run this training task using the YAML, simply execute:
```sh
terratorch fit --config <path_to_config_file>
```

To test your model on the test set, execute:
```
terratorch test --config  <path_to_config_file> --ckpt_path <path_to_checkpoint_file>
```

For inference, execute:
```sh
terratorch predict -c <path_to_config_file> --ckpt_path <path_to_checkpoint> --predict_output_dir <path_to_output_dir> --data.init_args.predict_data_root <path_to_input_dir> --data.init_args.predict_dataset_bands <all bands in the predicted dataset, e.g. [BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2,0]>
```

**Experimental feature**: Users that want to optimize hyperparameters or repeat best experiment might be interested in the plugin `terratorch-iterate`. For instance, to run TerraTorch Iterate to optimize hyperparameters, one can run: 
```sh
terratorch iterate --hpo --config <path_to_config_file> 
```
You can install the package with `pip install terratorch-iterate` and check the [usage description](https://github.com/IBM/terratorch-iterate?tab=readme-ov-file#usage) for more information.

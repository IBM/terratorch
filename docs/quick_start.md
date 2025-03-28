# Quick start
## Configuring the environment

### Python
TerraTorch is currently tested for Python in `3.10 <= Python <= 3.12`. 

### GDAL
GDAL is required  to read and write TIFF images. It is usually easy to install in Unix/Linux systems, but if it is not your case 
we reccomend using a conda environment and installing it with `conda install -c conda-forge gdal`.

## Installing TerraTorch
For a stable point-release, use `pip install terratorch`.
If you prefer to get the most recent version of the main branch, install the library with `pip install git+https://github.com/IBM/terratorch.git`.

To install as a developer (e.g. to extend the library) clone this repo, and run `pip install -e .` .

## Creating Backbones

You can interact with the library at several levels of abstraction. Each deeper level of abstraction trades off some amount of flexibility for ease of use and configuration.
In the simplest case, we might only want access a backbone and code all the rest ourselves. In this case, we can simply use the library as a backbone factory:

```python title="Instantiating a prithvi backbone"
from terratorch import BACKBONE_REGISTRY

# find available prithvi models
print([model_name for model_name in BACKBONE_REGISTRY if "terratorch_prithvi" in model_name])
>>> ['terratorch_prithvi_eo_tiny', 'terratorch_prithvi_eo_v1_100', 'terratorch_prithvi_eo_v2_300', 'terratorch_prithvi_eo_v2_600', 'terratorch_prithvi_eo_v2_300_tl', 'terratorch_prithvi_eo_v2_600_tl']

# show all models with list(BACKBONE_REGISTRY)

# check a model is in the registry
"terratorch_prithvi_eo_v2_300" in BACKBONE_REGISTRY
>>> True

# without the prefix, all internal registries will be searched until the first match is found
"prithvi_eo_v1_100" in BACKBONE_REGISTRY
>>> True

# instantiate your desired model
# the backbone registry prefix (e.g. `terratorch` or `timm`) is optional
# in this case, the underlying registry is terratorch.
model = BACKBONE_REGISTRY.build("prithvi_eo_v1_100", pretrained=True)

# instantiate your model with more options, for instance, passing weights from your own file
model = BACKBONE_REGISTRY.build(
    "prithvi_eo_v2_300", num_frames=1, ckpt_path='path/to/model.pt'
)
# Rest of your PyTorch / PyTorchLightning code

```

Internally, terratorch maintains several registries for components such as backbones or decoders. The top-level `BACKBONE_REGISTRY` collects all of them.

The name passed to `build` is used to find the appropriate model constructor, which will be the first model from the first registry found with that name.

To explicitly determine the registry that will build the model, you may prepend a prefix such as `timm_` to the model name. In this case, the `timm` model registry will be exclusively searched for the model.

## Directly creating a full model

We also provide a model factory for a task specific model built on one a backbones:

```python title="Building a full model, with task specific decoder"
import terratorch # even though we don't use the import directly, we need it so that the models are available in the timm registry
from terratorch.models import EncoderDecoderFactory
from terratorch.datasets import HLSBands

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
        HLSBands.BLUE,
        HLSBands.GREEN,
        HLSBands.RED,
        HLSBands.NIR_NARROW,
        HLSBands.SWIR_1,
        HLSBands.SWIR_2,
    ],
    necks=[{"name": "SelectIndices", "indices": [-1]},
           {"name": "ReshapeTokensToImage"}],
    decoder="FCNDecoder",
    decoder_channels=128,
    head_dropout=0.1,
    num_classes=4,
)

# Rest of your PyTorch / PyTorchLightning code
.
.
.

```

## Training with Lightning Tasks

At the highest level of abstraction, you can directly obtain a LightningModule ready to be trained.

```python title="Building a full Pixel-Wise Regression task"

model_args = dict(
  backbone="prithvi_eo_v2_300",
  backbone_pretrained=True,
  backbone_num_frames=1,
  backbone_bands=[
      HLSBands.BLUE,
      HLSBands.GREEN,
      HLSBands.RED,
      HLSBands.NIR_NARROW,
      HLSBands.SWIR_1,
      HLSBands.SWIR_2,
  ],
  necks=[{"name": "SelectIndices", "indices": [-1]},
               {"name": "ReshapeTokensToImage"}],
  decoder="FCNDecoder",
  decoder_channels=128,
  head_dropout=0.1
)

task = PixelwiseRegressionTask(
    model_args,
    "EncoderDecoderFactory",
    loss="rmse",
    lr=lr,
    ignore_index=-1,
    optimizer="AdamW",
    optimizer_hparams={"weight_decay": 0.05},
)

# Pass this LightningModule to a Lightning Trainer, together with some LightningDataModule
```
Alternatively, all the process can be summarized in configuration files written in YAML format, as seen below.
```yaml title="Configuration file for a Semantic Segmentation Task"
# lightning.pytorch==2.1.1
seed_everything: 0
trainer:
  accelerator: auto
  strategy: auto
  devices: auto
  num_nodes: 1
  precision: bf16
  logger:
    class_path: TensorBoardLogger
    init_args:
      save_dir: <path_to_experiment_dir>
      name: <experiment_name>
  callbacks:
    - class_path: RichProgressBar
    - class_path: LearningRateMonitor
      init_args:
        logging_interval: epoch

  max_epochs: 200
  check_val_every_n_epoch: 1
  log_every_n_steps: 50
  enable_checkpointing: true
  default_root_dir: <path_to_experiment_dir>
data:
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
      decoder: UperNetDecoder
      decoder_channels: 256
      head_channel_list: [256]
      head_dropout: 0.1
      num_classes: 2
    loss: dice
    ignore_index: -1
    freeze_backbone: false
    freeze_decoder: false    
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 1.e-4
    weight_decay: 0.1
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
```sh
terratorch test --config  <path_to_config_file> --ckpt_path <path_to_checkpoint_file>
```

For inference, execute:
```sh
terratorch predict -c <path_to_config_file> --ckpt_path<path_to_checkpoint> --predict_output_dir <path_to_output_dir> --data.init_args.predict_data_root <path_to_input_dir> --data.init_args.predict_dataset_bands <all bands in the predicted dataset, e.g. [BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2,0]>
```

**Experimental feature**: Users that want to optimize hyperparameters or repeat best experiment might be interest in in terratorch-iterate, a terratorch's plugin. For instance, to run terratorch-iterate to optimize hyperparameters, one can run: 
```sh
terratorch iterate --hpo --config <path_to_config_file> 
```
Please see how to install terratorch-iterate on this [link](https://github.com/IBM/terratorch-iterate?tab=readme-ov-file#installation) and how to use it on this [link](https://github.com/IBM/terratorch-iterate?tab=readme-ov-file#usage).

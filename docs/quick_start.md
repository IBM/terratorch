# Quick start
We suggest using Python==3.10.
To get started, make sure to have [PyTorch](https://pytorch.org/get-started/locally/) >= 2.0.0 and [GDAL](https://gdal.org/index.html) installed. 

Installing GDAL can be quite a complex process. If you don't have GDAL set up on your system, we reccomend using a conda environment and installing it with `conda install -c conda-forge gdal`.

For a stable point-release, use `pip install terratorch`. 
If you prefer to get the most recent version of the main branch, install the library with `pip install git+https://github.com/IBM/terratorch.git`.

To install as a developer (e.g. to extend the library) clone this repo, and run `pip install -e .`.

You can interact with the library at several levels of abstraction. Each deeper level of abstraction trades off some amount of flexibility for ease of use and configuration.

## Creating Prithvi Backbones
In the simplest case, we might only want access to one of the prithvi backbones and code all the rest ourselves. In this case, we can simply use the library as a backbone factory:

```python title="Instantiating a prithvi backbone from timm"
import timm
import terratorch # even though we don't use the import directly, we need it so that the models are available in the timm registry

# find available prithvi models by name
print(timm.list_models("prithvi*"))
# and those with pretrained weights
print(timm.list_pretrained("prithvi*"))

# instantiate your desired model with features_only=True to obtain a backbone
model = timm.create_model(
    "prithvi_vit_100", num_frames=1, pretrained=True, features_only=True
)

# instantiate your model with weights of your own
model = timm.create_model(
    "prithvi_vit_100", num_frames=1, pretrained=True, pretrained_cfg_overlay={"file": "<path to weights>"}, features_only=True
)
# Rest of your PyTorch / PyTorchLightning code

```

## Directly creating a full model
We also provide a model factory for a task specific model built on one of the Prithvi backbones:

```python title="Building a full model, with task specific decoder"
import terratorch # even though we don't use the import directly, we need it so that the models are available in the timm registry
from terratorch.models import PrithviModelFactory
from terratorch.datasets import HLSBands

model_factory = PrithviModelFactory()

# Let's build a segmentation model
# Parameters prefixed with backbone_ get passed to the backbone
# Parameters prefixed with decoder_ get passed to the decoder
# Parameters prefixed with head_ get passed to the head

model = model_factory.build_model(task="segmentation",
        backbone="prithvi_vit_100",
        decoder="FCNDecoder",
        in_channels=6,
        bands=[
            HLSBands.BLUE,
            HLSBands.GREEN,
            HLSBands.RED,
            HLSBands.NIR_NARROW,
            HLSBands.SWIR_1,
            HLSBands.SWIR_2,
        ],
        num_classes=4,
        pretrained=True,
        num_frames=1,
        decoder_channels=128,
        head_dropout=0.2
    )

# Rest of your PyTorch / PyTorchLightning code
```

## Training with Lightning Tasks

At the highest level of abstraction, you can directly obtain a LightningModule ready to be trained. 
We currently provide a semantic segmentation task and a pixel-wise regression task.

```python title="Building a full Pixel-Wise Regression task"
task = PixelwiseRegressionTask(
    model_args,
    prithvi_model_factory.PrithviModelFactory(),
    loss="rmse",
    aux_loss={"fcn_aux_head": 0.4},
    lr=lr,
    ignore_index=-1,
    optimizer=torch.optim.AdamW,
    optimizer_hparams={"weight_decay": 0.05},
    scheduler=OneCycleLR,
    scheduler_hparams={
        "max_lr": lr,
        "epochs": max_epochs,
        "steps_per_epoch": math.ceil(len(datamodule.train_dataset) / batch_size),
        "pct_start": 0.05,
        "interval": "step",
    },
    aux_heads=[
        AuxiliaryHead(
            "fcn_aux_head",
            "FCNDecoder",
            {"decoder_channels": 512, "decoder_in_index": 2, "decoder_num_convs": 2, "head_channel_list": [64]},
        )
    ],
)

# Pass this LightningModule to a Lightning Trainer, together with some LightningDataModule
```

At this level of abstraction, you can also provide a configuration file (see [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli)) with all the details of the training. See an example for semantic segmentation below:

!!! info

    To pass your own path from where to load the weights with the PrithviModelFactory, you can make use of timm's `pretrained_cfg_overlay`.
    E.g. to pass a local path, you can add, under model_args:
    
    ```yaml
    backbone_pretrained_cfg_overlay:
        file: <local_path>
    ```
    Besides `file`, you can also pass `url`, `hf_hub_id`, amongst others. Check timm's documentation for full details.

```yaml title="Configuration file for a Semantic Segmentation Task"
# lightning.pytorch==2.1.1
seed_everything: 0
trainer:
  accelerator: auto
  strategy: auto
  devices: auto
  num_nodes: 1
  precision: 16-mixed
  logger: True # will use tensorboardlogger
  callbacks:
    - class_path: RichProgressBar
    - class_path: LearningRateMonitor
      init_args:
        logging_interval: epoch

  max_epochs: 200
  check_val_every_n_epoch: 1
  log_every_n_steps: 50
  enable_checkpointing: true
  default_root_dir: <path to root dir>
data:
  class_path: GenericNonGeoSegmentationDataModule
  init_args:
    batch_size: 4
    num_workers: 8
    constant_scale: 0.0001
    rgb_indices:
      - 2
      - 1
      - 0
    filter_indices:
      - 2
      - 1
      - 0
      - 3
      - 4
      - 5
    train_data_root: <path to train data root>
    val_data_root: <path to val data root>
    test_data_root: <path to test data root>
    img_grep: "*_S2GeodnHand.tif"
    label_grep: "*_LabelHand.tif"
    means:
      - 0.107582
      - 0.13471393
      - 0.12520133
      - 0.3236181
      - 0.2341743
      - 0.15878009
    stds:
      - 0.07145836
      - 0.06783548
      - 0.07323416
      - 0.09489725
      - 0.07938496
      - 0.07089546
    num_classes: 2

model:
  class_path: SemanticSegmentationTask
  init_args:
    model_args:
      decoder: FCNDecoder
      pretrained: true
      backbone: prithvi_vit_100
      img_size: 512
      in_channels: 6
      bands:
        - RED
        - GREEN
        - BLUE
        - NIR_NARROW
        - SWIR_1
        - SWIR_2
      num_frames: 1
      num_classes: 2
      head_dropout: 0.1
      head_channel_list:
        - 256
    loss: ce
    aux_heads:
      - name: aux_head
        decoder: FCNDecoder
        decoder_args:
          decoder_channels: 256
          decoder_in_index: 2
          decoder_num_convs: 1
          head_channel_list:
            - 64
    aux_loss:
      aux_head: 1.0
    ignore_index: -1
    class_weights:
      - 0.3
      - 0.7
    freeze_backbone: false
    freeze_decoder: false
    model_factory: PrithviModelFactory
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 6.e-5
    weight_decay: 0.05
lr_scheduler:
  class_path: ReduceLROnPlateau
  init_args:
    monitor: val/loss

```

To run this training task, simply execute `terratorch fit --config <path_to_config_file>`

To test your model on the test set, execute `terratorch test --config  <path_to_config_file> --ckpt_path <path_to_checkpoint_file>`

For inference, execute `terratorch predict -c <path_to_config_file> --ckpt_path<path_to_checkpoint> --predict_output_dir <path_to_output_dir> --data.init_args.predict_data_root <path_to_input_dir> --data.init_args.predict_dataset_bands <all bands in the predicted dataset, e.g. [BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2,0]>`

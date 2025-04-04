# The YAML configuration file: an overview 

If you are using the command-line interface (CLI) to run jobs using TerraTorch, so you must became familiar with
YAML, the format used to configure all the workflow within the toolkit. Writing a YAML file is very similar to
coding, because even if you are not direclty handling the classes and others structures defined inside a codebase,
you need to know how they work, their input argments and their position in the pipeline. In this way, we
could call it a "low-code" task.
The YAML file used for TerraTorch has an almost closed format, since there are a few fixed fields that must be filled with
limited sets of classes, which makes easier for new users to get a pre-existing YAML file and adapt it to
their own purposes. 
 
In the next sections, we describe each field of a YAML file used for Earth Observation Foundation Models (EOFM) and try to make it clearer for a new user. However, we will not go into detail, since the complementary documentation (Lightning, PyTorch, ...) must fill this gap. The example can be
downloaded [here](config.yaml){:download="config.yaml"}. 

## Trainer

In the section called `trainer` are defined all the arguments that must be directly sent to the Lightning
Trainer object. If you need a deeper explantion about this object, check the [Lightning's documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html).
In the first lines we have:

```
trainer:
  accelerator: cpu
  strategy: auto
  devices: auto
  num_nodes: 1
  precision: 16-mixed
```
In which:

* `accelerator` refers to the kind of device is being used to run the experiment. We are usually
more interested in `cpu` and `gpu`, but if you set `auto`, it will automaticaly select allocate the GPU is
that is availble or otherwise run on CPU.
* `strategy` is related to the kind of parallelism is available. As we have usually ran the experiments using a
single device for finetuning or inference, we do not care about it and choose the option `auto` by default. 
* `devices` indicates the list of available devices to use for the experiment. Leave it as `auto` if you are
running with a single device. 
* `num_nodes` is self-explanatory. We have mostly tested TerraTorch for a single-node jobs, so, it is better to
set it as `1` for now. 
* `precision` is the kind of precision used for your model. `16-mixed` have been an usual choice. 

Just below this initial stage, we have `logger`:
```
  logger:
    class_path: TensorBoardLogger
    init_args:
      save_dir: tests/
      name: all_ecos_random
```
In this field we define the configuration for logging the model state. In this example we are using
[Tensorboard](https://www.tensorflow.org/tensorboard?hl=pt-br), and saving all the logs in a directory `tests/all_ecos_random`. 
Others frameworks, as [MLFlow](https://mlflow.org/) are also supported. Check the [Lightning documentation
about logging](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) for a more complete description. 

The `callbacks` field:
```
  callbacks:
    - class_path: RichProgressBar
    - class_path: LearningRateMonitor
      init_args:
        logging_interval: epoch
    - class_path: EarlyStopping
      init_args:
        monitor: val/loss
        patience: 100
```
Represents a list of operations that can be invoked with determined frequency. The user is free to add
others operations from Lightning or custom ones. In the current config we are basically defining: a progress
bar to be printed during the model training/validation and a learning rate monitor, determined to call early-stopping when the model shows
signals of overfitting. 
The rest of the arguments are:
```
  max_epochs: 1
  check_val_every_n_epoch: 1
  log_every_n_steps: 20
  enable_checkpointing: true
  default_root_dir: tests/
```

* `max_epochs`: the maximum number of epochs to train the model. Notice that, if you are using early-stopping,
    maybe the training will finish before achieving this number. 
* `check_val_every_n_epoch`: the frequency to evaluate the model using the validation dataset. The validation
    is important to verify if the model is tending to overfit and  can be used, for example, to define when update the learning rate, or to invoke the early-stopping. 
* `enable_checkpointing`: it enables the checkpointing, the action of periodically saving the state of the
    model to a file. 
* `default_root_dir`: the directory used to save the model checkpoints. 

## Datamodule

In this section, we start direclty handling TerraTorch's built-in structures. The field `data` is expected to
receive a [generic datamodule](../generic_datamodules.md) or any other datamodule compatible with [Lightning
Datamodules](https://lightning.ai/docs/pytorch/stable/data/datamodule.html), as those defined
in our [collection of datamodules](../datamodules.md). 

In the beginning of the field we have:
```
data:
  class_path: GenericNonGeoPixelwiseRegressionDataModule
  init_args:
```
It means that we have chosen the generic regression datamodule and we will pass all its required arguments below
`init_args` and with one new level of identation. The best practice here is to check the documentation of the
datamodule class you are using (in our case, [here][terratorch.datamodules.generic_pixel_wise_data_module])
and verify all the arguments it expects to receive ant then to fill the lines
with `<argument_name>: <argument_value>`. 
As the TerraTorch and Lightning modules were already imported in the CLI script (`terratorch/cli_tools.py`),
you do not need to provide the complete paths for them. Otherwise, if you are using a datamodule defined in an
external package, indicate the path to import the model, as `package.datamodules.SomeDatamodule`. 

## Model

The field `model` is, in fact, the configuration for `task + model`: 
```
model:
  class_path: terratorch.tasks.PixelwiseRegressionTask
  init_args:
    model_args:
      decoder: UperNetDecoder
      pretrained: false
      backbone: prithvi_eo_v2_600
      backbone_drop_path_rate: 0.3
      backbone_window_size: 8
      decoder_channels: 64
      num_frames: 1
      in_channels: 6
      bands:
        - BLUE
        - GREEN
        - RED
        - NIR_NARROW
        - SWIR_1
        - SWIR_2
      head_dropout: 0.5708022831486758
      head_final_act: torch.nn.ReLU
      head_learned_upscale_layers: 2
    loss: rmse
    ignore_index: -1
    freeze_backbone: true
    freeze_decoder: false
    model_factory: PrithviModelFactory
    tiled_inference_parameters:
       h_crop: 224
       h_stride: 192
       w_crop: 224
       w_stride: 192
       average_patches: true
```
Notice that there is a field `model_args`, which it is intended to receive all the necessary configuration to
instantiate the model itself, that means, the structure `backbone + decoder + head`. Inside `model_args`, it
is possible do define which arguments will be sent to each component by including a prefix to the argument
names, as `backbone_<argument>` or `decoder_<other_argument>`. Alternatively, it is possible to pass the
arguments using dictionaries `backbone_kwargs`, `decoder_kwargs` and `head_kwargs`. The same recommendation
made for the `data` field is repeated here, check the documentation of the [task](../tasks.md) and
model classes ([backbones](../backbones), [decoders](../decoders.md) and [heads](../heads)) you are using in
order to define which arguments to write for each subfield of `model`. 

## Optimizer and Learning Rate Scheduler

The last two fields of out example are the configuration of the optimizer and the lr scheduler. Those fields
are mostly self-explanatory for users already familiar with machine learning:

```
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 0.00013524680528283027
    weight_decay: 0.047782217873995426
lr_scheduler:
  class_path: ReduceLROnPlateau
  init_args:
    monitor: val/loss
```
Check the [PyTorch documentation about optimization](https://pytorch.org/docs/stable/optim.html) to understand them more deeply. 



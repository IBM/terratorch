# Creating a finetuning workload with the script interface
This tutorial does not intend to create an accurate finetuned example (we are running for a single epoch!), but to describe step-by-step how to instantiate and run this kind of task. 


```python
from lightning.pytorch import Trainer
import terratorch
import albumentations
from albumentations.pytorch import ToTensorV2
from terratorch.models import EncoderDecoderFactory
from terratorch.models.necks import SelectIndices, LearnedInterpolateToPyramidal, ReshapeTokensToImage
from terratorch.models.decoders import UNetDecoder
from terratorch.datasets import HLSBands
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask
```

### Defining fundamental parameters:
* `lr` - learning rate.
* `accelerator` - The kind of device in which the model will be executed. It is usually `gpu` or `cpu`. If we set it as `auto`, Lightning will select the most appropiate available device.
* `max_epochs` - The maximum number of epochs used to train the model. 


```python
lr = 1e-4
accelerator = "auto"
max_epochs = 1
```

### Next, we will instantiate the datamodule, the object we will use to load the files from disk to memory.


```python
datamodule = GenericNonGeoSegmentationDataModule(
    batch_size = 2,
    num_workers = 8,
    dataset_bands = [HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED, HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2],
    output_bands = [HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED, HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2],
    rgb_indices = [2, 1, 0],
    means = [
          0.033349706741586264,
          0.05701185520536176,
          0.05889748132001316,
          0.2323245113436119,
          0.1972854853760658,
          0.11944914225186566,
    ],
    stds = [
          0.02269135568823774,
          0.026807560223070237,
          0.04004109844362779,
          0.07791732423672691,
          0.08708738838140137,
          0.07241979477437814,
    ],
    train_data_root = "../burn_scars/hls_burn_scars/training",
    val_data_root = "../burn_scars/hls_burn_scars/validation",
    test_data_root = "../burn_scars/hls_burn_scars/validation",
    img_grep = "*_merged.tif",
    label_grep = "*.mask.tif",
    num_classes = 2,
    train_transform = [albumentations.D4(), ToTensorV2()],
    test_transform = [ToTensorV2()],
    no_data_replace = 0,
    no_label_replace =  -1,
)
```

### A dictionary containing all the arguments necessary to instantiate a complete `backbone-neck-decoder-head`, which will be passed to the task object.


```python
model_args = dict(
  backbone="prithvi_eo_v2_300",
  backbone_pretrained=True,
  backbone_num_frames=1,
  num_classes = 2,
  backbone_bands=[
      "BLUE",
      "GREEN",
      "RED",
      "NIR_NARROW",
      "SWIR_1",
      "WIR_2",
  ],
  decoder = "UNetDecoder",
  decoder_channels = [512, 256, 128, 64],
  necks=[{"name": "SelectIndices", "indices": [5, 11, 17, 23]},
         {"name": "ReshapeTokensToImage"},
         {"name": "LearnedInterpolateToPyramidal"}],
  head_dropout=0.1
)
```

### Creating the `task` object, which will be used to properly define how the model will be trained and used after it.


```python
task = SemanticSegmentationTask(
    model_args,
    "EncoderDecoderFactory",
    loss="ce",
    lr=lr,
    ignore_index=-1,
    optimizer="AdamW",
    optimizer_hparams={"weight_decay": 0.05},
    freeze_backbone = False,
    plot_on_val = False,
    class_names = ["Not burned", "Burn scar"],
)
```

### The object `Trainer` manages all the training process. It can be interpreted as an improved optimization loop, in which parallelism and checkpointing are transparently managed by the system. 


```python
trainer = Trainer(
    accelerator=accelerator,
    max_epochs=max_epochs,
)
```

### Executing the training.


```python
trainer.fit(model=task, datamodule=datamodule)
```

### Testing the trained model (extracting metrics). 


```python
trainer.test(dataloaders=datamodule)
```

The metrics output:

```
    [{'test/loss': 0.2669268250465393,
      'test/Multiclass_Accuracy': 0.9274423718452454,
      'test/multiclassaccuracy_Not burned': 0.9267654418945312,
      'test/multiclassaccuracy_Burn scar': 0.9340785145759583,
      'test/Multiclass_F1_Score': 0.9274423718452454,
      'test/Multiclass_Jaccard_Index': 0.7321492433547974,
      'test/multiclassjaccardindex_Not burned': 0.9205750226974487,
      'test/multiclassjaccardindex_Burn scar': 0.5437235236167908,
      'test/Multiclass_Jaccard_Index_Micro': 0.8647016882896423}]
```


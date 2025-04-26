import os
import sys
import torch
import torchgeo
import terratorch
import albumentations
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from pathlib import Path
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.models import EncoderDecoderFactory
from terratorch.models.decoders import IdentityDecoder
from albumentations.pytorch import ToTensorV2
import warnings

warnings.filterwarnings('ignore')

max_epochs = 1

# Our datamodule. 

datamodule = terratorch.datamodules.TorchNonGeoDataModule(
    transforms = [
      albumentations.augmentations.geometric.resize.Resize(height=224, width=224),
      ToTensorV2()],
      cls=torchgeo.datamodules.EuroSATDataModule,
      batch_size=32,
      num_workers=8,
      root="./EuroSat",
      download=True,
      bands = ["B02","B03", "B04", "B8A", "B11", "B12"]
)

# Instantiating the Trainer.

pl.seed_everything(0)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath="output/burnscars/checkpoints/",
    mode="max",
    monitor="val/Multiclass_Jaccard_Index", # Variable to monitor
    filename="best-{epoch:02d}",
)

# Lightning Trainer

trainer = pl.Trainer(
    accelerator="auto",
    strategy="auto",
    devices=1, # Deactivate multi-gpu because it often fails in notebooks
    precision='bf16-mixed',  # Speed up training
    num_nodes=max_epochs,
    logger=True,  # Uses TensorBoard by default
    max_epochs=max_epochs, # For demos
    log_every_n_steps=1,
    enable_checkpointing=True,
    default_root_dir="output/eurosat",
    detect_anomaly=True,
)


# Classification task. 

model = terratorch.tasks.ClassificationTask(
        model_args={
      "decoder": "IdentityDecoder",
      "backbone_pretrained": True,
      "backbone": "prithvi_eo_v1_100",
      "head_dim_list": [384, 128],
      "backbone_bands":
        ["BLUE",
        "GREEN",
        "RED",
        "NIR_NARROW",
        "SWIR_1",
        "SWIR_2"],
      "num_classes": 10,
     "head_dropout": 0.1
      },
     loss = "ce",
     freeze_backbone = False,
     model_factory = "EncoderDecoderFactory",
     optimizer = "AdamW",
     lr = 1.e-4,
     scheduler_hparams = {
         "weight_decay" : 0.05,
     }
)

# Executing the training. 

trainer.fit(model, datamodule=datamodule)

# Executing the test step. 

trainer.test(model, datamodule=datamodule)





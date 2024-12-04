from terratorch.models import EncoderDecoderFactory
from terratorch.datasets import HLSBands
import torch 

import os
import subprocess

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.models.model import AuxiliaryHead
from terratorch.tasks import SemanticSegmentationTask
from terratorch.models.model import AuxiliaryHeadWithDecoderWithoutInstantiatedHead, Model, ModelOutput

import shutil
import matplotlib.pyplot as plt
import rioxarray as rio


class CustomModel(torch.nn.Module):
    def __init__(self, model_1:torch.nn.Module=None, model_2:torch.nn.Module=None):

        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2

    def forward(self, x:torch.Tensor):

        output_1 = self.model_1(x)
        output_2 = self.model_2(x)
        mask =  (output_1.output + output_2.output)/2

        return ModelOutput(output=mask)

    def freeze_encoder(self):

        self.model_1.freeze_encoder()
        self.model_2.freeze_encoder()

    def freeze_decoder(self):

        self.model_1.freeze_decoder()
        self.model_2.freeze_decoder()

model_factory = EncoderDecoderFactory()

batch_size = 1
num_workers = 19

subprocess.run("wget https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-100M-Burn-scars-demo/resolve/main/subsetted_512x512_HLS.S30.T10TGS.2018285.v1.4_merged.tif", shell=True)

input_file_name = "subsetted_512x512_HLS.S30.T10TGS.2018285.v1.4_merged.tif"
label_file_name = "subsetted_512x512_HLS.S30.T10TGS.2018285.v1.4.mask.tif"

# organize the data directory
if not os.path.isdir("burn_scar_segmentation_toy"):
    os.mkdir("burn_scar_segmentation_toy")

    for data_dir in ["train_images", "test_images", "val_images"]:
        os.mkdir(os.path.join("burn_scar_segmentation_toy", data_dir))
        shutil.copy(input_file_name, os.path.join("burn_scar_segmentation_toy", data_dir, input_file_name))

    for label_dir in ["train_labels", "test_labels", "val_labels"]:
        os.mkdir(os.path.join("burn_scar_segmentation_toy", label_dir))
        shutil.copy(label_file_name, os.path.join("burn_scar_segmentation_toy", label_dir, label_file_name))

train_val_test = [
    "burn_scar_segmentation_toy/train_images",
    "burn_scar_segmentation_toy/val_images",
    "burn_scar_segmentation_toy/test_images",
]

train_val_test_labels = {
    "train_label_data_root": "burn_scar_segmentation_toy/train_labels",
    "val_label_data_root": "burn_scar_segmentation_toy/val_labels",
    "test_label_data_root": "burn_scar_segmentation_toy/test_labels",
}

# from https://github.com/NASA-IMPACT/hls-foundation-os/blob/main/configs/burn_scars.py
means=[
        0.033349706741586264,
        0.05701185520536176,
        0.05889748132001316,
        0.2323245113436119,
        0.1972854853760658,
        0.11944914225186566,
    ]
stds=[
        0.02269135568823774,
        0.026807560223070237,
        0.04004109844362779,
        0.07791732423672691,
        0.08708738838140137,
        0.07241979477437814,
    ]
datamodule = GenericNonGeoSegmentationDataModule(
    batch_size,
    num_workers,
    *train_val_test,
    "*_merged.tif", # img grep
    "*.mask.tif", # label grep
    means,
    stds,
    2, # num classes
    **train_val_test_labels,

    # if transforms are defined with Albumentations, you can pass them here
    # train_transform=train_transform,
    # val_transform=val_transform,
    # test_transform=test_transform,

    # edit the below for your usecase
    dataset_bands=[
        HLSBands.BLUE,
        HLSBands.GREEN,
        HLSBands.RED,
        HLSBands.NIR_NARROW,
        HLSBands.SWIR_1,
        HLSBands.SWIR_2,
    ],
    output_bands=[
        HLSBands.BLUE,
        HLSBands.GREEN,
        HLSBands.RED,
        HLSBands.NIR_NARROW,
        HLSBands.SWIR_1,
        HLSBands.SWIR_2,
    ],
    no_data_replace=0,
    no_label_replace=-1,
)
# we want to access some properties of the train dataset later on, so lets call setup here
# if not, we would not need to
datamodule.setup("fit")

model_1 = model_factory.build_model(task="segmentation",
        backbone="prithvi_vit_tiny",
        decoder="IdentityDecoder",
        backbone_bands=[
            HLSBands.BLUE,
            HLSBands.GREEN,
            HLSBands.RED,
            HLSBands.NIR_NARROW,
            HLSBands.SWIR_1,
            HLSBands.SWIR_2,
        ],
        num_classes=2,
        backbone_pretrained=False, #True,
        backbone_num_frames=1,
        head_dropout=0.2
    )

model_2 = model_factory.build_model(task="segmentation",
        backbone="prithvi_vit_tiny",
        decoder="IdentityDecoder",
        backbone_bands=[
            HLSBands.BLUE,
            HLSBands.GREEN,
            HLSBands.RED,
            HLSBands.NIR_NARROW,
            HLSBands.SWIR_1,
            HLSBands.SWIR_2,
        ],
        num_classes=2,
        backbone_pretrained=False, #True,
        backbone_num_frames=1,
        head_dropout=0.2
    )

model = CustomModel(model_1=model_1, model_2=model_2)
model.freeze_encoder()

epochs = 1 # 1 epoch for demo
lr = 1e-3

model_args = {
        "num_classes": 2,
        "backbone_bands": [
            HLSBands.RED,
            HLSBands.GREEN,
            HLSBands.BLUE,
            HLSBands.NIR_NARROW,
            HLSBands.SWIR_1,
            HLSBands.SWIR_2,
        ],
        "backbone_pretrained": False, #True,
        "backbone_num_frames":1, # this is the default
        "decoder_channels":128,
        "head_dropout":0.2,
        "necks": [
            {"name": "SelectIndices", "indices": [-1]},
            {"name": "ReshapeTokensToImage"}
        ]
}

task = SemanticSegmentationTask(
    model_args, 
    None,
    model=model,
    loss="ce",
    #aux_loss={"fcn_aux_head": 0.4},
    lr=lr,
    ignore_index=-1,
    optimizer="AdamW",
    optimizer_hparams={"weight_decay": 0.05},
)

accelerator = "gpu"
experiment = "tutorial"
if not os.path.isdir("tutorial_experiments"):
    os.mkdir("tutorial_experiments")
default_root_dir = os.path.join("tutorial_experiments", experiment)
checkpoint_callback = ModelCheckpoint(monitor=task.monitor, save_top_k=1, save_last=True)
early_stopping_callback = EarlyStopping(monitor=task.monitor, min_delta=0.00, patience=20)
logger = TensorBoardLogger(save_dir=default_root_dir, name=experiment)

trainer = Trainer(
    # precision="16-mixed",
    accelerator=accelerator,
    callbacks=[
        RichProgressBar(),
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
    ],
    logger=logger,
    max_epochs=epochs, # train only one epoch for demo
    default_root_dir=default_root_dir,
    log_every_n_steps=1,
    check_val_every_n_epoch=200

)

trainer.fit(model=task, datamodule=datamodule)

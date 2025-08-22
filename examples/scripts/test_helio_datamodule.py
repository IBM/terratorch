from terratorch.datamodules import HelioNetCDFDataModule
from terratorch.registry import BACKBONE_REGISTRY
from terratorch.tasks import InferenceTask

import lightning.pytorch as pl
import torch 

index_path = "/home/jalmeida/Datasets/helio/data/index_2011_test.csv"
scalers_path = "/home/jalmeida/Datasets/helio/data/scale_signum_log_factor1e-02_2011_2020.yaml"

channels = ['aia94', 'aia131', 'aia171', 'aia193', 'aia211', 'aia304',
            'aia335', 'aia1600', 'hmi_m', 'hmi_bx', 'hmi_by', 'hmi_bz', 'hmi_v']

datamodule = HelioNetCDFDataModule(
    train_index_path=index_path,
    test_index_path=index_path,
    val_index_path=index_path,
    predict_index_path=index_path,
    batch_size=2,
    time_delta_input_minutes=[-24, -12] ,
    time_delta_target_minutes=0,
    channels = channels,
    n_input_timestamps=2,
    rollout_steps=1,
    scalers=scalers_path)

model_name = "heliofm_backbone_surya"

backbone = BACKBONE_REGISTRY.build(model_name)

datamodule.setup("predict")

model = InferenceTask(model=backbone) 

pl.seed_everything(0)

# Lightning Trainer
trainer = pl.Trainer(
    accelerator="gpu",
    strategy="auto",
    devices=1, 
    precision='bf16-mixed',
    num_nodes=1,
    logger=True,
    max_epochs=1,
    log_every_n_steps=1,
    enable_checkpointing=True,
    callbacks=[pl.callbacks.RichProgressBar()],
    default_root_dir="output/heliofm",
    detect_anomaly=True,
)

# Training
prediction = trainer.predict(model, datamodule=datamodule,
                             ckpt_path=checkpoint_path)

print([p.shape for p in prediction])


import os
import glob
from datetime import datetime
from huggingface_hub import snapshot_download, hf_hub_download
from terratorch.datamodules import HelioNetCDFDataModule
from terratorch.registry import BACKBONE_REGISTRY
from terratorch.tasks import InferenceTask
import pickle 
import matplotlib.pyplot as plt
import pandas as pd
import lightning.pytorch as pl
import torch 

if not os.path.isdir("experiment"):
    os.mkdir("experiment")

root_dir = "experiment"

# Downloading validation data
snapshot_download(repo_id="nasa-ibm-ai4science/Surya-1.0_validation_data",
                  repo_type="dataset", local_dir=root_dir)

# Downloading scalers file
hf_hub_download(repo_id="nasa-ibm-ai4science/Surya-1.0",
                  filename="scalers.yaml", local_dir=root_dir)

# Creating index file
sample_files = glob.glob(os.path.join(root_dir,"*.nc"))
paths = sorted(sample_files)
present = len(paths)*[1]

timestamps  = []
for ff in paths:
    filename = os.path.basename(ff).replace(".nc", "")
    date, timestamp = filename.split("_")

    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:])
    hour = int(timestamp[:2])
    minutes = int(timestamp[2:])
    seconds = 0

    date_datetime = datetime(year, month, day, hour, minutes,
                             seconds).strftime("%Y-%m-%d %H:%M:%S")


    timestamps.append(date_datetime)

index_dict = {"path": paths, "timestep": timestamps, "present": present}
index_dataframe = pd.DataFrame(index_dict)
index_dataframe.to_csv(os.path.join(root_dir, "index.csv"))
index_path = os.path.join(root_dir, "index.csv")
scalers_path = os.path.join(root_dir, "scalers.yaml")

channels = ['aia94', 'aia131', 'aia171', 'aia193', 'aia211', 'aia304',
        'aia335', 'aia1600', 'hmi_m', 'hmi_bx', 'hmi_by', 'hmi_bz', 'hmi_v']

run = "predict"
#run = "postprocess"

if run == "predict":

    datamodule = HelioNetCDFDataModule(
        train_index_path=index_path,
        test_index_path=index_path,
        val_index_path=index_path,
        predict_index_path=index_path,
        batch_size=1,
        num_workers=0,
        time_delta_input_minutes=[-60,0] ,
        time_delta_target_minutes=+60,
        channels = channels,
        n_input_timestamps=2,
        rollout_steps=1,
        scalers=scalers_path)

    model_name = "heliofm_backbone_surya"

    backbone = BACKBONE_REGISTRY.build(model_name, pretrained=True)

    datamodule.setup("predict")

    model = InferenceTask(model=backbone) 

    pl.seed_everything(0)

    # Lightning Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="auto",
        devices=1, 
        #precision='bf16-mixed',
        num_nodes=1,
        logger=True,
        max_epochs=1,
        log_every_n_steps=1,
        enable_checkpointing=True,
        callbacks=[pl.callbacks.RichProgressBar()],
        default_root_dir="output/heliofm",
    )

    # Training
    prediction = trainer.predict(model, datamodule=datamodule)

    with open("output.pickle", "wb") as f:
        pickle.dump(prediction, f)

else:
    if not os.path.isdir("figs"):
        os.mkdir("figs")

    with open("output.pickle", "rb") as fp:
        data = pickle.load(fp)

        for bi, dset in enumerate(data):

           dset = dset.cpu().float().numpy().astype("float32")

           for channel in range(dset.shape[1]):

               plt.imshow(dset[0, channel, ...])
               plt.savefig(f"figs/image_batch_{bi}_channel_{channels[channel]}.png")

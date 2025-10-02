import glob
import os
import pickle
import time
from datetime import datetime

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import pandas as pd
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from terratorch.datasets import HelioNetCDFDataset
from terratorch.registry import BACKBONE_REGISTRY
from terratorch.tasks import InferenceTask

if not os.path.isdir("experiment"):
    os.mkdir("experiment")

root_dir = "experiment"

# Downloading validation data
snapshot_download(repo_id="nasa-ibm-ai4science/Surya-1.0_validation_data", repo_type="dataset", local_dir=root_dir)

# Downloading scalers file
hf_hub_download(repo_id="nasa-ibm-ai4science/Surya-1.0", filename="scalers.yaml", local_dir=root_dir)

# Creating index file
sample_files = glob.glob(os.path.join(root_dir, "*.nc"))
paths = sorted(sample_files)
present = len(paths) * [1]

timestamps = []
for ff in paths:
    filename = os.path.basename(ff).replace(".nc", "")
    date, timestamp = filename.split("_")

    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:])
    hour = int(timestamp[:2])
    minutes = int(timestamp[2:])
    seconds = 0

    date_datetime = datetime(year, month, day, hour, minutes, seconds).strftime("%Y-%m-%d %H:%M:%S")

    timestamps.append(date_datetime)

index_dict = {"path": paths, "timestep": timestamps, "present": present}
index_dataframe = pd.DataFrame(index_dict)
index_dataframe.to_csv(os.path.join(root_dir, "index.csv"))
index_path = os.path.join(root_dir, "index.csv")
scalers_path = os.path.join(root_dir, "scalers.yaml")

channels = [
    "aia94",
    "aia131",
    "aia171",
    "aia193",
    "aia211",
    "aia304",
    "aia335",
    "aia1600",
    "hmi_m",
    "hmi_bx",
    "hmi_by",
    "hmi_bz",
    "hmi_v",
]

dataset = HelioNetCDFDataset(
    index_path=index_path,
    time_delta_input_minutes=[-60, 0],
    time_delta_target_minutes=+60,
    channels=channels,
    n_input_timestamps=2,
    rollout_steps=1,
    scalers=scalers_path,
)

start_time = time.time()
for batch in dataset:
    batch_ = {k: torch.from_numpy(v).to("cuda:0") for k, v in batch[0].items()}
print(f"Elapsed time: {time.time() - start_time} s")

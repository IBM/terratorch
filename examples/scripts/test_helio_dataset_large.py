from terratorch.datasets import HelioNetCDFDataset
from terratorch.registry import BACKBONE_REGISTRY
import torch 
import pickle 

index_path = "/dccstor/jlsa931/Datasets/helio/data/index.csv"
scalers_path = "/dccstor/jlsa931/Datasets/helio/data/scalers.yaml"

channels = ['aia94', 'aia131', 'aia171', 'aia193', 'aia211', 'aia304',
            'aia335', 'aia1600', 'hmi_m', 'hmi_bx', 'hmi_by', 'hmi_bz', 'hmi_v']

dataset = HelioNetCDFDataset(
    index_path=index_path,
    time_delta_input_minutes=[-60,0],
    time_delta_target_minutes=+60,
    phase = "train",
    channels = channels,
    n_input_timestamps=2,
    rollout_steps=1,
    scalers=scalers_path)

model_name = "heliofm_backbone_surya"
backbone = BACKBONE_REGISTRY.build(model_name, pretrained=True)

print("Model restored")

outputs = []
for batch in dataset:
    batch_ = batch[0]
    batch_["ts"] = torch.from_numpy(batch_["ts"][None, ...])
    output = backbone(batch_)
    outputs.append(output)

with open("output_dataset.pickle", "wb") as f:
    pickle.dump(outputs, f)


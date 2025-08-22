from terratorch.datasets import HelioNetCDFDataset
from terratorch.registry import BACKBONE_REGISTRY
import torch 

index_path = "/home/jalmeida/Datasets/helio/Data/index.csv"
scalers_path = "/home/jalmeida/Datasets/helio/Data/scalers.yaml"

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
checkpoint_path = "/home/jalmeida/Datasets/helio/Data/surya.366m.v1.pt"
state_dict = torch.load(checkpoint_path, map_location="cuda:0")
backbone = BACKBONE_REGISTRY.build(model_name)
backbone.load_state_dict(state_dict)

for batch in dataset:
    batch_ = batch[0]
    batch_["ts"] = torch.from_numpy(batch_["ts"][None, ...])
    output = backbone(batch_)
    print(output.shape)

print(len(dataset))

import torch
from collections import OrderedDict
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input_file")
parser.add_argument("--output_file")
parser.add_argument("--n_copies", type=int, default=2)

args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file

state_dict = torch.load(input_file)
state_dict_ = OrderedDict({"model."+k: v for k,v in state_dict.items()})
content = {"state_dict": state_dict_, "pytorch-lightning_version": "2.5.0post0"}

torch.save(content, output_file)

import argparse

import numpy as np

parser = argparse.ArgumentParser('Arguments', add_help=False)

# data loader related
parser.add_argument('--input_file', type=str)
args = parser.parse_args()

input_file = args.input_file

data = np.genfromtxt(input_file, delimiter=',', dtype=str)

col1 = data[:,0].tolist()

col1_ = ["_".join(i.split("_")[:2]) for i in col1]

np.savetxt(input_file.replace("csv", "txt"), col1_, fmt="%s")

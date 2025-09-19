
import yaml, json; 
import os
import argparse


tool_description= """
This script transforms Terratorch configuration files into a format 
compatible with vLLM.
It takes a Terratorch config file as input and generates a config.json file 
required to initialize the model within the vLLM framework.
The output file is saved in the same directory as the original configuration file.
"""

def generate_configuration(terratorch_config_file: str,data_input: str):

    vllm_config={}

    with open(terratorch_config_file) as input_stream:
        terratorch_config = yaml.safe_load(input_stream)

    vllm_config["architectures"] = ["Terratorch"]
    vllm_config["num_classes"] = 0
    vllm_config["pretrained_cfg"] = terratorch_config

    if os.path.exists(data_input):
        vllm_config["pretrained_cfg"]['input'] = load_input_file(data_input)
    else:
        vllm_config["pretrained_cfg"]['input'] = load_input_string(data_input)

    config_dirname = os.path.dirname(terratorch_config_file)

    output_file_path = os.path.splitext(terratorch_config_file)[0]+".json"

    with open(f"{config_dirname}/config.json", 'w') as file:
        json.dump(vllm_config, file, indent=2)

    print(f"Configuration file available at the path: {output_file_path}")

def load_input_string(input):
    input_data_entries = json.loads(input)
    #loop through entries and update type and shape in place
    return input_data_entries

def load_input_file(input):
    with open(input) as f:
        input_data_entries = json.load(f)
    return input_data_entries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=tool_description)
    parser.add_argument("--ttconfig", 
                        help="Terratorch model configuration file",
                        type=str)
    parser.add_argument('-i','--input', 
                        help='<Required> Input data', 
                        required=True)
    args = parser.parse_args()

    generate_configuration(args.ttconfig,args.input)
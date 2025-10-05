# Serving a TerraTorch model with vLLM
All models to be served with vLLM via IOProcessor plugins must adhere to a specific configuration structure. In the specific all models are required to profide a configuration file named `config.json` that can be hosted on HuggingFace, or in a local folder alongside the model weights (`.pt` file).


## vLLM compatible model configuration
The snippet below shows the structure of the `config.json` file for a Prithvi 300M model finetuned on the Sen1Floods11 dataset:
```json title="Prithvi 300M model vLLM configuration file"
{
  "architectures": [
    "Terratorch"
  ],
  "num_classes": 0,
  "pretrained_cfg": {
    "seed_everything": 0,
    "input":{
      "target": "pixel_values",
      "data":{
        "pixel_values":{
          "type": "torch.Tensor",
          "shape": [6, 512, 512]
        },
        "location_coords":{
          "type":"torch.Tensor",
          "shape": [1, 2]
        }
      }
    },
    "model": {
      "class_path": "terratorch.tasks.SemanticSegmentationTask",
      "init_args": {
        "model_args": {
          "backbone_pretrained": true,
          "backbone": "prithvi_eo_v2_300_tl",
          "decoder": "UperNetDecoder",
          "decoder_channels": 256,
          "decoder_scale_modules": true,
          "num_classes": 2,
          "rescale": true,
          "backbone_bands": [
            "BLUE",
            "GREEN",
            "RED",
            "NIR_NARROW",
            "SWIR_1",
            "SWIR_2"
          ],
          "head_dropout": 0.1,
          "necks": [
            {
              "name": "SelectIndices",
              "indices": [
                5,
                11,
                17,
                23
              ]
            },
            {
              "name": "ReshapeTokensToImage"
            }
          ]
        },
        "model_factory": "EncoderDecoderFactory",
        "loss": "ce",
        "ignore_index": -1,
        "lr": 0.001,
        "freeze_backbone": false,
        "freeze_decoder": false,
        "plot_on_val": 10
      }
    }
  }
}
```

from the above we highlight two main sections: 1) vLLM required info, 2) model configuration.

### vLLM Required Information Section
At the top of the configuration file we find

```json title="vLLM specific configuration information"
"architectures": [
    "Terratorch"
  ],
"num_classes": 0,
```

These values are mandatory and must be kept unchanged.

### Model Specification Section

The model specification section is contained in the `pretrained_cfg` section of the configuration file and is in turn composed of three sub-sections: 1) model input specification, 2) model configuration and 3) datamodule configuration.

#### Model Input Specification
The model info specification is necessary to support vLLM in performing a set of warm-up runs used for measuring the model's memory consumption.

Below is an extract from the Prithvi 300M configuration file.
```json title="Prithvi 300M input specification"
"input":{
    "target": "pixel_values",
    "data":{
        "pixel_values":{
            "type": "torch.Tensor",
            "shape": [6, 512, 512]
        },
        "location_coords":{
            "type":"torch.Tensor",
            "shape": [1, 2]
        }
    }
}
```

From the above example, the `data` field is mandatory and contains one entry for each input field, described with their type (e.g., `torch.Tensor`) and shape (e.g., `[6, 512, 512]`). Also, we support models whose forward function accept arguments in the form or named argument, or as a combination of one positional argument and named arguments. The one positional argument is specified with the `target` field. With the above input configuration vLLM would invoke the model forward function as in the snippet below:

```python
model.forward(pixel_values, location_coords=location_coords)
```

If no positional argument is required, the `target` field can be omitted and all entries in the `data` field would be passed as named arguments.

#### Model and Datamodule Configuration

The model configuration is contained in the `model` section of the configuration file, while the datamodule configuration is in the `data` section. Please refer to the full snippet at the beginning of this page. Both model and datamodule configuration come straight from the model yaml configuration used when performing inference via the LightningCLI.
The presence of the `data` section is not mandatory but highly advised, since most plugins would use the model datamodule for performing pre/post processing operations on the input/output data.

### Automatic generation of vLLM model configurations

To help the user we have developed a script ([vllm_config_generator.py](https://github.com/IBM/terratorch/blob/updated-vllm_plugin/examples/scripts/vllm_config_generator.py)) for automatically generating the `config.json` file. The script takes two arguments: 1) the model configuration in yaml format, 2) the input specification. The specification of the input can be specified as both a json string or the path to a json file. See the examples below.

```bash title="Generating vLLM configuration with input specification as string"
python vllm_config_generator.py \
   --ttconfig config.yaml \
   -i '{"data":{"pixel_values":{"type": "torch.Tensor","shape": [6, 512, 512]},"location_coords":{"type":"torch.Tensor","shape": [1, 2]}}}'
```


```bash title="Generating vLLM configuration with input specification as file"
cat << EOF > ./input_sample.json
{
  "target": "pixel_values",
  "data": {
    "pixel_values": {
      "type": "torch.Tensor",
      "shape": [6,512,512]
    },
    "location_coords": {
      "type": "torch.Tensor",
      "shape": [1,2]
    }
  }
}
EOF

python vllm_config_generator.py \
   --ttconfig config.yaml \
   -i ./input_sample.json
```
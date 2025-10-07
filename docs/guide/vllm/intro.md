# Serving TerraTorch models with vLLM

TerraTorch models can be served using the [vLLM](https://github.com/vllm-project/vllm) serving engine. The are no special requirements for a model to be served via vLLM except for a properly structured configuration file. Thanks to a feature called IOProcessor plugins, vLLM is capable of processing and generating data in any modality (e.g., geoTiff). In TerraTorch, we define IOProcessor plugins for the most popular tasks (e.g., semantic segmentation).
Please refer to the rest of this documentation for more information on the model configuration as well as a available IOProcessor plugins and how to use them.


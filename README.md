[![Build Status](https://v3.travis.ibm.com/GeoFM-Finetuning/terratorch.svg?token=tGjexp9kUqxxz24pGxYt&branch=main)](https://v3.travis.ibm.com/GeoFM-Finetuning/terratorch)
# TerraTorch

:book: [Documentation](https://IBM.github.io/terratorch/)

## Overview
TerraTorch is a library based on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and the [TorchGeo](https://github.com/microsoft/torchgeo) domain library
for geospatial data. TerraTorch’s main purpose is to provide a flexible fine-tuning framework for Geospatial Foundation Models, which can be interacted with at different abstraction levels.

The library provides:
    • Easy access to open source pre-trained Geospatial Foundation Model backbones (e.g., [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M), [SatMAE](https://sustainlab-group.github.io/SatMAE/)
 and [ScaleMAE](https://github.com/bair-climate-initiative/scale-mae) and other backbones available in the [timm](https://github.com/huggingface/pytorch-image-models) (Pytorch image models)
 or [SMP](https://github.com/qubvel/segmentation_models.pytorch) (Pytorch Segmentation models with pre-training backbones) packages.
    • Flexible trainers for Image Segmentation, Classification and Pixel Wise Regression fine-tuning tasks
    • Launching of fine-tuning tasks through flexible configuration files

## Install
### Pip
In order to use th file `pyproject.toml` it is necessary to guarantee `pip>=21.8`. If necessary upgrade `pip` using `python -m pip install --upgrade pip`. 

Install the library with `pip install git+ssh://git@github.com:IBM/terratorch.git`

To install as a developer (e.g. to extend the library) clone this repo, install dependencies using `pip install -r requirements.txt` and run `pip install -e .`

### Conda
It is also possible to restore a conda environment with the same dependencies from `terratorch_env.yaml`:
`conda env create -f <file_name>.yml -n <environment_name>`

## Quick start

To get started, check out the [quick start guide](https://ibm.github.io/terratorch/quick_start)

## For developers

Check out the [architecture overview](https://ibm.github.io/terratorch/architecture)

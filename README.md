[![Build Status](https://v3.travis.ibm.com/GeoFM-Finetuning/terratorch.svg?token=tGjexp9kUqxxz24pGxYt&branch=main)](https://v3.travis.ibm.com/GeoFM-Finetuning/terratorch)
# Terratorch

:book: [Documentation](https://IBM.github.io/terratorch/)

## Overview

The purpose of this library is twofold:

1. To integrate prithvi backbones into the TorchGeo framework
2. To provide generic LightningDataModules that can be built at runtime
3. To build a flexible fine-tuning framework based on TorchGeo which can be interacted with at different abstraction levels.

This library provides:

- All the functionality in TorchGeo
- Easy access to prithvi backbones
- Flexible trainers for Image Segmentation and Pixel Wise Regression (more in progress)
- Launching of fine-tuning tasks through powerful configuration files

A good starting place is familiarization with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), which this project is built on, and to a certain extent [TorchGeo](https://torchgeo.readthedocs.io/en/stable/)

## Install
### Pip
In order to use th file `pyproject.toml` it is necessary to guarantee `pip>=21.8`. If necessary upgrade `pip` using `python -m pip install --upgrade pip`. 

Install the library with `pip install git+ssh://git@github.com:IBM/terratorch.git`

To install as a developer (e.g. to extend the library) clone this repo, install dependencies using `pip install -r requirements.txt` and run `pip install -e .`

### Conda
It is also possible to restore a conda environment with the same dependencies from `terratorch_env.yaml`:
`conda env create -f <file_name>.yml -n <environment_name>`

## Quick start

To get started, check out the [quick start guide](https://pages.github.ibm.com/GeoFM-Finetuning/terratorch/quick_start)

## For developers

Check out the [architecture overview](https://pages.github.ibm.com/GeoFM-Finetuning/terratorch/quick_start/architecture)

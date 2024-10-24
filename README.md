# TerraTorch

:book: [Documentation](https://IBM.github.io/terratorch/)

## Overview
TerraTorch is a library based on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and the [TorchGeo](https://github.com/microsoft/torchgeo) domain library
for geospatial data. 

TerraTorch’s main purpose is to provide a flexible fine-tuning framework for Geospatial Foundation Models, which can be interacted with at different abstraction levels.

The library provides:

- Easy access to open source pre-trained Geospatial Foundation Model backbones (e.g., [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M), [SatMAE](https://sustainlab-group.github.io/SatMAE/) and [ScaleMAE](https://github.com/bair-climate-initiative/scale-mae), other backbones available in the [timm](https://github.com/huggingface/pytorch-image-models) (Pytorch image models) or [SMP](https://github.com/qubvel/segmentation_models.pytorch) (Pytorch Segmentation models with pre-training backbones) packages, as well as fine-tuned models such as [granite-geospatial-biomass](https://huggingface.co/ibm-granite/granite-geospatial-biomass)
- Flexible trainers for Image Segmentation, Classification and Pixel Wise Regression fine-tuning tasks
- Launching of fine-tuning tasks through flexible configuration files

## Install
### Pip
In order to use th file `pyproject.toml` it is necessary to guarantee `pip>=21.8`. If necessary upgrade `pip` using `python -m pip install --upgrade pip`. 

For a stable point-release, use `pip install terratorch`. 
If you prefer to get the most recent version of the main branch, install the library with `pip install git+https://github.com/IBM/terratorch.git`.

Another alternative is to install using [pipx](https://github.com/pypa/pipx) via `pipx install terratorch`, which creates an isolated environment and allows the user to run the application as 
a common CLI tool, with no need of installing dependencies or activating environments. 

TerraTorch requires gdal to be installed, which can be quite a complex process. If you don't have GDAL set up on your system, we reccomend using a conda environment and installing it with `conda install -c conda-forge gdal`.

To install as a developer (e.g. to extend the library) clone this repo, install dependencies using `pip install -r requirements/required.txt -r requirements/dev.txt` and run `pip install -e .`
To install terratorch with partial (work in development) support for Weather Foundation Models, `pip install -e .[wxc]`, which currently works just for `Python >= 3.11`. 

## Quick start

To get started, check out the [quick start guide](https://ibm.github.io/terratorch/quick_start)

## For developers

Check out the [architecture overview](https://ibm.github.io/terratorch/architecture)
A simple hint for any contributor. If you want to met the GitHub DCO checks, just do your commits as below:
```
git commit -s -m <message>
```
It will sign the commit with your ID and the check will be met. 

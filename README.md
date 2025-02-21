
<img src="https://github.com/user-attachments/assets/f8c9586f-6220-4a53-9669-2aee3300b492" alt="TerraTorch"  width="400"/>

## Overview
TerraTorch is a library based on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and the [TorchGeo](https://github.com/microsoft/torchgeo) domain library
for geospatial data. 

TerraTorch’s main purpose is to provide a flexible fine-tuning framework for Geospatial Foundation Models, which can be interacted with at different abstraction levels. The library provides:

- Convenient modelling tools:
    - Flexible trainers for Image Segmentation, Classification and Pixel Wise Regression fine-tuning tasks
    - Model factories that allow to easily combine backbones and decoders for different tasks
    - Ready-to-go datasets and datamodules that require only to point to your data with no need of creating new custom classes
    - Launching of fine-tuning tasks through CLI and flexible configuration files, or via jupyter notebooks
- Easy access to:
    - Open source pre-trained Geospatial Foundation Model backbones:
      * [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)
      * [SatMAE](https://sustainlab-group.github.io/SatMAE/)
      * [ScaleMAE](https://github.com/bair-climate-initiative/scale-mae)
      * Satlas (as implemented in [TorchGeo](https://github.com/microsoft/torchgeo))
      * DOFA (as implemented in [TorchGeo](https://github.com/microsoft/torchgeo))
      * SSL4EO-L and SSL4EO-S12 models (as implemented in [TorchGeo](https://github.com/microsoft/torchgeo))
      * [Clay](https://github.com/Clay-foundation/model)
    - Backbones available in the [timm](https://github.com/huggingface/pytorch-image-models) (Pytorch image models)
    - Decoders available in [SMP](https://github.com/qubvel/segmentation_models.pytorch) (Pytorch Segmentation models with pre-training backbones) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) packages
    - Fine-tuned models such as [granite-geospatial-biomass](https://huggingface.co/ibm-granite/granite-geospatial-biomass)
    - All GEO-Bench datasets and datamodules
    - All [TorchGeo](https://github.com/microsoft/torchgeo) datasets and datamodules 

## Install
### Pip
In order to use th file `pyproject.toml` it is necessary to guarantee `pip>=21.8`. If necessary upgrade `pip` using `python -m pip install --upgrade pip`. 

For a stable point-release, use `pip install terratorch==<version>`.

[comment]: <If you prefer to get the most recent version of the main branch, install the library with `pip install git+https://github.com/IBM/terratorch.git`.>
To get the most recent version of the main branch, install the library with `pip install git+https://github.com/IBM/terratorch.git`.

[comment]: <Another alternative is to install using [pipx](https://github.com/pypa/pipx) via `pipx install terratorch`, which creates an isolated environment and allows the user to run the application as a common CLI tool, with no need of installing dependencies or activating environments.>

TerraTorch requires gdal to be installed, which can be quite a complex process. If you don't have GDAL set up on your system, we reccomend using a conda environment and installing it with `conda install -c conda-forge gdal`.

To install as a developer (e.g. to extend the library):
```
git clone https://github.com/IBM/terratorch.git
cd terratorch
pip install -r requirements_test.txt
conda install -c conda-forge gdal
pip install -e .
```

To install terratorch with partial (work in development) support for Weather Foundation Models, `pip install -e .[wxc]`, which currently works just for `Python >= 3.11`. 

## Documentation

To get started, check out the [quick start guide](https://ibm.github.io/terratorch/quick_start).

Developers, check out the [architecture overview](https://ibm.github.io/terratorch/architecture).

## Contributing

This project welcomes contributions and suggestions.

A simple hint for any contributor. If you want to meet the GitHub DCO checks, just do your commits as below:
```
git commit -s -m <message>
```
It will sign the commit with your ID and the check will be met. 

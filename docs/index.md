# Welcome to TerraTorch
<img src="figs/logo.png#only-light" alt="TerraTorch"  width="400"/>
<img src="figs/logo_grey.png#only-dark" alt="TerraTorch"  width="400"/>

## Overview

The purpose of this package is to build a flexible fine-tuning framework for Geospatial Foundation Models (GFMs) based on TorchGeo and Lightning
which can be employed at different abstraction levels. It currently supports models from the
[Prithvi](https://huggingface.co/ibm-nasa-geospatial)
and [Granite](https://huggingface.co/ibm-granite/granite-geospatial-land-surface-temperature) series, and also have been tested with others models available on HuggingFace. 

This library provides:

- All the functionality in TorchGeo.
- Easy access to Prithvi, timm and smp backbones.
- Flexible trainers for Image Segmentation, Pixel Wise Regression and Classification (more in progress).
- Launching of fine-tuning tasks through powerful configuration files.

A good starting place is familiarization with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), which this project is built on. 
[TorchGeo](https://torchgeo.readthedocs.io/en/stable/) is also an important complementary reference. 

Check out the [architecture overview](architecture.md) for a general description about how TerraTorch is
organized. 

## Quick start

To get started, check out the [quick start guide](quick_start.md)

## License
TerraTorch is distributed under the terms of License Apache 2.0, see [here](license.md) for more details. 

# Welcome to TerraTorch

## Overview

The purpose of this library is twofold:

1. To integrate prithvi backbones into the TorchGeo framework.
2. To provide generic LightningDataModules that can be built at runtime.
3. To build a flexible fine-tuning framework based on TorchGeo which can be interacted with at different abstraction levels.

This library provides:

- All the functionality in TorchGeo.
- Easy access to prithvi, timm and smp backbones.
- Flexible trainers for Image Segmentation, Pixel Wise Regression and Classification (more in progress).
- Launching of fine-tuning tasks through powerful configuration files.

A good starting place is familiarization with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), which this project is built on. 
[TorchGeo](https://torchgeo.readthedocs.io/en/stable/) is also an important complementary reference. 

Check out the [architecture overview](architecture.md) for a general description about how TerraTorch is
organized. 

## Quick start

To get started, check out the [quick start guide](quick_start.md)

## License
TerraTorch is distributed under the terms of License Apache 2.0, see [here](licence.md) for more details. 

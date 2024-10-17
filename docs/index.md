# Welcome to Terratorch

## Overview

The purpose of this library is twofold:

1. To integrate prithvi backbones into the TorchGeo framework
2. To provide generic LightningDataModules that can be built at runtime
3. To build a flexible fine-tuning framework based on TorchGeo which can be interacted with at different abstraction levels.

This library provides:

- All the functionality in TorchGeo
- Easy access to prithvi, timm and smp backbones
- Flexible trainers for Image Segmentation and Pixel Wise Regression (more in progress)
- Launching of fine-tuning tasks through powerful configuration files

A good starting place is familiarization with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), which this project is built on, and to a certain extent [TorchGeo](https://torchgeo.readthedocs.io/en/stable/)

## Quick start

To get started, check out the [quick start guide](quick_start.md)

## For developers

Check out the [architecture overview](architecture.md)

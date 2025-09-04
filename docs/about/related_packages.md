# Related packages

TerraTorch is part of a larger open-source ecosystem for geospatial AI.

## Frameworks

TerraTorch uses [Lightning](https://lightning.ai/docs/pytorch/latest/) as a training and inference engine. 
Lightning handles many basic tasks like GPU allocation, logging and more. It uses [PyTorch](https://pytorch.org) as the machine learning framework.

The tasks and data modules in TerraTorch are based on [TorchGeo](https://torchgeo.readthedocs.io/en/latest/). 
Therefore, TorchGeo datasets are directly compatible with TerraTorch. 

## Data Tools

[Xarray](https://docs.xarray.dev/en/stable/) is a well suited tool for handling multidimensional data and supports lazy loading and more.  
It's extension [rioxarray](https://corteva.github.io/rioxarray/html/rioxarray.html) supports reading or writing tif files and handles geospatial functionalities like the CRS.  

For geospatial table data like polygons, [GeoPandas](https://geopandas.org/en/stable/) is a well suited tool.

## Model Libraries

TerraTorch includes meta registries for backbones, decoders and more. 
Models from [Pytorch Image Models (timm)](https://huggingface.co/timm) and [Segmentation Models PyTorch (SMP)](https://segmentation-modelspytorch.readthedocs.io/en/latest/) are directly available in TerraTorch.

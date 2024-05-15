# Data
We rely on TorchGeo for the implementation of datasets and data modules.

Check out [the TorchGeo tutorials on datasets](https://torchgeo.readthedocs.io/en/stable/tutorials/custom_raster_dataset.html) for more in depth information.

In general, it is reccomended you create a TorchGeo dataset specifically for your dataset. This gives you complete control and flexibility on how data is loaded, what transforms are done over it, and even how it is plotted if you log with tools like TensorBoard.

TorchGeo provides `GeoDataset` and `NonGeoDataset`.

- If your data is already nicely tiled and ready for consumption by a neural network, you can inherit from `NonGeoDataset`. This is essentially a wrapper of a regular torch dataset.
- If your data consists of large GeoTiffs you would like to sample from during training, you can leverage the powerful `GeoDataset` from torch. This will automatically align your input data and labels and enable a variety of geo-aware samplers.

## Using Datasets already implemented in TorchGeo
Using existing TorchGeo DataModules is very easy! Just plug them in!
For instance, to use the `EuroSATDataModule`, in your config file, set the data as:
```yaml
data:
  class_path: torchgeo.datamodules.EuroSATDataModule
  init_args:
    batch_size: 32
    num_workers: 8
  dict_kwargs:
    root: /dccstor/geofm-pre/EuroSat
    download: True
    bands:
      - B02
      - B03
      - B04
      - B08A
      - B09
      - B10
```
Modifying each parameter as you see fit.

You can also do this outside of config files! Simply instantiate the data module as normal and plug it in.

!!! warning
    To define `transforms` to be passed to DataModules from TorchGeo from config files, you must use the following format:
    ```yaml
    data:
    class_path: terratorch.datamodules.TorchNonGeoDataModule
    init_args:
      cls: torchgeo.datamodules.EuroSATDataModule
      transforms:
        - class_path: albumentations.augmentations.geometric.resize.Resize
          init_args:
            height: 224
            width: 224
        - class_path: ToTensorV2
    ```
    Note the class_path is `TorchNonGeoDataModule` and the class to be used is passed through `cls` (there is also a `TorchGeoDataModule` for geo modules).
    This has to be done as the `transforms` argument is passed through `**kwargs` in TorchGeo, making it difficult to instantiate with LightningCLI.
    See more details below.

:::terratorch.datamodules.torchgeo_data_module


## Generic datasets and data modules
For the `NonGeoDataset` case, we also provide "generic" datasets and datamodules. These can be used when you would like to load data from given directories, in a style similar to the [MMLab](https://github.com/open-mmlab) libraries.

### Generic Datasets
#### :::terratorch.datasets.generic_pixel_wise_dataset
#### :::terratorch.datasets.generic_scalar_label_dataset

### Generic Data Modules
#### :::terratorch.datamodules.generic_pixel_wise_data_module
#### :::terratorch.datamodules.generic_scalar_label_data_module
## Custom datasets and data modules
Below is a documented example of how a custom dataset and data module class can be implemented.

#### :::terratorch.datasets.fire_scars

#### :::terratorch.datamodules.fire_scars
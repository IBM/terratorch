# Data Processing

In our workflow, we leverage TorchGeo to implement datasets and data modules, ensuring robust and flexible data handling. For a deeper dive into working with datasets using TorchGeo, please refer to the [TorchGeo tutorials on datasets](https://torchgeo.readthedocs.io/en/stable/tutorials/custom_raster_dataset.html).

In most cases, it’s best to create a custom TorchGeo dataset tailored to your specific data. Doing so gives you complete control over:
- Data Loading: Customize how your data is read and organized.
- Transforms: Decide which preprocessing or augmentation steps to apply.
- Visualization: Define custom plotting methods (for example, when logging with TensorBoard).

TorchGeo offers two primary classes to suit different data formats:
- `NonGeoDataset`:
  Use this if your dataset is already split into neatly tiled pieces ready for neural network consumption. Essentially, `NonGeoDataset` is a wrapper around a standard PyTorch dataset, making it straightforward to integrate into your pipeline.
- `GeoDataset`:  
  Opt for this class if your data comes in the form of large GeoTiff files from which you need to sample during training. `GeoDataset` automatically aligns your input data with corresponding labels and supports a range of geo-aware sampling techniques.


In addition to these specialized TorchGeo datasets, TerraTorch offers generic datasets and data modules designed to work with directory-based data structures, similar to those used in MMLab libraries. These generic tools simplify data loading when your data is organized in conventional file directories:
- The Generic Pixel-wise Dataset is ideal for tasks where each pixel represents a sample (e.g., segmentation or dense prediction problems).
- The Generic Scalar Label Dataset is best suited for classification tasks where each sample is associated with a single label.

TerraTorch also provides corresponding generic data modules that bundle the dataset with training, validation, and testing splits, integrating seamlessly with PyTorch Lightning. This arrangement makes it easy to manage data loading, batching, and preprocessing with minimal configuration.

While generic datasets offer a quick start for common data structures, many projects require more tailored solutions. Custom datasets and data modules give you complete control over the entire data handling process—from fine-tuned data loading and specific transformations to enhanced visualization. By developing your own dataset and data module classes, you ensure that every step—from data ingestion to final model input—is optimized for your particular use case. TerraTorch’s examples provide an excellent starting point to build these custom components and integrate them seamlessly into your training pipeline.

For additional examples on fine-tuning a TerraTorch model using these components, please refer to the [Prithvi EO Examples](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) repository.

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

Our custom datasets and data modules are crafted to handle specific data, offering enhanced control and flexibility throughout the workflow. 
In case you want to use TerraTorch on your specific data, we invite you to develop your own dataset and data module classes by following the examples below. 

### Datasets
#### :::terratorch.datasets.biomassters
#### :::terratorch.datasets.burn_intensity
#### :::terratorch.datasets.carbonflux
#### :::terratorch.datasets.forestnet
#### :::terratorch.datasets.fire_scars
#### :::terratorch.datasets.landslide4sense
#### :::terratorch.datasets.m_eurosat
#### :::terratorch.datasets.m_bigearthnet
#### :::terratorch.datasets.m_brick_kiln
#### :::terratorch.datasets.m_forestnet
#### :::terratorch.datasets.m_so2sat
#### :::terratorch.datasets.m_pv4ger
#### :::terratorch.datasets.m_cashew_plantation
#### :::terratorch.datasets.m_nz_cattle
#### :::terratorch.datasets.m_chesapeake_landcover
#### :::terratorch.datasets.m_pv4ger_seg
#### :::terratorch.datasets.m_SA_crop_type
#### :::terratorch.datasets.m_neontree
#### :::terratorch.datasets.multi_temporal_crop_classification
#### :::terratorch.datasets.open_sentinel_map
#### :::terratorch.datasets.openearthmap
#### :::terratorch.datasets.pastis
#### :::terratorch.datasets.sen1floods11
#### :::terratorch.datasets.sen4agrinet
#### :::terratorch.datasets.sen4map

### Datamodules
#### :::terratorch.datamodules.biomassters
#### :::terratorch.datamodules.burn_intensity
#### :::terratorch.datamodules.carbonflux
#### :::terratorch.datamodules.forestnet
#### :::terratorch.datamodules.fire_scars
#### :::terratorch.datamodules.landslide4sense
#### :::terratorch.datamodules.m_eurosat
#### :::terratorch.datamodules.m_bigearthnet
#### :::terratorch.datamodules.m_brick_kiln
#### :::terratorch.datamodules.m_forestnet
#### :::terratorch.datamodules.m_so2sat
#### :::terratorch.datamodules.m_pv4ger
#### :::terratorch.datamodules.m_cashew_plantation
#### :::terratorch.datamodules.m_nz_cattle
#### :::terratorch.datamodules.m_chesapeake_landcover
#### :::terratorch.datamodules.m_pv4ger_seg
#### :::terratorch.datamodules.m_SA_crop_type
#### :::terratorch.datamodules.m_neontree
#### :::terratorch.datamodules.multi_temporal_crop_classification
#### :::terratorch.datamodules.open_sentinel_map
#### :::terratorch.datamodules.openearthmap
#### :::terratorch.datamodules.pastis
#### :::terratorch.datamodules.sen1floods11
#### :::terratorch.datamodules.sen4agrinet
#### :::terratorch.datamodules.sen4map

## Transforms
The transforms module provides a set of specialized image transformations designed to manipulate spatial, temporal, and multimodal data efficiently. 
These transformations allow for greater flexibility when working with multi-temporal, multi-channel, and multi-modal datasets, ensuring that data can be formatted appropriately for different model architectures.

### :::terratorch.datasets.transforms



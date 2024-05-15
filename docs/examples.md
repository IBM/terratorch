# Examples

For some examples of training using the existing tasks, check out the following pages on our github repo:

## From config files (reccomended)

Under `examples/confs`

* Flood Segmentation with Swin: `segmentation_config.yaml`

* Flood Segmentation with ViT: `segmentation_config_vit.yaml`

* Above Ground Biomass (regression): `regression_agb.yaml`

* Multitemporal Crop Segmentation: `multitemporal_crop.yaml`

* Scene Classification: `eurosat.yaml`

* Usage of an SMP backbone `geobench/segmentation/m_chesapeake_landcover_smp_resnet_unet.yaml`

* Usage of a timm backbone `geobench/classification/m_bigearthnet_timm_resnet.yaml`

### From a python file

Under `examples/scripts`

* Above Ground Biomass (regression): `train_generic_dataset_reg.py`

* *experimental* Above Ground Biomass (regression) with a UNet from SMP: `train_generic_dataset_reg_unet.py`

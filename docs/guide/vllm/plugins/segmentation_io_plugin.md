# Segmentation IOProcessor Plugin

This plugin targets segmentation tasks and allows for the input image to be split in tiles of a size that depends on the model and on an arbitrary number of bands.

During initialization, the plugin accesses the model's data module configuration from the vLLM configuration and instantiates a DataModule object dynamically.

This plugin is installed as `terratorch_segmentation`.

## Plugin specification 
### Plugin configuration

This plugin allows for additional configuration data to be passed via the `TERRATORCH_SEGMENTATION_IO_PROCESSOR_CONFIG` environment variable. If set, the variable should contain the plugin configuration in json string format. 

The plugin configuration format is defined in the `PluginConfig` class.

:::terratorch.vllm.plugins.segmentation.types.PluginConfig

### Request Data Format
The input format for for the plugin is defined in the `RequestData` class.

:::terratorch.vllm.plugins.segmentation.types.RequestData


Depending on the values set in `data_format`, the plugin expects `data` to contain a string that complies to the format. Similarly, `out_data_format` controls the data format returned to the user.

### Request Output Format

The output format for the plugin is defined in the `RequestOutput` class.

:::terratorch.vllm.plugins.segmentation.types.RequestOutput

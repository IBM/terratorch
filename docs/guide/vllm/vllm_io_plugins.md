# vLLM IOProcessor Plugins
vLLM's IOProcessor plugins are a mechanism that enables processing of input/output inference data from/to any modality. So, as an example, these plugins allow for the output of a model to be transformed into an image.

TerraTorch provides plugins for the handling of input/output GeoTiff images when serving models via vLLM.

More information can be found in the [vLLM official documentation](https://docs.vllm.ai/en/latest/design/io_processor_plugins.html)

## Using IOProcessor Plugins

IOProcessor plugins are instantiated at vLLM startup time via a dedicated flag `--io_processor_plugin`. The snippet below shows an example of a vLLM server started for serving a TerraTorch model using the `terratorch_segmentation_plugin`.


```bash
vllm serve \
    --model=ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
    --model-impl terratorch \
    --task embed --trust-remote-code \
    --skip-tokenizer-init --enforce-eager \
    --io-processor-plugin terratorch_segmentation
```

Inference requests are then sent to the vLLM server URL under the `/pooling` endpoint.

The format of the request is described below for the `terratorch_segmentation` plugin, where the `model` and `softmax` fields are pre-defined and are only processed by vLLM, while the `data` field is plugin dependent. Refer to the documentation of each plugin to get more information on the request data format.

```python
request_payload = {
    "data": {
        "data": "image_url",
        "data_format": "url",
        "out_data_format": "b64_json",
    },
    "model": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
    "softmax": False,
}
```

## Available TerraTorch IOProcessor plugins


| Plugin name           | Tasks Supported             | Description |
|-----------------------|-----------------------------|-------------|
| [terratorch_segmentation](../plugins/segmentation_io_plugin) | Semantic Segmentation   | Plugin operations: Splits the image in tiles, performs inference on all the tiles and creates a geoTiff out of all the inference outputs.<br>input format: geoTiff<br> output format: geoTiff            |

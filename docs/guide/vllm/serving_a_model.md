# Initializing and serving a model with vLLM

This section shows an example of how to bootstrap a TerraTorch model on vLLM and perform a sample inference. This section assumes that you have prepared your model for serving with vLLM and you have identified the IOProcessor to be used.

The examples in the rest of this document will use the [Prithvi-EO-2.0-300M-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11) model finetuned to segment the extent of floods on Sentinel-2 images from the Sen1Floods11 dataset and the `terratorch_segmentation` IOProcessor plugin. However, the commands can be adapted to work with any other supported models and plugins.

## Starting the vLLM serving instance

The information required to start the serving instance is the model identifier on HuggingFace and the name of the IOProcessor plugin. In this example:
- Model identifier: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`
- IO Processor plugin name: `terratorch_segmentation`

To start the serving instance, run the below command:

```bash title="Starting a vLLM serving instance"
vllm serve ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
--io-processor-plugin terratorch_segmentation \
--model-impl terratorch \
--skip-tokenizer-init \
--enforce-eager
```

The snippet below shows the logs of a successfully initialized vLLM serving instance

```title="vLLM instance ready to serve requests"
(APIServer pid=532339) INFO 10-01 09:01:06 [launcher.py:44] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=532339) INFO 10-01 09:01:06 [launcher.py:44] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=532339) INFO 10-01 09:01:06 [launcher.py:44] Route: /invocations, Methods: POST
(APIServer pid=532339) INFO 10-01 09:01:06 [launcher.py:44] Route: /metrics, Methods: GET
(APIServer pid=532339) INFO:     Started server process [532339]
(APIServer pid=532339) INFO:     Waiting for application startup.
(APIServer pid=532339) INFO:     Application startup complete.
```

## Send An Inference Request To The Model

TerraTorch models can be served in vLLM via the `/pooling` endpoint with the below payload

```python title="vLLM pooling request payload"

request_payload = {
    "data": Any,
    "model": "model_name",
    "softmax": False
}
```

The `data` field accepts any format and its schema is defined by the IOProcessor plugin used. The `softmax=True` is mandatory as it is required for the plugins to receive the raw model output. The `model` field must contain the same model name used for starting the server.

In this example, the format of the `data` field is defined by the `terratorch_segmentation` plugin, and for this example we will use the below request_payload:

```python title="Request payload for the terratorch_segmentation plugin"
request_payload = {
    "data": {
        "data": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff", 
        "data_format": "url",
        "out_data_format": "path",
        "image_format": "geoTiff"
    },
    "model": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
    "softmax": False
}
```

With this payload the IOProcessor plugin will download the input geoTiff from a URL and return the path on local filesystem of the output geoTiff.

Assuming the vLLM server is listening on `localhost:8000` the below snippet shows how to send the inference request and retrieve the output file path.

```python title="Request inference to the vLLM serving instance"
ret = requests.post("http://localhost:8000/pooling", json=request_payload)
response = ret.json()
out_file_path = response["data"]["data"]
```
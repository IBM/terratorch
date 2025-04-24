# TerraTorch-Examples
Examples for fine-tuning Prithvi on EO downstream tasks via TerraTorch

## Notebook examples

### Sen1Floods11 

Dataset: https://github.com/cloudtostreet/Sen1Floods11

Tutorial: [prithvi_v2_eo_300_tl_unet_sen1floods11.ipynb](prithvi_v2_eo_300_tl_unet_sen1floods11.ipynb)

[Open in Colab](https://colab.research.google.com/github/ibm/TerraTorch/blob/main/examples/tutorial/prithvi_v2_eo_300_tl_unet_sen1floods11.ipynb)

Check out the demo of the fine-tuned model (Prithvi EO 2.0): https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-EO-2.0-Sen1Floods11-demo

### HLS Burn Scars

Dataset: https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars

Tutorial: [prithvi_v2_eo_300_tl_unet_burnscars.ipynb](prithvi_v2_eo_300_tl_unet_burnscars.ipynb)

[Open in Colab](https://colab.research.google.com/github/ibm/TerraTorch/blob/main/examples/tutorial/prithvi_v2_eo_300_tl_unet_burnscars.ipynb)

Check out the demo of the fine-tuned model (Prithvi EO 2.0): https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-EO-2.0-BurnScars-demo

### Multi-temporal Crop

Dataset: https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification

Tutorial: [prithvi_v2_eo_300_tl_unet_multitemporal_crop.ipynb](prithvi_v2_eo_300_tl_unet_multitemporal_crop.ipynb)

[Open in Colab](https://colab.research.google.com/github/ibm/TerraTorch/blob/main/examples/tutorial/prithvi_v2_eo_300_tl_unet_multitemporal_crop.ipynb)

Check out the demo of the fine-tuned model (Prithvi EO 1.0): https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-100M-multi-temporal-crop-classification-demo

## Config examples for CLI

Sen1Floods11 config: [prithvi_v2_eo_300_tl_unet_sen1floods11.yaml](configs%2Fprithvi_v2_eo_300_tl_unet_sen1floods11.yaml)

Fine-tuning via CLI:
```shell
terratorch fit -c configs/prithvi_v2_eo_300_tl_unet_sen1floods11.yaml
```

HLS Burn Scars config: [prithvi_v2_eo_300_tl_unet_burnscars.yaml](configs%2Fprithvi_v2_eo_300_tl_unet_burnscars.yaml)

Fine-tuning via CLI:
```shell
terratorch fit -c configs/prithvi_v2_eo_300_tl_unet_burnscars.yaml
```

Multi-temporal Crop config: 

Fine-tuning via CLI:
```shell
terratorch fit -c configs/prithvi_v2_eo_300_tl_unet_multitemporal_crop.yaml
```

## More information

Prithvi models: https://huggingface.co/ibm-nasa-geospatial

Prithvi EO 2.0 examples: https://github.com/NASA-IMPACT/Prithvi-EO-2.0

More TerraTorch examples: https://github.com/IBM/terratorch/tree/main/examples

TerraTorch docs: https://ibm.github.io/terratorch/

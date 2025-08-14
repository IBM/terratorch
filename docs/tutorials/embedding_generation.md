# Embedding Generation 

Handles workflows where a model backbone from the TerraTorch backbone registry is used to extract and save embeddings from a frozen backbone.

It is implemented as an embedding generation task, designed to be run in a PyTorch Lightning predict workflow, for example:

```
python -m terratorch predict -c terramind_embeddings.yaml
```

A sample configuration for TerraMind embedding generation can be found under `examples/confs/embedding_generation`

The current embedding generation task supports both single- and multi-modal inputs, depending on the chosen backbone. Temporal data can optionally be processed by non-temporal models via the temporal wrapper, which encodes each timestep individually before saving the embeddings.

Supported output formats include GeoTIFF (.tif) and GeoParquet (.parquet).

Intermediate layer outputs can be saved by specifying the `layers` parameter, which defines from which intermediate layers to extract embeddings.

## **Embedding Aggregation / Pooling**

The `embedding_pooling` option defines a downsampling strategy applied to the extracted embeddings before saving.  
This can be used, for example, to:
- Average all patch embeddings to produce a single image embedding.
- Select only the CLS token from a ViT backbone.
- Apply spatial reduction to CNN feature maps.

If set to `None`, the full embedding from the specified layers is saved.

If an aggregation strategy is not compatible with the model backbone, an error is raised.

Currently supported:
**For ViT backbones:**
- `vit_mean` — Mean across the patch dimension (excluding CLS token if present).
- `vit_max` — Max across the patch dimension (excluding CLS token if present).
- `vit_min` — Min across the patch dimension (excluding CLS token if present).
- `vit_cls` — Select only the CLS token.

**For CNN backbones:**
- `cnn_mean` — Spatial mean across height and width.
- `cnn_max` — Spatial max across height and width.
- `cnn_min` — Spatial min across height and width.
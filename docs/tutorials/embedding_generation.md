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
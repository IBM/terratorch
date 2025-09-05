### TemporalWrapper

The `TemporalWrapper` allows any TerraTorch backbone to process a temporal stack of inputs and defines how features are returned. Inputs can be given either as a tensor with shape `[B, C, T, H, W]`, or as a dict of modalities `{modality: [B, C_mod, T, H, W]}`, depending on what the backbone expects. Each timestep is forwarded independently through the backbone, and the resulting outputs are collected. The temporal aggregation strategy is controlled by the `pooling` parameter:

- `"keep"` — preserves the individual timestep outputs and returns them as a temporal stack.  
- `"concat"` — flattens the temporal axis by concatenating features from all timesteps along the channel/feature dimension.  
- `"mean"` — averages the features across timesteps.  
- `"max"` — takes the element-wise maximum across timesteps.  
- `"diff"` — computes the difference between the first two timesteps (`t0 − t1`), requiring at least two frames.  

The wrapped backbone may produce a single tensor, a list or tuple of multi-scale features, or a dictionary of modalities. In all cases, the `TemporalWrapper` applies pooling and post-processing consistently. For example, when the model outputs multimodal features (e.g., TerraMind), each modality is processed independently and returned in a multimodal dict.

Models wrapped with `TemporalWrapper` can be used as backbones in TerraTorch encoder–decoder pipelines. Select a temporal aggregation strategy (`mean`, `max`, `diff`) or `concat`. For `pooling='concat'`, specify `n_timestamps` so that `backbone.out_channels` can be computed at build time for decoders/necks. See a minimal example under `examples/notebooks/TemporalWrapper.ipynb`
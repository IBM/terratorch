# Registries

TerraTorch keeps a set of registries which map strings to instances of those strings. They can be imported from `terratorch.registry`.

!!! info
    If you are using tasks with existing models, you may never have to interact with registries directly. The [model factory](models.md#model-factories) will handle interactions with registries.

Registries behave like python sets, exposing the usual `contains` and `iter` operations. This means you can easily operate on them in a pythonic way, such as  `"model" in registry` or `list(registry)`.

To create the desired instance, registries expose a `build` method, which accepts the name and the arguments to be passed to the constructor.

```python title="Using registries"
from terratorch import BACKBONE_REGISTRY

# find available prithvi models
print([model_name for model_name in BACKBONE_REGISTRY if "prithvi" in model_name])
>>> ['timm_prithvi_swin_B', 'timm_prithvi_swin_L', 'timm_prithvi_vit_100', 'timm_prithvi_vit_300', 'timm_prithvi_vit_tiny']

# show all models with list(BACKBONE_REGISTRY)

# check a model is in the registry
"timm_prithvi_swin_B" in BACKBONE_REGISTRY
>>> True

# without the prefix, all internal registries will be searched until the first match is found
"prithvi_swin_B" in BACKBONE_REGISTRY
>>> True

# instantiate your desired model
# the backbone registry prefix (in this case 'timm') is optional
# in this case, the underlying registry is timm, so we can pass timm arguments to it
model = BACKBONE_REGISTRY.build("prithvi_vit_100", num_frames=1, pretrained=True)

# instantiate your model with more options, for instance, passing weights of your own through timm
model = BACKBONE_REGISTRY.build(
    "prithvi_vit_100", num_frames=1, pretrained=True, pretrained_cfg_overlay={"file": "<path to weights>"}
)
# Rest of your PyTorch / PyTorchLightning code

```

## MultiSourceRegistries

`BACKBONE_REGISTRY` and `DECODER_REGISTRY` are special registries which dynamically aggregate multiple registries. They behave as if they were a single large registry by searching over multiple registries.

For instance, the `DECODER_REGISTRY` holds the `TERRATORCH_DECODER_REGISTRY`, which is responsible for decoders implemented in terratorch, as well as the `SMP_DECODER_REGISTRY` and the `MMSEG_DECODER_REGISTRY` (if mmseg is installed).

To make sure you access the object from a particular registry, you may prepend your string with the prefix from that registry.

```python
from terratorch import DECODER_REGISTRY

# decoder registries always take at least one extra argument, the channel list with the channel dimension of each embedding passed to it
DECODER_REGISTRY.build("FCNDecoder", [32, 64, 128])

DECODER_REGISTRY.build("terratorch_FCNDecoder", [32, 64, 128])

# Find all prefixes
DECODER_REGISTRY.keys()
>>> odict_keys(['terratorch', 'smp', 'mmseg'])
```

If a prefix is not added, the `MultiSourceRegistry` will search each registry in the order it was added (starting with the `TERRATORCH_` registry) until it finds the first match.

For both of these registries, only `TERRATORCH_X_REGISTRY` is mutable. To register backbones or decoders to terratorch, you should decorate the constructor function (or the model class itself) with `@TERRATORCH_DECODER_REGISTRY.register` or `@TERRATORCH_BACKBONE_REGISTRY.register`.

To add a new registry to these top level registries, you should use the `.register` method, taking the register and the prefix that will be used for it.

### :::terratorch.registry.registry.MultiSourceRegistry

### :::terratorch.registry.registry.Registry

## Other Registries

Additionally, terratorch has the `NECK_REGISTRY`, where all necks must be registered, and the `MODEL_FACTORY_REGISTRY`, where all model factories must be registered.

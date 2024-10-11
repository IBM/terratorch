# Models

To interface with terratorch tasks correctly, models must conform to the [Model][terratorch.models.model.Model] ABC:

::: terratorch.models.model.Model

and have a forward method which returns a [ModelOutput][terratorch.models.model.ModelOutput]:

:::terratorch.models.model.ModelOutput


## Model Factories

In order to be used by tasks, models must have a Model Factory which builds them.

Factories must conform to the [ModelFactory][terratorch.models.model.ModelFactory] ABC:

::: terratorch.models.model.ModelFactory

You most likely do not need to implement your own model factory, unless you are wrapping another library which generates full models.

For most cases, the [encoder decoder factory](encoder_decoder_factory.md) can be used to combine a backbone with a decoder.

To add new backbones or decoders, to be used with the [encoder decoder factory](encoder_decoder_factory.md) they should be [registered](registry.md). 

To add a new model factory, it should be registered in the `MODEL_FACTORY_REGISTRY`.

## Adding a new model
To add a new backbone, simply create a class and annotate it (or a constructor function that instantiates it) with `@TERRATORCH_BACKBONE_FACTORY.register`. 

The model will be registered with the same name as the function. To create many model variants from the same class, the reccomended approach is to annotate a constructor function from each with a fully descriptive name.

```python
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY, BACKBONE_REGISTRY

from torch import nn

# make sure this is in the import path for terratorch
@TERRATORCH_BACKBONE_REGISTRY.register
class BasicBackbone(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer = nn.Linear(224*224, out_channels)
        self.out_channels = [out_channels]

    def forward(self, x):
        return self.layer(self.flatten(x))

# you can build directly with the TERRATORCH_BACKBONE_REGISTRY
# but typically this will be accessed from the BACKBONE_REGISTRY
>>> BACKBONE_REGISTRY.build("BasicBackbone", out_channels=64)
BasicBackbone(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (layer): Linear(in_features=50176, out_features=64, bias=True)
)

@TERRATORCH_BACKBONE_REGISTRY.register
def basic_backbone_128():
    return BasicBackbone(out_channels=128)

>>> BACKBONE_REGISTRY.build("basic_backbone_128")
BasicBackbone(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (layer): Linear(in_features=50176, out_features=128, bias=True)
)
```

Adding a new decoder can be done in the same way with the `TERRATORCH_DECODER_REGISTRY`.

!!! info
    All decoders will be passed the channel_list as the first argument for initialization.
# Models

To interface with terratorch tasks correctly, models must inherit from the [Model][terratorch.models.model.Model] parent class
and have a forward method which returns an object [ModelOutput][terratorch.models.model.ModelOutput]:

## Model Factories

In order to be used by tasks, models must have a Model Factory which builds them.
Factories must conform to the [ModelFactory][terratorch.models.model.ModelFactory] parent class. 

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

    To pass your own path from where to load the weights with the PrithviModelFactory, you can make use of timm's `pretrained_cfg_overlay`.
    E.g. to pass a local path, you can pass the parameter `backbone_pretrained_cfg_overlay = {"file": "<local_path>"}` to the model factory.
    
    Besides `file`, you can also pass `url`, `hf_hub_id`, amongst others. Check timm's documentation for full details.

# Adding new model types
Adding new model types is as simple as creating a new factory that produces models. See for instance the example below for a potential `SMPModelFactory`
```python
from terratorch.models.model import register_factory

@register_factory
class SMPModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module,
        in_channels: int,
        **kwargs,
    ) -> Model:
       
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=in_channels, classes=1)
        return SMPModelWrapper(model)

@register_factory
class SMPModelWrapper(Model, nn.Module):
    def __init__(self, smp_model) -> None:
        super().__init__()
        self.smp_model = smp_model

    def forward(self, *args, **kwargs):
        return ModelOutput(self.smp_model(*args, **kwargs).squeeze(1))

    def freeze_encoder(self):
        pass

    def freeze_decoder(self):
        pass
```

# Custom modules with CLI

Custom modules must be in the import path in order to be registered in the appropriate registries. 

In order to do this without modifying the code when using the CLI, you may place your modules under a `custom_modules` directory. This must be in the directory from which you execute terratorch.

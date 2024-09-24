# Models

## Prithvi backbones

We provide access to the Prithvi backbones through integration with `timm`.

By passing `features_only=True`, you can conveniently get access to a model that outputs the features produced at each layer of the model.

Passing `features_only=False` will let you access the full original model.

```python title="Instantiating a prithvi backbone from timm"
import timm
import terratorch # even though we don't use the import directly, we need it so that the models are available in the timm registry

# find available prithvi models by name
print(timm.list_models("prithvi*"))
# and those with pretrained weights
print(timm.list_pretrained("prithvi*"))

# instantiate your desired model with features_only=True to obtain a backbone
model = timm.create_model(
    "prithvi_vit_100", num_frames=1, pretrained=True, features_only=True
)

# instantiate your model with weights of your own
model = timm.create_model(
    "prithvi_vit_100", num_frames=1, pretrained=True, pretrained_cfg_overlay={"file": "<path to weights>"}, features_only=True
)
# Rest of your PyTorch / PyTorchLightning code

```

We also provide a model factory that can build a task specific model for a downstream task based on a Prithvi backbone.

By passing a list of bands being used to the constructor, we automatically filter out unused bands, and randomly initialize weights for new bands that were not pretrained on.

!!! info

    To pass your own path from where to load the weights with the PrithviModelFactory, you can make use of timm's `pretrained_cfg_overlay`.
    E.g. to pass a local path, you can pass the parameter `backbone_pretrained_cfg_overlay = {"file": "<local_path>"}` to the model factory.
    
    Besides `file`, you can also pass `url`, `hf_hub_id`, amongst others. Check timm's documentation for full details.

:::terratorch.models.backbones.select_patch_embed_weights

## Decoders
### :::terratorch.models.decoders.fcn_decoder
### :::terratorch.models.decoders.identity_decoder
### :::terratorch.models.decoders.upernet_decoder

## Heads
### :::terratorch.models.heads.regression_head
### :::terratorch.models.heads.segmentation_head
### :::terratorch.models.heads.classification_head

## Auxiliary Heads
### :::terratorch.models.model.AuxiliaryHead

## Model Output
### :::terratorch.models.model.ModelOutput

## Model Factory
### :::terratorch.models.PrithviModelFactory
### :::terratorch.models.SMPModelFactory

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
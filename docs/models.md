# Models

To interface with terratorch tasks correctly, models must conform to the [Model][terratorch.models.Model] ABC:

::: terratorch.models.Model

and have a forward method which returns a [ModelOutput][terratorch.models.model.ModelOutput]:

:::terratorch.models.model.ModelOutput.

In order to be used by tasks, models must have a Model Factory which builds them.

Factories must conform to the [ModelFactory][terratorch.models.model.ModelFactory] ABC:

::: terratorch.models.model.ModelFactory

You most likely do not need to implement your own model factory, unless you are wrapping another library which generates full models.

For most cases, the [encoder decoder factory](encoder_decoder_factory.md) can be used to combine a backbone with a decoder.

To add new backbones or decoders, they should be [registered](registry.md). 

## Adding a new model
TODO
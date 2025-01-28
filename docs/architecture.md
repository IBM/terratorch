# Overview (for developers)

The main goal of the design is to extend TorchGeo's existing tasks to be able to handle Prithvi backbones with appropriate decoders and heads.
At the same time, we wish to keep the existing TorchGeo functionality intact so it can be leveraged with pretrained models that are already included.

We achieve this by making new tasks that accept model factory classes, containing a `build_model` method. This strategy in principle allows arbitrary models to be trained for these tasks, given they respect some reasonable minimal interface.
Together with this, we provide the [EncoderDecoderFactory][terratorch.models.encoder_decoder_factory.EncoderDecoderFactory], which should enable users to plug together different Encoders and Decoders, with the aid of Necks for intermediate operations.

Additionally, we extend TorchGeo with generic datasets and datamodules which can be defined at runtime, rather than requiring classes to be defined beforehand.

The glue that holds everything together is [LightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html#lightning.pytorch.cli.LightningCLI), allowing the model, datamodule and Lightning Trainer to be instantiated from a config file or from the CLI. We make extensive use of for training and inference.

Initial reading for a full understanding of the platform includes:

- Familiarity with [PyTorch Lightning](https://lightning.ai/pytorch-lightning)
- Familiarity with [TorchGeo](https://torchgeo.readthedocs.io/en/stable/)
- Familiarity with [LightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html#lightning.pytorch.cli.LightningCLI)

## Tasks

Tasks are the main coordinators for training and inference for specific tasks. They are LightningModules that contain a model and abstract away all the logic for training steps, metric computation and inference.

One of the most important design decisions was delegating the model construction to a model factory. This has a few advantages:
    
- Avoids code repetition among tasks - different tasks can use the same factory
- Prefers composition over inheritance
- Allows new models to be easily added by introducing new factories

Models are expected to be `torch.nn.Module`s and implement the [Model][terratorch.models.model.Model] interface, providing:
    
- `freeze_encoder()`
- `freeze_decoder()`
- `forward()`

Additionally, the `forward()` method is expected to return an object of type [ModelOutput][terratorch.models.model.ModelOutput], containing the main head's output, as well as any additional auxiliary outputs. The names of these auxiliary heads are matched with the names of the provided auxiliary losses.

### Models

Models constructed by the [EncoderDecoderFactory][terratorch.models.encoder_decoder_factory.EncoderDecoderFactory] have an internal structure explicitly divided into backbones, necks, decoders and heads. This structure is provided by the [PixelWiseModel][terratorch.models.pixel_wise_model.PixelWiseModel] and [ScalarOutputModel][terratorch.models.scalar_output_model.ScalarOutputModel] classes.

However, as long as models implement the [Model][terratorch.models.model.Model] interface, and return [ModelOutput][terratorch.models.model.ModelOutput] in their forward method, they can take on any structure.

#### :::terratorch.models.pixel_wise_model.PixelWiseModel
#### :::terratorch.models.scalar_output_model.ScalarOutputModel


### EncoderDecoderFactory

We expect this factory to be widely employed by users. With that in mind, we dive deeper into it [here](encoder_decoder_factory.md).

### Loss
For convenience, we provide a loss handler that can be used to compute the full loss (from the main head and auxiliary heads as well).

:::terratorch.tasks.loss_handler

## Generic datasets / datamodules
Refer to the section on [data](data.md)

## Exporting models
Models are saved using the PyTorch format, which basically serializes the model weights using pickle
and store them into a binary file. 

<A future feature would be the possibility to save models in ONNX format, and export them that way. This would bring all the benefits of onnx.>

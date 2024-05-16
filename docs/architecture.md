# Overview (for developers)

The main goal of the design is to extend TorchGeo's existing tasks to be able to handle Prithvi backbones with appropriate decoders and heads.
At the same time, we wish to keep the existing TorchGeo functionality intact so it can be leveraged with pretrained models that are already included.

We achieve this by making new tasks that accept model factory classes, containing a `build_model` method.. This strategy in principle allows arbitrary models to be trained for these tasks, given they respect some reasonable minimal interface.

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

### :::terratorch.models.model.Model

Additionally, the `forward()` method is expected to return an object of type [ModelOutput][terratorch.models.model.ModelOutput], containing the main head's output, as well as any additional auxiliary outputs. The names of these auxiliary heads are matched with the names of the provided auxiliary losses.

### :::terratorch.models.model.ModelOutput

### Models
In the currently existing model implementations, we explicitly divide the models into backbones, decoders and heads. This structure is provided by the [PixelWiseModel][terratorch.models.pixel_wise_model.PixelWiseModel] and [ScalarOutputModel][terratorch.models.scalar_output_model.ScalarOutputModel] classes.

However, as long as models implement the [Model][terratorch.models.model.Model] interface, and return [ModelOutput][terratorch.models.model.ModelOutput] in their forward method, they can take on any structure.

#### :::terratorch.models.pixel_wise_model.PixelWiseModel
#### :::terratorch.models.scalar_output_model.ScalarOutputModel


### Backbones
We decide to leverage `timm` for handling backbones. It is important to understand the advantages and disadvantages of this decision:

#### Advantages
1. `timm` provides an incredibly rich variety of already implemented and validated architectures.
2. `timm` provides an API for creating backbones directly as feature extractors, with the `features_only=True` argument.
3. `timm` provides an existing and powerful model registry and factory.

#### Disadvantages
1. The documentation on using `timm` is reasonable, but the documentation on how to develop with `timm`, particularly for adding new models, is not great.
2. A few hacks had to be put in place to make our existing model architectures play nicely with the `features_only=True` functionality.
    1. In particular, we allow models to define a `prepare_features_for_image_model` function, which is called on the model features just before they are passed to the decoder. This function can be defined on the backbone code itself, or passed to the [PixelWiseModel][terratorch.models.pixel_wise_model.PixelWiseModel] constructor.

### Decoders
Currently, we have implemented a simple Fully Convolutional Decoder as well as an UperNetDecoder, which exactly match the definitions in the MMSeg framework.
This was mostly done to ensure we could replicate the results from that framework.

However, libraries such as [pytorch-segmentation](https://github.com/yassouali/pytorch-segmentation) or [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) provide a large set of already implemented decoders that can be leveraged. 

This is probably a reasonable next step in the implementation. In order to do this, a new factory can simply be created which leverages these libraries. See as an example [this section](models.md#adding-new-model-types)

### Heads
In the current implementation, the heads perform the final step in going from the output of the decoder to the final desired output. Often this can just be e.g. a single convolutional head going from the final decoder depth to the number of classes, in the case of segmentation, or to a depth of 1, in the case of regression.

### Loss
For convenience, we provide a loss handler that can be used to compute the full loss (from the main head and auxiliary heads as well).

:::terratorch.tasks.loss_handler

## Generic datasets / datamodules
Refer to the section on [data](data.md)

## Exporting models
A future feature would be the possibility to save models in ONNX format, and export them that way. This would bring all the benefits of onnx.
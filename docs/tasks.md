# Tasks
Tasks provide a convenient abstraction over the training of a model for a specific downstream task. 

They encapsulate the model, optimizer, metrics, loss as well as training, validation and testing steps.

The task expects to be passed a model factory, to which the model_args arguments are passed. The models produced by this model factory should output ModelOutput instances and conform to the Model ABC. This is an easy way to extend these tasks to models other than Pritvhi ones produced by the [PrithviModelFactory][terratorch.models.PrithviModelFactory].

Tasks are best leveraged using config files. Check out some examples [here](./examples.md).

!!! info "Argument parsing in configs"
    Argument parsing of configs relies on argument names and type hints in the code.
    To pass arguments that do not conform to this (e.g. for classes that make use of **kwargs)
    put those arguments in `dict_kwargs` instead of `init_args`.


---
### Multi Temporal Inputs

Multi temporal inputs are also supported! 
However, we leverage albumentations for augmentations, and it does not support multitemporal input.
We currently get around this using the following strategy in the transform:

```yaml
train_transform:
      - class_path: FlattenTemporalIntoChannels
      # your transforms here, wrapped by these other ones
      # e.g. a random flip
      - class_path: albumentations.Flip
      # end of your transforms
      - class_path: ToTensorV2
      - class_path: UnflattenTemporalFromChannels
        init_args:
          n_timesteps: 3 # your number of timesteps here
          # alternatively, n_channels can be specified
```
See an example of this [here](examples.md).

---

## ::: terratorch.tasks.regression_tasks.PixelwiseRegressionTask

## :::terratorch.tasks.segmentation_tasks.SemanticSegmentationTask

## :::terratorch.tasks.classification_tasks.ClassificationTask

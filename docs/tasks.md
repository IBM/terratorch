Tasks provide a convenient abstraction over the training of a model for a specific downstream task. 
They encapsulate the model, optimizer, metrics, loss as well as training, validation and testing steps.
The task expects to be passed a model factory, to which the model_args arguments are passed to instantiate the model that will be trained.
The models produced by this model factory should output ModelOutput instances and conform to the Model ABC.
Tasks are best leveraged using config files, where they are specified in the `model` section under `class_path`. You can check out some examples of config files [here](https://github.com/IBM/terratorch/tree/main/examples/confs).
Below are the details of the tasks currently implemented in TerraTorch (Pixelwise Regression, Semantic Segmentation and Classification). 

:::terratorch.tasks.segmentation_tasks.SemanticSegmentationTask
    options:
        show_source: false
:::terratorch.tasks.regression_tasks.PixelwiseRegressionTask
    options:
        show_source: false
:::terratorch.tasks.classification_tasks.ClassificationTask
    options:
        show_source: false


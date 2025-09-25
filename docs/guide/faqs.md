# Frequently Asked Questions

If you don't find your question in the FAQs or the user guide, feel free to open an issue in [GitHub](https://github.com/IBM/terratorch).


??? faq "What is TerraTorch and what does it do?"
    TerraTorch is a fine-tuning framework for Geospatial Foundation Models (GFMs) built on PyTorch Lightning and TorchGeo.
    It provides flexible tools for training models on Earth observation tasks.


??? faq "How do I install TerraTorch?"
    For stable releases, use `pip install terratorch`. For the latest development version, install with `pip install git+https://github.com/IBM/terratorch.git`.


??? faq "How do I use the CLI for training and inference?"
    You need to define a data module and model in a yaml config file.    
    Pass this config files in commands like `terratorch fit --config config.yaml` for training, `terratorch test --config config.yaml --ckpt_path model.ckpt` for testing, and `terratorch predict --config config.yaml --ckpt_path model.ckpt --predict_output_dir output_dir` for inference.
    See `terratorch --help` for details.


??? faq "How do I configure a YAML file for my experiment?"
    YAML files define the trainer, data, model, optimizer, and lr_scheduler sections. 
    The model section specifies the task type (like SemanticSegmentationTask) and model architecture parameters, while the data section configures the DataModule for loading your datasets.
    Please check this [tutorial](../tutorials/the_yaml_config.md) for details.


??? faq "What data formats does TerraTorch support?"
    TerraTorch supports both single-modal and multi-modal geospatial data through generic datasets and datamodules.
    These generic classes are using `rioxarray` to load data. You can check if your data is working with:
    ```python
    import rioxarray as rxr
    rxr.open_rasterio('<testfile>')
    ```
    You can implement a custom datamodule if the generic ones are not working.


??? faq "What foundation models are supported?"
    TerraTorch provides access to multiple pre-trained geospatial foundation models including Prithvi, TerraMind, SatMAE, ScaleMAE, Clay, and models from TorchGeo like DOFA and SSL4EO.
    It also supports standard computer vision models from the timm library.
    You can list or filter all available models with:
    ```python
    from terratorch import BACKBONE_REGISTRY    

    list(BACKBONE_REGISTRY)

    # Filter
    print([model_name for model_name in BACKBONE_REGISTRY if "prithvi" in model_name])
    ```


??? faq "What image size do I have to use with TerraTorch?"
    TerraTorch supports every image size, so you are mainly limited by your GPU memory.
    For training, we recommend using a chip size between 224 and 512.
    Ideally, your chips are a multiple of the patch size if you use ViT backbone (16 for most models) or 32 if you use a CNN.
    However, TerraTorch models automatically apply padding to make it work with any input size. 
    If the automatic padding does not work, pass `patch_size: 16` or similar to the model.

    During inference, you can use [tiled inference](inference.md) that can process full satellite tiles if needed.

 
??? faq "How does the model architecture work?"
    TerraTorch uses an encoder-decoder architecture where models are constructed by combining backbones, necks, decoders, and heads through model factories like [`EncoderDecoderFactory`](encoder_decoder_factory.md).
    Tasks coordinate training and inference while delegating model construction to these factories.


??? faq "How do I perform inference on new data?"
    Use the `terratorch predict ...` command with your trained model checkpoint and specify the input data directory. 
    For programmatic inference, you can use the `LightningInferenceModel` class to load models and perform inference on directories or individual files.
    See the [inference](inference.md) guide for details.


??? faq "How can I on-board new datasets?"
    You can either use one of the generic datamodules and bring your dataset in a supported format, or create a custom datamodule and dataset class.


??? faq "How do I add custom models or components?"
    You can extend TerraTorch by creating custom modules and registering them with the appropriate registries. 
    For example, register a custom backbone with `@TERRATORCH_BACKBONE_REGISTRY.register`.
    Place your code in a folder and adding the following to your yaml:
    ```
    custom_modules_path: <your/folder>
    ```

    See this [tutorial](../tutorials/adding_custom_modules.md) for details.


??? faq "Can I customize a model or task? E.g., I want to use another loss or another decoder."
    If you want to use a custom module which can be registered in TerraTorch (backbones, necks, and decoders), you can just add the `.register` function as a decorator and add your code to `custom_modules`.
    For edits in tasks like changing the loss or metric, you can copy the class from TerraTorch or inherit from one, change the relevant function, and it to `custom_modules` and select your new task in the yaml config.  


??? faq "I get an error during inference that is related to some size mismatches?"
    This can happen when the padding is not working correctly. 
    An easy fix is to add tiled inference to your config or pipeline as described in [Inference](inference.md). 


??? faq "My training just stops after one epoch without a stack trace. What is wrong?"
    Lightning automatically stops if only NaN losses are produced. Please check the logs (e.g. in Tensorboard).


??? faq "How can I fix NaN losses?"
    There can be multiple reasons for NaN losses. If you are training with mixed precision, try a run with full precision.
    Otherwise, your data or labels could include NaN values, or you are using a wrong `ignore_label`.


??? faq "Why is my dataset length 0?"
    Please check all paths and image/label grep patterns in your config. Likely, TerraTorch does not find the data.
    If the paths are correct, you can either debug the generic dataset class constructor or open an issue to get support from the developers. 

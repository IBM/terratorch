# Welcome to TerraTorch

<figure markdown="span">
    <img src="figs/logo.png#only-light" alt="TerraTorch"  width="400"/>
    <img src="figs/logo_grey.png#only-dark" alt="TerraTorch"  width="400"/>
    <h3>The Geospatial Foundation Models Toolkit</h3>
</figure>


<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Fine-tuning made easy__

    ---

    Run `pip install terratorch` and get started in minutes. 

    [:octicons-arrow-right-24: Quick start](guide%2Fquick_start.md)

-   :material-book-open-variant-outline:{ .lg .middle } __Overview__

    ---

    Get an overview of the functionality and architecture with in-depth explanations.

    [:octicons-arrow-right-24: User Guide](guide%2Farchitecture.md)

-   :fontawesome-regular-lightbulb:{ .lg .middle } __The big picture__

    ---

    We outline the ideas and reasons why we build TerraTorch in the paper. 

    [:octicons-arrow-right-24: arXiv](https://arxiv.org/abs/2503.20563)

-   :material-package-variant-closed-check:{ .lg .middle } __Open Source__

    ---

    TerraTorch is distributed under the terms of the Apache 2.0 license.

    [:octicons-arrow-right-24: License](about%2Flicense.md)

</div>


The purpose of this package is to build a flexible fine-tuning framework for Geospatial Foundation Models (GFMs) based on TorchGeo and Lightning which can be employed at different abstraction levels. 
It supports models from the [Prithvi](https://huggingface.co/ibm-nasa-geospatial), [TerraMind](https://huggingface.co/ibm-esa-geospatial), and [Granite](https://huggingface.co/ibm-granite/granite-geospatial-land-surface-temperature) series as well as models from [TorchGeo](https://torchgeo.readthedocs.io/en/latest/api/models.html) and [timm](https://huggingface.co/timm). 

This library provides:

- Ease-of-use and all the functionality from Lightning and TorchGeo.
- A modular model factory that combines any backbone with different decoders for full flexibility.
- Ready-to-use tasks for image segmentation, pixelwise regression, classification, and more.
- Multiple abstraction levels and inference pipelines to power enterprise applications.

A good starting place is familiarization with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), which this project is built on. 
[TorchGeo](https://torchgeo.readthedocs.io/en/stable/) is also an important complementary reference with many available models and specific datasets.
If you have any open questions, please check the [FAQs](guide/faqs.md) or open an issue in [GitHub](https://github.com/IBM/terratorch/issues).

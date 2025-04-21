# How to Add Custom Modules to TerraTorch

TerraTorch is designed to be extensible, allowing you to integrate your own custom components, such as models (backbones, decoders), tasks, datamodules, callbacks, or augmentation transforms, into the fine-tuning pipeline. This is primarily achieved by making your custom Python code discoverable by TerraTorch and then referencing your custom classes within the YAML configuration file.

This tutorial outlines the steps required to add and use a custom module. We'll use a simple custom model component as an example.

## Prerequisites

*   A working installation of TerraTorch.
*   Your custom Python code (e.g., a new model architecture implemented as a `torch.nn.Module`).

## Step 1: Create Your Custom Module File(s)

First, organize your custom code into Python files (`.py`). It's recommended to place these files within a dedicated directory. For example, let's create a directory named `custom_modules` in your project's working directory and place our custom model definition inside a file named `my_custom_model.py`:

```text
my_project_root/
├── custom_modules/
│   ├── __init__.py       <-- Optional, but good practice
│   └── my_custom_model.py
├── my_config.yaml
└── ... (other project files)
```

Inside `custom_modules/my_custom_model.py`, define your custom class. For instance, a simple custom PyTorch model component:

```python
# custom_modules/my_custom_model.py
import torch
import torch.nn as nn

class MySimpleCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, num_classes, kernel_size=1)
        print(f"Initialized MySimpleCNN with in_channels={in_channels}, num_classes={num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

# You could also define custom tasks, datamodules, etc. here
# from terratorch.tasks import SemanticSegmentationTask
# class MyCustomTask(SemanticSegmentationTask):
#     # ... override methods ...
#     pass
```

Make sure your custom classes inherit from appropriate base classes if needed (e.g., `torch.nn.Module` for models, `lightning.pytorch.LightningModule` or `terratorch.tasks.BaseTask` for tasks, `lightning.pytorch.LightningDataModule` for datamodules).

## Step 2: Inform TerraTorch About Your Custom Module Directory

TerraTorch needs to know where to find your custom code. You can specify the path to your custom modules directory using the `custom_modules_path` argument either in your YAML configuration file or directly via the command line.

**Option A: In YAML Configuration (`my_config.yaml`)**

Add the `custom_modules_path` key at the top level of your configuration:

```yaml
# my_config.yaml
custom_modules_path: ./custom_modules  # Path relative to where you run terratorch

# ... rest of your configuration ...

model:
  # ... other model config ...
data:
  # ... data config ...
trainer:
  # ... trainer config ...

```

**Option B: Via Command Line**

```bash
terratorch fit --config my_config.yaml --custom_modules_path ./custom_modules
```

When provided, TerraTorch will add the specified directory (`./custom_modules` in this case) to Python's `sys.path`, making the modules within it importable.

## Step 3: Use Your Custom Component in the Configuration

Now that TerraTorch can find your custom code, you can reference your custom classes in the YAML configuration using their full Python class path. The class path is constructed as `<directory_name>.<filename>.<ClassName>`.

For example, to use `MySimpleCNN` defined in `custom_modules/my_custom_model.py` as part of your model configuration (e.g., as a decoder):

```yaml
# my_config.yaml
custom_modules_path: ./custom_modules

# Assuming you are using a task that expects a model with certain components
model:
  # Example: Using the custom model as a segmentation decoder
  decoder:
    class_path: custom_modules.my_custom_model.MySimpleCNN
    init_args:
      in_channels: 128 # Example input channels from backbone
      num_classes: 5   # Example number of output classes
  backbone:
    # ... configure backbone ...
  # ... other model args ...

task:
  class_path: terratorch.tasks.SemanticSegmentationTask # Or your custom task
  init_args:
    # ... task args ...

data:
  # ... data config ...

trainer:
  # ... trainer config ...
```

Similarly, you could reference a custom task like `custom_modules.my_custom_model.MyCustomTask` under the `task.class_path` key.

## Summary

By following these steps:
1.  Creating your custom Python code in a dedicated directory.
2.  Specifying the path to this directory using `custom_modules_path`.
3.  Referencing your custom classes via their full Python path in the YAML configuration.

You can seamlessly integrate your own modules into the TerraTorch framework. 
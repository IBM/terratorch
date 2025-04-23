import os
import sys
import torch
import terratorch
import warnings
import os
import sys

#os.environ["TENSORBOARD_PROXY_URL"]= os.environ["NB_PREFIX"]+"/proxy/6006/"
warnings.filterwarnings('ignore')
modules_path = "custom_modules"
sys.path.append(modules_path)

# Model
model = terratorch.tasks.PixelwiseRegressionTask(
    model_factory="EncoderDecoderFactory",
    model_args={
        "backbone": "AlexNetEncoder", 
        "backbone_num_channels": 6,
        "decoder": "IdentityDecoder"
    },

    loss="rmse",
    optimizer="AdamW",
    lr=1e-3,
    ignore_index=-1,
    freeze_backbone=True,
    freeze_decoder=False,
    plot_on_val=True,
)

data = torch.randn(1,6,224,224)
assert model(data).output.shape == (1,224,224)

# Model
model = terratorch.tasks.PixelwiseRegressionTask(
    model_factory="GenericModelFactory",
    model_args={
        "backbone": "AlexNet", 
        "backbone_num_classes": 10,
        "backbone_num_channels": 6,
    },

    loss="rmse",
    optimizer="AdamW",
    lr=1e-3,
    ignore_index=-1,
    freeze_backbone=True,
    freeze_decoder=False,
    plot_on_val=True,
)

data = torch.randn(1,6,224,224)
assert model(data).output.shape == (1,10)

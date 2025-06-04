import pytest
import torch
import lightning
from terratorch.models import SMPModelFactory
from torch import nn
from terratorch.tasks.optimizer_factory import optimizer_factory


NUM_CHANNELS = 3
MODEL_TYPE = "Unet"
NUM_CLASSES = 2
PRETRAINED_BANDS = ["RED", "GREEN", "BLUE"]
PRETRAINED=False
BACKBONE="resnet50"
TASK_TYPE="segmentation"
IMAGE_SIZE=224
EXPECTED_SEGMENTATION_OUTPUT_SHAPE = (1, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)
DEFAULT_INPUTS = {
    "optimizer": "AdamW",
    "optimizer_hparams": {
        "weight_decay": 0.01,
        },
    "monitor": "val_loss",
    "lr": 0.001
    }

WARMUP_DECAY_INPUTS = [
    (
        {"schedulers":{
                "LambdaLR": {"lr_lambda": "linear_warmup"},
                "CosineAnnealingLR": {"T_max": 10}
                },
        "milestones": [5],
        }, DEFAULT_INPUTS, 10
    ),
    (
        {"schedulers":{
                "LambdaLR": {"lr_lambda": "linear_warmup"},
                "ExponentialLR": {"gamma":0.95}
                },
        "milestones": [2],
        },DEFAULT_INPUTS, 5
    )
]
WARMUP_DECAY_TEST_IDS = [str(i) for i in range(0, len(WARMUP_DECAY_INPUTS))]



@pytest.fixture(scope="session")
def model() -> nn.Module:
    model_factory = SMPModelFactory()
    model = model_factory.build_model(
        task=TASK_TYPE,
        backbone=BACKBONE,
        model=MODEL_TYPE,
        in_channels=NUM_CHANNELS,
        bands=PRETRAINED_BANDS,
        pretrained=PRETRAINED,
        num_classes=NUM_CLASSES,
    )
    return model

@pytest.mark.parametrize(
    "scheduler_hparams, defaults, epochs", 
    WARMUP_DECAY_INPUTS,
    ids=WARMUP_DECAY_TEST_IDS,
)
def test_warmup_sequential_scheduling(
    scheduler_hparams:dict,
    defaults: str,
    epochs: int,
    model: nn.Module,
    ):
    """ Test sequentiallr scheduler with warm up stage then decay stage
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    defaults.update({"scheduler": "SequentialLR"})
    defaults.update({"scheduler_hparams":scheduler_hparams})
    defaults.update({"params_to_be_optimized": model.parameters()})
    optimizer_factory_output = optimizer_factory(**defaults)

    #check type
    assert isinstance(optimizer_factory_output["optimizer"], torch.optim.Optimizer)
    assert isinstance(optimizer_factory_output["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.SequentialLR )

    scheduler = optimizer_factory_output["lr_scheduler"]["scheduler"]
    optimizer = optimizer_factory_output["optimizer"]

    def training_step(optimizer, scheduler, model, loss_fn):
        optimizer.zero_grad()
        model_input = torch.rand((1, len(PRETRAINED_BANDS), IMAGE_SIZE, IMAGE_SIZE))
        target = torch.ones(EXPECTED_SEGMENTATION_OUTPUT_SHAPE)
        model_output = model(model_input)
        loss = loss_fn(model_output.output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    #check that learning rate is changing according to schedule
    for i in range(epochs):
        pre_step_lr = scheduler.get_last_lr()
        training_step(optimizer, scheduler, model, loss_fn)
        post_step_lr = scheduler.get_last_lr()
        if i < scheduler_hparams["milestones"][0]-1:
            assert pre_step_lr < post_step_lr, "LR is not increasing during warmup stage"
        elif i == scheduler_hparams["milestones"][0]-1:
            assert pre_step_lr == post_step_lr, "LR does not match expected LR at this point"
        else:
            assert pre_step_lr > post_step_lr, "LR is not decreasing during decay stage"

#add tests for repeated (cyclic) warmup and decay

#add parametrized tests with other oprimizers

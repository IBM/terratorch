import pytest
import torch
import lightning
from terratorch.models import SMPModelFactory
from torch import nn
from terratorch.models.encoder_decoder_factory import optimizer_factory


NUM_CHANNELS = 3
MODEL_TYPE = "Unet"
NUM_CLASSES = 2
PRETRAINED_BANDS = 3
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
    ({
        "schedulers":{
                "LambdaLR": {"lr_lambda": "linear_warmup"},
                "CosineAnnealingLR": {"T_max": 50}
                },
        "milestones": [2],
    }, DEFAULT_INPUTS, 5)
    ({
        "schedulers":{
                "LambdaLR": {"lr_lambda": "linear_warmup"},
                "optim.lr_scheduler.ExponentialLR": {"gamma":0.95}
                },
        "milestones": [2],
    },DEFAULT_INPUTS, 5)
]
WARMUP_DECAY_TEST_IDS = [str(i) for i in range(0, len(WARMUP_DECAY_INPUTS))]



@pytest.fixture(scope="session")
def get_model() -> nn.Module:
    model = SMPModelFactory.build_model(
        TASK_TYPE,
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
    ):
    """ Test sequentiallr scheduler with warm up stage then decay stage
    """

    model = get_model()
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    defaults.update({"scheduler": "SequentialLR"})
    defaults.update({"scheduler_hparams":scheduler_hparams})
    defaults.update({"params_to_be_optimized": model.parameters()})
    optimizer_factory_output = optimizer_factory(**defaults)

    #should have the correct type
    assert isinstance(optimizer_factory_output, 
        lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig)

    scheduler = optimizer_factory_output["lr_scheduler"]["scheduler"]
    print(f"scheduler: {scheduler} \ntype:{type(scheduler)}")

    def training_step(scheduler, model, loss_fn):
        #optimizer.zero_grad()
        model_input = torch.rand((1, PRETRAINED_BANDS, IMAGE_SIZE, IMAGE_SIZE))
        target = torch.ones(EXPECTED_SEGMENTATION_OUTPUT_SHAPE)
        model_output = model(model_input)
        loss = loss_fn(model_output, target)
        loss.backward()
        scheduler.step()
    
    #check that learning rate is changing according to schedule
    for i in range(epochs):
        pre_step_lr = scheduler.get_lr()
        training_step(scheduler, model, loss_fn)
        post_step_lr = scheduler.get_lr()
        print(f"i: {i} pre_step_lr: {pre_step_lr} post_step_lr: {post_step_lr}")
        if i < scheduler_hparams["milestones"][0]:
            assert pre_step_lr < post_step_lr, "LR is not increasing during warmup stage"
        else:
            assert pre_step_lr > post_step_lr, "LR is not decreasing during decay stage"


#add tests for repeated warmup and decay

#add parametrized tests with other oprimizers

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.finetuning import BaseFinetuning


class ProgressiveResizing(Callback):
    def __init__(self):
        self.resize_schedule = {
            0: {"batch_size": 4, "num_workers": 4, "size": 64},
            10: {"batch_size": 2, "num_workers": 2, "size": 128},
            20: {"batch_size": 1, "num_workers": 1, "size": 256},
        }

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.resize_schedule:
            params = self.resize_schedule[trainer.current_epoch]

            trainer.datamodule.size = params["size"]
            trainer.datamodule.batch_size = params["batch_size"]
            trainer.datamodule.num_workers = params["num_workers"]

            trainer.datamodule.setup(stage="fit")

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.resize_schedule:
            params = self.resize_schedule[trainer.current_epoch]

            trainer.datamodule.size = params["size"]
            trainer.datamodule.batch_size = params["batch_size"]
            trainer.datamodule.num_workers = params["num_workers"]

            trainer.datamodule.setup(stage="validate")


class LayerwiseFinetuning(BaseFinetuning):
    def __init__(self, phase, train_bn=True):
        """Initializes with phase & batch-norm information.

        Args:
            phase (List): Phases of fine-tuning the backbone network.
            train_bn (bool, optional): Trains just the batch-norm layers even
            when all the other layers of the network are freezed. Defaults to True.
        """
        super().__init__()
        self.phase = phase
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module):
        """Freezes the encoder before starting the training."""
        self.freeze(
            modules=[
                pl_module.model.encoder.patch_embedding,
                pl_module.model.encoder.transformer,
            ],
            train_bn=self.train_bn,
        )

    def finetune_function(self, pl_module, epoch, optimizer):
        if epoch == self.phase:
            """Unfreezes the encoder for training."""
            print(f"In Phase {self.phase}: Full throttle")
            self.unfreeze_and_add_param_group(
                modules=[
                    pl_module.model.encoder.patch_embedding,
                    pl_module.model.encoder.transformer,
                ],
                optimizer=optimizer,
                train_bn=self.train_bn,
            )
            params = list(pl_module.parameters())
            active = list(filter(lambda p: p.requires_grad, params))
            print(f"active: {len(active)}, all: {len(params)}")

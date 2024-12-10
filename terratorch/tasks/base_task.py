
from lightning.pytorch.callbacks import Callback
from torchgeo.trainers import BaseTask


BATCH_IDX_FOR_VALIDATION_PLOTTING = 10

class TerraTorchTask(BaseTask):

    """
    Parent used to share common methods among all the 
    tasks implemented in terratorch 
    """

    def __init__(self, task:str=None):

        self.task = task 

        super().__init__()

    # overwrite early stopping
    def configure_callbacks(self) -> list[Callback]:
        return []

    def configure_models(self) -> None:
        if not hasattr(self, "model_factory"):
            if self.hparams["freeze_backbone"] or self.hparams["freeze_decoder"]:
                logger.warning("freeze_backbone and freeze_decoder are ignored if a custom model is provided.")
            # Skipping model factory because custom model is provided
            return

        self.model: Model = self.model_factory.build_model(
            self.task, aux_decoders=self.aux_heads, **self.hparams["model_args"]
        )

        if self.hparams["freeze_backbone"]:
            if self.hparams.get("peft_config", None) is not None:
                msg = "PEFT should be run with freeze_backbone = False"
                raise ValueError(msg)
            self.model.freeze_encoder()

        if self.hparams["freeze_decoder"]:
            self.model.freeze_decoder()

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        optimizer = self.hparams["optimizer"]
        if optimizer is None:
            optimizer = "Adam"
        return optimizer_factory(
            optimizer,
            self.hparams["lr"],
            self.parameters(),
            self.hparams["optimizer_hparams"],
            self.hparams["scheduler"],
            self.monitor,
            self.hparams["scheduler_hparams"],
        )

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.train_metrics.reset()
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()
        return super().on_validation_epoch_end()


    def _do_plot_samples(self, batch_index):
        if not self.plot_on_val:  # dont plot if self.plot_on_val is 0
            return False

        return (
            batch_index < BATCH_IDX_FOR_VALIDATION_PLOTTING
            and hasattr(self.trainer, "datamodule")
            and self.logger
            and not self.current_epoch % self.plot_on_val  # will be True every self.plot_on_val epochs
            and hasattr(self.logger, "experiment")
            and (hasattr(self.logger.experiment, "add_figure") or hasattr(self.logger.experiment, "log_figure"))
        )



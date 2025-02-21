"""This module contains pre-training tasks."""

import torch
import matplotlib.pyplot as plt
from torch import nn
from lightning.pytorch.callbacks import Callback
from torchgeo.datasets.utils import unbind_samples
from torchgeo.trainers import BaseTask
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from terratorch.models.model import Model
from terratorch.registry import MODEL_FACTORY_REGISTRY
from terratorch.tasks.loss_handler import LossHandler
from terratorch.tasks.tiled_inference import TiledInferenceParameters, tiled_inference
from terratorch.tasks.regression_tasks import IgnoreIndexLossWrapper, IgnoreIndexMetricWrapper, RootLossWrapper

BATCH_IDX_FOR_VALIDATION_PLOTTING = 10


class ReconstructionTask(BaseTask):
    """
    Task with image reconstruction.
    """

    def __init__(
            self,
            model_factory: str,
            model_args: dict,
            loss: str = "mse",
            ignore_index: int | float = torch.nan,
            masked_metric: bool = True,
            lr: float = 0.001,
            optimizer: str | None = None,
            optimizer_hparams: dict | None = None,
            scheduler: str | None = None,
            scheduler_hparams: dict | None = None,
            freeze_encoder: bool = False,  # noqa: FBT001, FBT002
            freeze_decoder: bool = False,  # noqa: FBT001, FBT002
            plot_on_val: bool | int = 10,
            tiled_inference_parameters: TiledInferenceParameters | None = None,
            modalities: list[str] | None = None
    ) -> None:
        """Constructor

        Args:
            model_factory (str): Name of ModelFactory class to be used to instantiate the model.
            model_args (Dict): Arguments passed to the model factory.
            loss (str, optional): Loss to be used. Currently, supports 'mse', 'rmse', 'mae' or 'huber' loss.
                Defaults to "mse".
            aux_loss (dict[str, float] | None, optional): Auxiliary loss weights.
                Should be a dictionary where the key is the name given to the loss
                and the value is the weight to be applied to that loss.
                The name of the loss should match the key in the dictionary output by the model's forward
                method containing that output. Defaults to None.
            ignore_index (int | float, optional): Values to ignore in the metric computation. Defaults to torch.nan.
            masked_metric (bool, optional): Whether to mask the pred for the metric computation, so that metrics are
                only computed on masked pixels. Expects masks with 0 == not masked and 1 = masked. Defaults to True.
            lr (float, optional): Learning rate to be used. Defaults to 0.001.
            optimizer (str | None, optional): Name of optimizer class from torch.optim to be used.
                If None, will use Adam. Defaults to None. Overriden by config / cli specification through LightningCLI.
            optimizer_hparams (dict | None): Parameters to be passed for instantiation of the optimizer.
                Overriden by config / cli specification through LightningCLI.
            scheduler (str, optional): Name of Torch scheduler class from torch.optim.lr_scheduler
                to be used (e.g. ReduceLROnPlateau). Defaults to None.
                Overriden by config / cli specification through LightningCLI.
            scheduler_hparams (dict | None): Parameters to be passed for instantiation of the scheduler.
                Overriden by config / cli specification through LightningCLI.
            freeze_encoder (bool, optional): Whether to freeze the encoder. Defaults to False.
            freeze_decoder (bool, optional): Whether to freeze the decoder and segmentation head. Defaults to False.
            plot_on_val (bool | int, optional): Whether to plot visualizations on validation.
                If true, log every epoch. Defaults to 10. If int, will plot every plot_on_val epochs.
            tiled_inference_parameters (TiledInferenceParameters | None, optional): Inference parameters
                used to determine if inference is done on the whole image or through tiling.
        """
        self.tiled_inference_parameters = tiled_inference_parameters
        self.model_factory = MODEL_FACTORY_REGISTRY.build(model_factory)
        self.modalities = modalities
        self.ignore_index = ignore_index
        self.masked_metric = masked_metric
        super().__init__()
        self.train_loss_handler = LossHandler('train/')
        self.val_loss_handler = LossHandler('val/')
        self.test_loss_handler = LossHandler('test/')
        self.monitor = f"val/loss"
        self.plot_on_val = int(plot_on_val)

    # overwrite early stopping
    def configure_callbacks(self) -> list[Callback]:
        return []

    def configure_models(self) -> None:
        self.model: Model = self.model_factory.build_model(**self.hparams["model_args"])

        if self.hparams["freeze_encoder"]:
            if self.hparams.get("peft_config", None) is not None:
                msg = "PEFT should be run with freeze_encoder = False and freeze_decoder = False."
                raise ValueError(msg)
            self.model.freeze_encoder()
        if self.hparams["freeze_decoder"]:
            self.model.freeze_decoder()

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"].lower()
        if loss == "mse":
            self.criterion: nn.Module = IgnoreIndexLossWrapper(
                nn.MSELoss(reduction="none"), self.ignore_index
            )
        elif loss == "mae":
            self.criterion = IgnoreIndexLossWrapper(nn.L1Loss(reduction="none"), self.ignore_index)
        elif loss == "rmse":
            # IMPORTANT! Root is done only after ignore index! Otherwise, the mean taken is incorrect
            self.criterion = RootLossWrapper(
                IgnoreIndexLossWrapper(nn.MSELoss(reduction="none"), self.ignore_index), reduction=None
            )
        elif loss == "huber":
            self.criterion = IgnoreIndexLossWrapper(nn.HuberLoss(reduction="none"), self.ignore_index)
        else:
            exception_message = f"Loss type '{loss}' is not valid. Currently, supports 'mse', 'rmse' or 'mae' loss."
            raise ValueError(exception_message)

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""

        def instantiate_metrics():
            metrics = {
                "RMSE": MeanSquaredError(squared=False),
                "MSE": MeanSquaredError(squared=True),
                "MAE": MeanAbsoluteError(),
            }

            return {name: IgnoreIndexMetricWrapper(metric, ignore_index=self.hparams["ignore_index"])
                    for name, metric in metrics.items()}

        if self.modalities:
            self.train_metrics = nn.ModuleDict(
                {modality: MetricCollection(instantiate_metrics(), prefix=f"train/{modality}_")
                 for modality in self.modalities}
            )
            self.val_metrics = nn.ModuleDict(
                {modality: MetricCollection(instantiate_metrics(), prefix=f"val/{modality}_")
                 for modality in self.modalities}
            )
            self.test_metrics = nn.ModuleDict(
                {modality: MetricCollection(instantiate_metrics(), prefix=f"test/{modality}_")
                 for modality in self.modalities}
            )
        else:
            self.train_metrics = MetricCollection(instantiate_metrics(), prefix="train/")
            self.val_metrics = MetricCollection(instantiate_metrics(), prefix="val/")
            self.test_metrics = MetricCollection(instantiate_metrics(), prefix="test/")

    def mask_input(self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, image_size: int | tuple[int]
                   ) -> (torch.Tensor, torch.Tensor):
        if isinstance(x, dict):
            # Multimodal data
            for key in x.keys():
                x[key], mask[key] = self.mask_input(x[key], pred[key], mask[key], image_size)
        else:
            if mask.shape != pred.shape:  # Assuming mask with dims [B, T] and pred with [B, T, E]
                mask = self.model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1]), image_size=image_size)
            if mask.shape != x.shape and len(x.shape):
                if len(mask.shape) == 5 and len(x.shape) == 4:
                    # Add time dim for non-temporal input and temporal models
                    x = x.unsqueeze(2)
                else:
                    raise ValueError(f'Found incompatible shape for mask {mask.shape} and input {x.shape}')
            # Mask input
            x[mask == True] = self.ignore_index

        return x, mask


    def training_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Compute the train loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        if isinstance(x, dict):
            batch_size = list(x.values())[0].shape[0]
            image_size = list(x.values())[0].shape[:-2]
        elif isinstance(x, torch.Tensor):
            batch_size = x.shape[0]
            image_size = x.shape[-2:]
        else:
            raise ValueError("Could not infer batch size from input data.")

        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}

        loss, pred, mask = self(x, **rest)

        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=batch_size)

        if self.masked_metric and mask is not None:
            x, mask = self.mask_input(x, pred, mask, image_size)

        if isinstance(x, dict):
            # Multimodal data
            for key in pred.keys():
                pred[key] = self.model.unpatchify(pred[key], image_size=image_size)

            for modality in x.keys():
                if modality in self.val_metrics:
                    self.train_metrics[modality].update(pred[modality], x[modality])
        else:
            # Single modality
            pred = self.model.unpatchify(pred, image_size=image_size)
            self.train_metrics.update(pred, x)

        return loss["loss"]

    def on_train_epoch_end(self) -> None:
        if isinstance(self.test_metrics, MetricCollection):
            self.log_dict(self.train_metrics.compute(), sync_dist=True)
            self.train_metrics.reset()
        else:
            for metric in self.train_metrics.values():
                self.log_dict(metric.compute(), sync_dist=True)
                metric.reset()
        return super().on_train_epoch_end()

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

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        if isinstance(x, dict):
            batch_size = list(x.values())[0].shape[0]
            image_size = list(x.values())[0].shape[:-2]
        elif isinstance(x, torch.Tensor):
            batch_size = x.shape[0]
            image_size = x.shape[-2:]
        else:
            raise ValueError("Could not infer batch size from input data.")

        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}

        loss, pred, mask = self(x, **rest)

        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=batch_size)

        if self.masked_metric and mask is not None:
            x, mask = self.mask_input(x, pred, mask, image_size)

        if isinstance(x, dict):
            if not isinstance(self.val_metrics, nn.ModuleDict):
                raise ValueError(f'Multimodal data provided but no image modalities. '
                                 f'Please provide modalities (list of str) to the ReconstructionTask.')
            # Multimodal data
            for key in pred.keys():
                pred[key] = self.model.unpatchify(pred[key], image_size=image_size)

            for modality in x.keys():
                if modality in self.val_metrics:
                    self.val_metrics[modality].update(pred[modality], x[modality])
        else:
            # Single modality
            pred = self.model.unpatchify(pred, image_size=image_size)
            self.val_metrics.update(pred, x)

        if self._do_plot_samples(batch_idx):
            try:
                datamodule = self.trainer.datamodule
                if isinstance(batch["image"], dict):
                    # Multimodal input
                    rgb_modality = getattr(datamodule, 'rgb_modality', None) or list(batch["image"].keys())[0]
                    batch["image"] = batch["image"][rgb_modality]
                    batch["prediction"] = pred[rgb_modality].cpu()
                    batch["mask"] = mask[rgb_modality].cpu()
                else:
                    batch["prediction"] = pred.cpu()
                    batch["mask"] = mask.cpu()
                batch['image'] = batch['image'].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.val_dataset.plot(sample)
                if fig:
                    summary_writer = self.logger.experiment
                    if hasattr(summary_writer, "add_figure"):
                        summary_writer.add_figure(f"image/{batch_idx}", fig, global_step=self.global_step)
                    elif hasattr(summary_writer, "log_figure"):
                        summary_writer.log_figure(
                            self.logger.run_id, fig, f"epoch_{self.current_epoch}_{batch_idx}.png"
                        )
            except ValueError:
                pass
            finally:
                plt.close()

    def on_validation_epoch_end(self) -> None:
        if isinstance(self.test_metrics, MetricCollection):
            self.log_dict(self.val_metrics.compute(), sync_dist=True)
            self.val_metrics.reset()
        else:
            for metric in self.val_metrics.values():
                self.log_dict(metric.compute(), sync_dist=True)
                metric.reset()
        return super().on_validation_epoch_end()

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        if isinstance(x, dict):
            batch_size = list(x.values())[0].shape[0]
            image_size = list(x.values())[0].shape[:-2]
        elif isinstance(x, torch.Tensor):
            batch_size = x.shape[0]
            image_size = x.shape[-2:]
        else:
            raise ValueError("Could not infer batch size from input data.")

        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}

        loss, pred, mask = self(x, **rest)

        self.test_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=batch_size)

        if self.masked_metric and mask is not None:
            x, mask = self.mask_input(x, pred, mask, image_size)

        if isinstance(x, dict):
            # Multimodal data
            for key in pred.keys():
                pred[key] = self.model.unpatchify(pred[key], image_size=image_size)

            for modality in x.keys():
                if modality in self.val_metrics:
                    self.test_metrics[modality].update(pred[modality], x[modality])
        else:
            # Single modality
            pred = self.model.unpatchify(pred, image_size=image_size)
            self.test_metrics.update(pred, x)

    def on_test_epoch_end(self) -> None:
        if isinstance(self.test_metrics, MetricCollection):
            self.log_dict(self.test_metrics.compute(), sync_dist=True)
            self.test_metrics.reset()
        else:
            # multiple metric dicts
            for metric in self.test_metrics.values():
                self.log_dict(metric.compute(), sync_dist=True)
                metric.reset()
        return super().on_test_epoch_end()

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        image_size = x.shape[-2:]
        file_names = batch["filename"]
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}

        def model_forward(x):
            pred = self(x)[1]
            pred = self.model.unpatchify(pred, image_size=image_size)
            return pred

        # Tiled inference for autoencoder ? Is it making sense ?
        if self.tiled_inference_parameters:
            # TODO: tiled inference does not work with additional input data (**rest)
            pred: torch.Tensor = tiled_inference(model_forward, x, 1, self.tiled_inference_parameters)
        else:
            pred: torch.Tensor = model_forward(x)
        # TODO: Return mask and file_names?
        return pred, file_names

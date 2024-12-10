# Copyright contributors to the Terratorch project

import importlib.util
import itertools
import json
import logging
import os
import shutil
import sys
import tempfile
import warnings
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import albumentations
import cv2  # noqa: F401
import numpy as np
import rasterio
import torch

# Allows classes to be referenced using only the class name
import torchgeo.datamodules
import yaml
from albumentations.pytorch import ToTensorV2  # noqa: F401
from jsonargparse import set_dumper
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter, ModelCheckpoint, RichProgressBar
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI, SaveConfigCallback
from torchgeo.trainers import BaseTask

import terratorch.datamodules
import terratorch.tasks  # noqa: F401
from terratorch.datamodules import (  # noqa: F401
    GenericNonGeoClassificationDataModule,
    GenericNonGeoPixelwiseRegressionDataModule,
    GenericNonGeoSegmentationDataModule,
)
from terratorch.datasets.transforms import (
    FlattenTemporalIntoChannels,  # noqa: F401
    Rearrange,  # noqa: F401
    SelectBands,  # noqa: F401
    UnflattenTemporalFromChannels,  # noqa: F401
)
from terratorch.datasets.utils import HLSBands

# GenericNonGeoRegressionDataModule,
from terratorch.models import PrithviModelFactory  # noqa: F401
from terratorch.models.model import AuxiliaryHead  # noqa: F401
from terratorch.tasks import (
    ClassificationTask,  # noqa: F401
    PixelwiseRegressionTask,  # noqa: F401
    SemanticSegmentationTask,  # noqa: F401
)

logger = logging.getLogger("terratorch")

def flatten(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))


def open_tiff(fname):
    with rasterio.open(fname, "r") as src:
        data = src.read()
        metadata = src.meta
    return data, metadata


def is_one_band(img):
    return len(img.shape) == 2  # noqa: PLR2004


def write_tiff(img_wrt, filename, metadata):

    with rasterio.open(filename, "w", **metadata) as dest:
        if is_one_band(img_wrt):
            img_wrt = img_wrt[None]

        for i in range(img_wrt.shape[0]):
            dest.write(img_wrt[i, :, :], i + 1)
    return filename


def save_prediction(prediction, input_file_name, out_dir, dtype:str="int16"):
    mask, metadata = open_tiff(input_file_name)
    mask = np.where(mask == metadata["nodata"], 1, 0)
    mask = np.max(mask, axis=0)
    result = np.where(mask == 1, -1, prediction.detach().cpu())

    ##### Save file to disk
    metadata["count"] = 1
    metadata["dtype"] = dtype
    metadata["compress"] = "lzw"
    metadata["nodata"] = -1
    file_name = os.path.basename(input_file_name)
    file_name_no_ext = os.path.splitext(file_name)[0]
    out_file_name = file_name_no_ext + "_pred.tif"
    logger.info(f"Saving output to {out_file_name} ...")
    write_tiff(result, os.path.join(out_dir, out_file_name), metadata)


def import_custom_modules(custom_modules_path: str | Path | None = None) -> None:

    if custom_modules_path:

        custom_modules_path = Path(custom_modules_path)

        if custom_modules_path.is_dir():

            # Add 'custom_modules' folder to sys.path
            workdir = custom_modules_path.parents[0]
            module_dir = custom_modules_path.name

            sys.path.append(workdir)

            try:
                importlib.import_module(module_dir)
                logger.info(f"Found {custom_modules_path}")
            except ImportError:
                raise ImportError(f"It was not possible to import modules from {custom_modules_path}.")
        else:
            raise ValueError(f"Modules path {custom_modules_path} isn't a directory. Check if you have defined it properly.")
    else:
        logger.debug("No custom module is being used.")

class CustomWriter(BasePredictionWriter):
    """Callback class to write geospatial data to file."""

    def __init__(self, output_dir: str | None = None, write_interval: str = "epoch"):

        super().__init__(write_interval)

        self.output_dir = output_dir

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):  # noqa: ARG002
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank

        # by default take self.output_dir. If None, look for one in trainer
        if self.output_dir is None:
            try:
                output_dir = trainer.predict_output_dir
            except AttributeError as err:
                msg = "Output directory must be passed to CustomWriter constructor or the `predict_output_dir`\
                attribute must be present in the trainer."
                raise Exception(msg) from err
        else:
            output_dir = self.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        pred_batch, filename_batch = prediction

        for prediction, file_name in zip(torch.unbind(pred_batch, dim=0), filename_batch, strict=False):
            save_prediction(prediction, file_name, output_dir, dtype=trainer.out_dtype)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):  # noqa: ARG002
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank

        # by default take self.output_dir. If None, look for one in trainer
        if self.output_dir is None:
            try:
                output_dir = trainer.predict_output_dir
            except AttributeError as err:
                msg = "Output directory must be passed to CustomWriter constructor or the `predict_output_dir`\
                attribute must be present in the trainer."
                raise Exception(msg) from err
        else:
            output_dir = self.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for pred_batch, filename_batch in predictions:
            for prediction, file_name in zip(torch.unbind(pred_batch, dim=0), filename_batch, strict=False):
                save_prediction(prediction, file_name, output_dir, dtype=trainer.out_dtype)


def clean_config_for_deployment_and_dump(config: dict[str, Any]):
    deploy_config = deepcopy(config)
    ## General
    # drop ckpt_path
    deploy_config.pop("ckpt_path", None)
    # drop checkpoints
    deploy_config.pop("ModelCheckpoint", None)
    deploy_config.pop("StateDictModelCheckpoint", None)
    # drop optimizer and lr sheduler
    deploy_config.pop("optimizer", None)
    deploy_config.pop("lr_scheduler", None)
    ## Trainer
    # remove logging
    deploy_config["trainer"]["logger"] = False
    # remove callbacks
    deploy_config["trainer"].pop("callbacks", None)
    # remove default_root_dir
    deploy_config["trainer"].pop("default_root_dir", None)
    # set mixed precision by default for inference
    deploy_config["trainer"]["precision"] = "16-mixed"
    ## Model
    # set pretrained to false
    if "model_args" in deploy_config["model"]["init_args"]:
        deploy_config["model"]["init_args"]["model_args"]["pretrained"] = False

    return yaml.safe_dump(deploy_config)


class StudioDeploySaveConfigCallback(SaveConfigCallback):
    """Subclass of the SaveConfigCallback to save configs.
    Besides saving the usual config, will also save a modified config ready to be uploaded.
    Handles things like disabling the pretrained flag, removing loggers, ...
    """

    def __init__(
        self,
        parser,
        config,
        config_filename: str = "config.yaml",
        overwrite: bool = False,
        multifile: bool = False,
        save_to_log_dir: bool = True,
    ):
        super().__init__(parser, config, config_filename, overwrite, multifile, save_to_log_dir)
        set_dumper("deploy_config", clean_config_for_deployment_and_dump)

        # Preparing information to save config file to log dir
        config_dict = config.as_dict()
        self.config_path_original = str(config_dict["config"][0])
        _, self.config_file_original = os.path.split(self.config_path_original)

        self.deploy_config_file = config_dict["deploy_config_file"]

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        if self.save_to_log_dir:
            log_dir = trainer.log_dir or trainer.default_root_dir  # this broadcasts the directory
            if log_dir is None:
                msg = "log_dir was none. Please set a logger or set default_root_dir for the trainer"
                raise Exception(msg)
            config_path = os.path.join(log_dir, self.config_filename)
            fs = get_filesystem(log_dir)

            if not self.overwrite:
                # check if the file exists on rank 0
                file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
                # broadcast whether to fail to all ranks
                file_exists = trainer.strategy.broadcast(file_exists)
                if file_exists:
                    msg = (
                        f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                        " results of a previous run. You can delete the previous config file,"
                        " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                        ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                    )
                    raise RuntimeError(msg)

            if trainer.is_global_zero:
                # save only on rank zero to avoid race conditions.
                # the `log_dir` needs to be created as we rely on the logger to do it usually
                # but it hasn't logged anything at this point
                fs.makedirs(log_dir, exist_ok=True)

                if self.deploy_config_file:
                    self.parser.save(
                        self.config, config_path, skip_none=True, overwrite=self.overwrite, multifile=self.multifile
                    )

        if trainer.is_global_zero:
            if self.deploy_config_file:
                # also save the config that will be deployed
                config_name, config_ext = os.path.splitext(self.config_filename)
                config_name += "_deploy"
                config_name += config_ext
                config_path = os.path.join(log_dir, config_name)
                self.parser.save(
                    self.config,
                    config_path,
                    format="deploy_config",
                    skip_none=True,
                    overwrite=self.overwrite,
                    multifile=self.multifile,
                )
                self.already_saved = True

        config_path_dir, config_path_file = os.path.split(config_path)
        self.config_path_new = os.path.join(config_path_dir, self.config_file_original)

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)
        # Copying config file to log dir
        shutil.copyfile(self.config_path_original, self.config_path_new)

class StateDictAwareModelCheckpoint(ModelCheckpoint):
    # necessary as we wish to have one model checkpoint with only state dict and one with standard lightning checkpoints
    # for this, the state key needs to be different, and thus to include this save_weights_only parameter
    def __init__(
        self,
        dirpath: _PATH | None = None,
        filename: str | None = None,
        monitor: str | None = None,
        verbose: bool = False,
        save_last: bool | None = None,
        save_top_k: int = 1,
        mode: str = "min",
        save_weights_only: bool = False,
        auto_insert_metric_name: bool = True,
        every_n_train_steps: int | None = None,
        train_time_interval: timedelta | None = None,
        every_n_epochs: int | None = None,
        save_on_train_epoch_end: bool | None = None,
        enable_version_counter: bool = True,
    ):
        super().__init__(
            dirpath,
            filename,
            monitor,
            verbose,
            save_last,
            save_top_k,
            save_weights_only,
            mode,
            auto_insert_metric_name,
            every_n_train_steps,
            train_time_interval,
            every_n_epochs,
            save_on_train_epoch_end,
            enable_version_counter,
        )

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor=self.monitor,
            mode=self.mode,
            every_n_train_steps=self._every_n_train_steps,
            every_n_epochs=self._every_n_epochs,
            train_time_interval=self._train_time_interval,
            save_weights_only=self.save_weights_only,
        )


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--predict_output_dir", default=None)
        parser.add_argument("--out_dtype", default="int16")
        parser.add_argument("--deploy_config_file", type=bool, default=True)
        parser.add_argument("--custom_modules_path", type=str, default=None)

        # parser.set_defaults({"trainer.enable_checkpointing": False})

        parser.add_lightning_class_args(StateDictAwareModelCheckpoint, "ModelCheckpoint")
        parser.set_defaults({"ModelCheckpoint.filename": "{epoch}", "ModelCheckpoint.monitor": "val/loss"})

        parser.add_lightning_class_args(StateDictAwareModelCheckpoint, "StateDictModelCheckpoint")
        parser.set_defaults(
            {
                "StateDictModelCheckpoint.filename": "{epoch}_state_dict",
                "StateDictModelCheckpoint.save_weights_only": True,
                "StateDictModelCheckpoint.monitor": "val/loss",
            }
        )

        parser.link_arguments("ModelCheckpoint.dirpath", "StateDictModelCheckpoint.dirpath")

    def instantiate_classes(self) -> None:

        super().instantiate_classes()
        # get the predict_output_dir. Depending on the value of run, it may be in the subcommand
        try:
            config = self.config.predict
        except AttributeError:
            config = self.config
        if hasattr(config, "predict_output_dir"):
            self.trainer.predict_output_dir = config.predict_output_dir

        if hasattr(config, "out_dtype"):
            self.trainer.out_dtype = config.out_dtype

        if hasattr(config, "deploy_config_file"):
            self.trainer.deploy_config = config.deploy_config_file

        # Custom modules path
        if hasattr(self.config, "fit") and hasattr(self.config.fit, "custom_modules_path"):
            custom_modules_path = self.config.fit.custom_modules_path
        elif hasattr(self.config, "validate") and hasattr(self.config.validate, "custom_modules_path"):
            custom_modules_path = self.config.validate.custom_modules_path
        elif hasattr(self.config, "test") and hasattr(self.config.test, "custom_modules_path"):
            custom_modules_path = self.config.test.custom_modules_path
        elif hasattr(self.config, "predict") and hasattr(self.config.predict, "custom_modules_path"):
            custom_modules_path = self.config.predict.custom_modules_path
        else:
            custom_modules_path = os.getenv("TERRATORCH_CUSTOM_MODULE_PATH", None)

        import_custom_modules(custom_modules_path)

def build_lightning_cli(
    args: ArgsType = None,
    run=True,  # noqa: FBT002
) -> LightningCLI:
    """Command-line interface to GeospatialCV."""
    # Taken from https://github.com/pangeo-data/cog-best-practices
    rasterio_best_practices = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
        "GDAL_SWATH_SIZE": "200000000",
        "VSI_CURL_CACHE_SIZE": "200000000",
    }
    os.environ.update(rasterio_best_practices)
    if os.environ.get("terratorch_FLOAT_32_PRECISION") is not None:
        precision = os.environ.get("terratorch_FLOAT_32_PRECISION")
        allowed_values = ["highest", "high", "medium"]
        if precision in allowed_values:
            torch.set_float32_matmul_precision(precision)
        else:
            warnings.warn(
                "Found terratorch_FLOAT_32_PRECISION env variable but value was set to precision.\
                Set to one of {allowed_values}. Will be ignored this run.",
                UserWarning,
                stacklevel=1,
            )

    return MyLightningCLI(
        model_class=BaseTask,
        subclass_mode_model=True,
        subclass_mode_data=True,
        seed_everything_default=0,
        save_config_callback=StudioDeploySaveConfigCallback if run else None,
        save_config_kwargs={"overwrite": True},
        args=args,
        # save only state_dict as well as full state. Only state_dict will be used for exporting the model
        trainer_defaults={"callbacks": [CustomWriter(write_interval="batch")]},
        run=run,
    )


class LightningInferenceModel:
    """This class provides all the same functionalities as loading a model from the CLI,
    but makes the model accessible programatically.

    Example usage:
        model = LightningInferenceModel.from_config(
        <config path>,
        <checkpoint path>,
        <output dir path>,
        )
        model.inference(<path to inference data dir>)
    """

    def __init__(
        self,
        trainer: Trainer,
        model: LightningModule,
        datamodule: LightningDataModule,
        checkpoint_path: Path | None = None,
    ):
        self.trainer = trainer
        self.model = model
        self.datamodule = datamodule
        if checkpoint_path:
            weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if "state_dict" in weights:
                weights = weights["state_dict"]
            weights = {k.replace("model.", ""): v for k, v in weights.items() if k.startswith("model.")}
            self.model.model.load_state_dict(weights)

        # dont write
        non_writing_callbacks = []
        for callback in self.trainer.callbacks:
            if isinstance(callback, BasePredictionWriter):
                continue
            non_writing_callbacks.append(callback)
        self.trainer.callbacks = non_writing_callbacks

    @staticmethod
    def from_config(
        config_path: Path,
        checkpoint_path: Path | None = None,
        predict_dataset_bands: list[str] | None = None,
        predict_output_bands: list[str] | None = None,
    ):
        """
        Args:
            config_path (Path): Path to the config of the model to be loaded.
            checkpoint_path (Path): Path to the checkpoint to be loaded.
            predict_dataset_bands (list[str] | None, optional): List of bands present in input data.
                Defaults to None.
        """
        # use cli only to load
        arguments = [
            "--config",
            config_path,
        ]

        if predict_dataset_bands is not None:
            arguments.extend([ "--data.init_args.predict_dataset_bands",
            "[" + ",".join(predict_dataset_bands) + "]",])

        if predict_output_bands is not None:
            arguments.extend([ "--data.init_args.predict_output_bands",
            "[" + ",".join(predict_output_bands) + "]",])

        cli = build_lightning_cli(arguments, run=False)
        trainer = cli.trainer
        # disable logging metrics
        trainer.logger = None
        datamodule = cli.datamodule
        model = cli.model
        return LightningInferenceModel(trainer, model, datamodule, checkpoint_path=checkpoint_path)

    @staticmethod
    def from_task_and_datamodule(
        task: LightningModule,
        datamodule: LightningDataModule,
        checkpoint_path: Path | None = None,
    ):
        trainer = Trainer(
            # precision="16-mixed",
            callbacks=[RichProgressBar()],
        )
        return LightningInferenceModel(trainer, task, datamodule, checkpoint_path=checkpoint_path)

    def inference_on_dir(self, data_root: Path | None = None) -> tuple[torch.Tensor, list[str]]:
        """Perform inference on the given data root directory

        Args:
            data_root (Path): Directory with data to perform inference on
        Returns:
            A tuple with a torch tensor with all predictions and a list of corresponding input file paths

        """

        if data_root:
            self.datamodule.predict_root = data_root
        predictions = self.trainer.predict(model=self.model, datamodule=self.datamodule, return_predictions=True)
        concat_predictions = torch.cat([batch[0] for batch in predictions])
        concat_file_names = flatten([batch[1] for batch in predictions])

        return concat_predictions, concat_file_names

    def inference(self, file_path: Path) -> torch.Tensor:
        """Perform inference on a single input file path

        Args:
            file_path (Path): Path to a single input file.
        Returns:
            A torch tensor with the prediction
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            os.symlink(file_path, os.path.join(tmpdir, os.path.basename(file_path)))
            prediction, file_name = self.inference_on_dir(
                tmpdir,
            )
            return prediction.squeeze(0)

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import yaml

from terratorch.cli_tools import build_lightning_cli

# try:
#     import geobench
# except ImportError as err:
#     msg = "This benchmark requires the geobench module. Please run `pip install geobench`."
#     raise Exception(msg) from err

TASKS = ["agb.yaml", "segmentation_config.yaml", "eurosat.yaml"]

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
handle = "benchmark"
logger = logging.getLogger(handle)


def benchmark(benchmark_run_path, backbone_name=None, freeze_backbone=False, backbone_ckpt=None):
    if os.path.isdir(benchmark_run_path):
        if len(os.listdir(benchmark_run_path)) != 0:
            msg = "Benchmark run directory already exists"
            raise Exception(msg)
    else:
        os.mkdir(benchmark_run_path)

    for task in TASKS:
        logger.info(
            f"Running task {task} with backbone {backbone_name} and freeze_backbone set to {freeze_backbone}..."
        )
        # override some of the configs e.g. so that logs are stored under benchmark directory
        try:
            with open(Path("examples/confs") / task) as f:
                task_config = yaml.safe_load(f)
            if backbone_name:
                task_config["model"]["init_args"]["model_args"]["backbone"] = backbone_name
            if freeze_backbone:
                task_config["model"]["init_args"]["freeze_backbone"] = True
            if backbone_ckpt:
                task_config["model"]["init_args"]["model_args"]["backbone_pretrained_cfg_overlay"] = {
                    "file": backbone_ckpt
                }
            task_config["trainer"]["default_root_dir"] = benchmark_run_path

            if "logger" in task_config["trainer"] and "init_args" in task_config["trainer"]["logger"]:
                # if logger with init_args already defined, change where it is saving logs
                task_config["trainer"]["logger"]["init_args"]["save_dir"] = benchmark_run_path
                task_config["trainer"]["logger"]["init_args"]["name"] = Path(task).stem
            else:
                # if no logger key or it is just defined as true
                task_config["trainer"]["logger"] = {
                    "class_path": "TensorBoardLogger",
                    "init_args": {"save_dir": benchmark_run_path, "name": Path(task).stem},
                }
            with tempfile.NamedTemporaryFile(mode="w") as f:
                yaml.dump(task_config, f)
                f.flush()

                # is this ok?
                arguments = ["fit", "--config", f.name]
                build_lightning_cli(arguments)  # run training
        except Exception as e:
            logger.exception(f"Exception during training: {e}")
            logger.warn("Continuing to next task.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("benchmark_run_path", type=str)
    argparser.add_argument("-backbone_name", type=str)
    argparser.add_argument("-freeze_backbone", action="store_true")
    argparser.add_argument("-backbone_ckpt", type=str)
    args = argparser.parse_args()
    benchmark(
        args.benchmark_run_path,
        backbone_name=args.backbone_name,
        freeze_backbone=args.freeze_backbone,
        backbone_ckpt=args.backbone_ckpt,
    )

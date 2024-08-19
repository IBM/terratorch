import copy
import importlib
from typing import Any

import lightning
import torch.optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def get_layer_id_for_vit(name, max_layer):
    name = name.removeprefix("model.")
    if name.startswith("decoder"):
        return 0
    if name in ("encoder.cls_token", "encoder.mask_token", "encoder.pos_embed"):
        return 0
    elif name.startswith("encoder.patch_embed"):
        return 0
    elif name.startswith("encoder.blocks"):
        layer_id = int(name.split(".")[2])
        return layer_id + 1
    else:
        return max_layer - 1


def optimizer_factory(
    optimizer: str,
    lr: float,
    model: torch.nn.Module,
    optimizer_hparams: dict | None = None,
    scheduler: str | None = None,
    monitor: str = "val_loss",
    scheduler_hparams: dict | None = None,
) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
    """Instantiates an optimizer with a scheduler

    Args:
        optimizer (str): Name of optimizer class.
        lr (float): LR to be used. Will not override lr in optimizer_hparams if present.
        model (torch.nn.Module): Model to be optimized
        optimizer_hparams (dict | None, optional): Parameters to be passed to the optimizer constructor.
            Defaults to None.
        scheduler (str | None, optional): Name of scheduler class. Defaults to None.
        monitor (str, optional): Quantity for the scheduler to monitor. Defaults to "val_loss".
        scheduler_hparams (dict | None, optional): Parameters to be passed to the scheduler constructor.
            Defaults to None.

    Returns:
        lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig
    """
    optimizer_hparams = optimizer_hparams if optimizer_hparams else {}
    try:
        optimizer: type[torch.optim.Optimizer] = getattr(torch.optim, optimizer)
    except AttributeError:
        module_name, class_name = optimizer.rsplit(".", 1)
        optimizer: type[torch.optim.Optimizer] = getattr(importlib.import_module(module_name), class_name)

    if "lr" not in optimizer_hparams:
        optimizer_hparams["lr"] = lr

    instantiation_optimizer_hparams: dict = copy.deepcopy(optimizer_hparams)
    layer_decay = instantiation_optimizer_hparams.pop("layer_decay", False)
    if layer_decay:
        parameter_groups = {}
        base_lr = instantiation_optimizer_hparams["lr"]
        num_layers = instantiation_optimizer_hparams.pop("num_layers")
        decay_rate = instantiation_optimizer_hparams.pop("decay_rate", 0.95)
        weight_decay = instantiation_optimizer_hparams.get("weight_decay", 0)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if len(param.shape) == 1 or name.endswith(".bias") or name in ("pos_embed", "cls_token"):
                group_name = "no_decay"
                this_weight_decay = 0.0
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            layer_id = get_layer_id_for_vit(name, num_layers)
            group_name = f"layer_{layer_id}_{group_name}"

            if group_name not in parameter_groups:
                scale = decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [],
                    "lr_scale": scale,
                    "group_name": group_name,
                    "lr": scale * base_lr,
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)
        optimizer = optimizer(
            parameter_groups.values(),
            **instantiation_optimizer_hparams,
        )
    else:
        optimizer = optimizer(model.parameters(), **instantiation_optimizer_hparams)

    if scheduler is None:
        return {"optimizer": optimizer}

    try:
        scheduler: type[torch.optim.lr_scheduler.LRScheduler] = getattr(torch.optim.lr_scheduler, scheduler)
    except AttributeError:
        module_name, class_name = scheduler.rsplit(".", 1)
        scheduler: type[torch.optim.lr_scheduler.LRScheduler] = getattr(
            importlib.import_module(module_name), class_name
        )

    scheduler_hparams = scheduler_hparams if scheduler_hparams else {}
    interval = scheduler_hparams.get("interval", "epoch")
    scheduler_hparams_no_interval = {k: v for k, v in scheduler_hparams.items() if k != "interval"}
    scheduler = scheduler(optimizer, **scheduler_hparams_no_interval)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "monitor": monitor, "interval": interval},
    }

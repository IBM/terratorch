from typing import Any

import lightning
import torch.optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def optimizer_factory(
    optimizer: str,
    lr: float,
    params_to_be_optimized: Any,
    optimizer_hparams: dict | None = None,
    scheduler: str | None = None,
    monitor: str = "val_loss",
    scheduler_hparams: dict | None = None,
) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
    """Instantiates an optimizer with a scheduler

    Args:
        optimizer (str): Name of optimizer class from torch.optim.
        lr (float): Learning rate
        params_to_be_optimized (Any): Parameters to be optimized
        optimizer_hparams (dict | None, optional): Parameters to be passed to the optimizer constructor.
            Defaults to None.
        scheduler (str | None, optional): Name of scheduler class from torch.optim.lr_scheduler. Defaults to None.
        monitor (str, optional): Quantity for the scheduler to monitor. Defaults to "val_loss".
        scheduler_hparams (dict | None, optional): Parameters to be passed to the scheduler constructor.
            Defaults to None.

    Returns:
        lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig
    """
    optimizer_hparams = optimizer_hparams if optimizer_hparams else {}
    optimizer = getattr(torch.optim, optimizer)
    optimizer = optimizer(params_to_be_optimized, lr=lr, **optimizer_hparams)

    if scheduler is None:
        return {"optimizer": optimizer}

    scheduler = getattr(torch.optim.lr_scheduler, scheduler)
    scheduler_hparams = scheduler_hparams if scheduler_hparams else {}
    interval = scheduler_hparams.get("interval", "epoch")
    scheduler_hparams_no_interval = {k: v for k, v in scheduler_hparams.items() if k != "interval"}
    scheduler = scheduler(optimizer, **scheduler_hparams_no_interval)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "monitor": monitor, "interval": interval},
    }

from typing import Any

import lightning
import torch.optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LambdaFns():
    def __init__(
            milestone,
        ) -> None:
        self.milestone = milestone

    def linear_warmup(self, current_step: int):
        return float(current_step / self.milestone)
    #cosine warmup
    #etc


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

    scheduler_hparams = scheduler_hparams if scheduler_hparams else {}
    interval = scheduler_hparams.get("interval", "epoch")
    scheduler_hparams_no_interval = {k: v for k, v in scheduler_hparams.items() if k != "interval"}

    #unpack sequential schedule
    if scheduler == "SequentialLR":
        assert "schedulers" in scheduler_hparams_no_interval, "Please provide scheduler for SequentialLR"
        assert "milestones" in scheduler_hparams_no_interval, "Please provide milestones for SequentialLR"
        schedule_sequence = []
        milestones = []
        for k, v in scheduler_hparams_no_interval["schedulers"].items():
            if k == "LambdaLR":
                lr_lambda = v.get("lr_lambda", "linear_warmup")
                #v["lr_lambda"] =  getattr(LambdaFns, lr_lambda) 
                #defin lambd a funtcion as a partial???
                def linear_warmup(current_step: int):
                    return float(current_step / 5)
                v["lr_lambda"] = linear_warmup
                print(f'v["lr_lambda"] : {v["lr_lambda"]}')
            nested_scheduler = getattr(torch.optim.lr_scheduler, k)
            nested_scheduler = nested_scheduler(optimizer, **v)
            schedule_sequence.append(nested_scheduler)
        scheduler_hparams_no_interval["schedulers"] = schedule_sequence

    scheduler = getattr(torch.optim.lr_scheduler, scheduler)
    scheduler = scheduler(optimizer, **scheduler_hparams_no_interval)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "monitor": monitor, "interval": interval},
    }

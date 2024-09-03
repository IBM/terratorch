from collections.abc import Callable

from torch import Tensor

from terratorch.models.model import ModelOutput


class LossHandler:
    """Class to help handle the computation and logging of loss"""

    def __init__(self, loss_prefix: str, multiple_inputs:bool=False) -> None:
        """Constructor

        Args:
            loss_prefix (str): Prefix to be prepended to all the metrics (e.g. training).
        """
        self.loss_prefix = loss_prefix
        self.multiple_inputs = multiple_inputs

        if self.multiple_inputs:
            self._eval_loss = self._eval_loss_multiple_inputs
        else:
            self._eval_loss = self._eval_loss_single_input

    def _eval_loss_single_input(self, model_output, ground_truth, criterion):
        loss = self._compute_loss(model_output.output, ground_truth, criterion)
        return loss

    def _eval_loss_multiple_inputs(self, model_output, ground_truth, criterion):
        loss = self._compute_loss(model_output, ground_truth, criterion)
        return loss

    def compute_loss(
        self,
        model_output: ModelOutput,
        ground_truth: Tensor,
        criterion: Callable,
        aux_loss_weights: dict[str, float] | None,
    ) -> dict[str, Tensor]:
        """Compute the loss for the mean decode head as well as other heads

        Args:
            model_output (ModelOutput): Output from the model
            ground_truth (Tensor): Tensor with labels
            criterion (Callable): Loss function to be applied
            aux_loss_weights (Union[dict[str, float], None]): Dictionary of names of model auxiliary
                heads and their weights

        Raises:
            Exception: If the keys in aux_loss_weights and the model output do not match, will raise an exception.

        Returns:
            dict[str, Tensor]: Dictionary of computed losses. Total loss is returned under the key "loss".
                If there are auxiliary heads, the main decode head is returned under the key "decode_head".
                All other heads are returned with the same key as their name.
        """
        loss = self._eval_loss(model_output, ground_truth, criterion)
        if not model_output.auxiliary_heads:
            return {"loss": loss}

        if aux_loss_weights is None:
            msg = "Auxiliary heads given with no aux_loss_weights"
            raise Exception(msg)
        all_losses = {}
        all_losses["decode_head"] = loss
        total_loss = loss.clone()
        # incorporate aux heads
        model_output_names = set(model_output.auxiliary_heads.keys())
        aux_loss_names = set(aux_loss_weights.keys())
        if aux_loss_names != model_output_names:
            msg = f"Found difference in declared auxiliary losses and model outputs.\n \
                Found in declared losses but not in model output: {aux_loss_names - model_output_names}. \n \
                Found in model output but not in delcared losses: {model_output_names - aux_loss_names}"
            raise Exception(msg)

        for loss_name, loss_weight in aux_loss_weights.items():
            output = model_output.auxiliary_heads[loss_name]
            loss_value: Tensor = self._compute_loss(output, ground_truth, criterion)
            all_losses[loss_name] = loss_value
            total_loss = total_loss + loss_value * loss_weight

        all_losses["loss"] = total_loss
        return all_losses

    def _compute_loss(self, y_hat: Tensor, ground_truth: Tensor, criterion: Callable):
        loss: Tensor = criterion(y_hat, ground_truth)
        return loss

    def log_loss(
        self, log_function: Callable, loss_dict: dict[str, Tensor] | None = None, batch_size: int | None = None
    ) -> None:
        """Log the loss. If auxiliary heads exist, log the full loss suffix "loss", and then all other losses.

        Args:
            log_function (Callable): _description_
            loss_dict (dict[str, Tensor], optional): _description_. Defaults to None.
        """

        # dont alter passed dict
        all_losses = dict(loss_dict)
        full_loss = all_losses.pop("loss")
        log_function(f"{self.loss_prefix}loss", full_loss, sync_dist=True, batch_size=batch_size)

        for loss_name, loss_value in all_losses.items():
            log_function(
                f"{self.loss_prefix}{loss_name}",
                loss_value,
                on_epoch=True,
                on_step=True,
                sync_dist=True,
                batch_size=batch_size,
            )

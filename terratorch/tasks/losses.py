
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from kornia.losses.hausdorff import _HausdorffERLossBase

class HausdorffERLoss(_HausdorffERLossBase):
    """
    Multiclass 2D Hausdorff-ER loss with `ignore_index` support.

    This class inherits Kornia's `_HausdorffERLossBase` and mimics the 2D version. It neutralizes ignored pixels
    so they contribute zero to the erosion-based Hausdorff surrogate (see Karimi & Salcudean, 2019).

    Args:
        alpha (float): controls the erosion rate in each iteration.
        k (int): number of erosion iterations.
        reduction (str): 'none' | 'mean' | 'sum'.
        ignore_index (int | None): label value to ignore in `target`. If None, nothing is ignored.
        neutral_class (int): class index to assign at ignored pixels (default: 0).
        from_logits (bool): if True, apply softmax to `pred` along channel dim before neutrality (default: False).

    Shapes:
        pred:   (B, C, H, W)  — probabilities in [0,1] if `from_logits=False`; raw logits otherwise.
        target: (B, 1, H, W)  — Long indices in [0, C-1] or = `ignore_index`.

    Returns:
        Tensor: loss as per `reduction`. 'none' -> (B,) per-sample loss.
    """

    conv = torch.conv2d
    max_pool = nn.AdaptiveMaxPool2d(1)

    def __init__(
        self,
        alpha: float = 2.0,
        k: int = 10,
        reduction: str = "mean",
        ignore_index: int | None = None,
        neutral_class: int = 0,
        from_logits: bool = False,
    ) -> None:
        super().__init__(alpha=alpha, k=k, reduction=reduction)
        self.ignore_index = ignore_index
        self.neutral_class = neutral_class
        self.from_logits = from_logits

    # same 3x3 cross kernel as the 2D version
    def get_kernel(self) -> Tensor:
        cross = torch.tensor([[[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]]], dtype=torch.float32, device=None)
        kernel = cross * 0.2
        return kernel[None]  # (1,1,3,3)

    @staticmethod
    def _force_one_hot_at_mask(probs: Tensor, mask: Tensor, class_idx: int) -> Tensor:
        """
        Set probs to one-hot (value=1) for `class_idx` at positions where mask==True; others 0.
        probs: (B, C, H, W) ; mask: (B, 1, H, W) bool
        """
        out = probs.clone()
        # zero all channels at ignored positions
        out = out.masked_fill(mask.expand_as(out), 0.0)
        # set neutral channel to 1.0 at ignored positions
        out[:, class_idx:class_idx+1].masked_fill_(mask, 1.0)
        return out

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.dim() != 4:
            raise ValueError(f"Only 2D images supported. Got pred.dim()={pred.dim()}.")

        B, C, H, W = pred.shape
        if target.shape == (B, H, W):
            target = target.unsqueeze(1)
        elif target.shape != (B, 1, H, W):
            raise ValueError(f"Expected target shape (B,1,H,W), got {tuple(target.shape)}.")
        if target.dtype != torch.long:
            raise ValueError("target must be torch.long with class indices.")

        if self.neutral_class < 0 or self.neutral_class >= C:
            raise ValueError(f"`neutral_class` {self.neutral_class} out of range [0, {C-1}].")

        # Optional softmax to probabilities
        probs = F.softmax(pred, dim=1) if self.from_logits else pred

        # Validate labels (allow ignore_index if provided)
        tmin, tmax = int(target.min()), int(target.max())
        if self.ignore_index is None:
            if not (tmin >= 0 and tmax < C):
                raise ValueError(f"Expect target in [0,{C-1}]. Got range [{tmin},{tmax}].")
        else:
            # all non-ignored must be within [0, C-1]
            non_ign = target[target != self.ignore_index]
            if non_ign.numel() > 0:
                nmin, nmax = int(non_ign.min()), int(non_ign.max())
                if not (nmin >= 0 and nmax < C):
                    raise ValueError(f"Non-ignored labels must be in [0,{C-1}]. Got [{nmin},{nmax}].")

        # Handle ignore_index by neutralizing predictions and remapping target
        if self.ignore_index is not None:
            ignore_mask = (target == self.ignore_index)
            if ignore_mask.any():
                probs = self._force_one_hot_at_mask(probs, ignore_mask, self.neutral_class)
                target = target.clone()
                target[ignore_mask] = self.neutral_class

        return super().forward(probs, target)
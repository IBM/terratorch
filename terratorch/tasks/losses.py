import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

class HausdorffDTLoss(nn.Module):
    """
    Simplified Distance Transform Hausdorff Loss (Karimi et al., TMI 2020),
    inspired by MONAI's HausdorffDTLoss implementation:
    https://github.com/Project-MONAI/MONAI/blob/1.5.1/monai/losses/hausdorff_loss.py

    Computes:
        L = mean( (p - g)^2 * (dt_pred^alpha + dt_gt^alpha) )

    Args:
        alpha (float): exponent for distance weighting (default=2.0)
        ignore_index (int or None): label to ignore in target
        from_logits (bool): if True, apply softmax to input
    """

    def __init__(self, alpha: float = 2.0, ignore_index: int | None = None, from_logits: bool = True):
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.from_logits = from_logits

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input: (B, C, H, W), target: (B, H, W) or (B, 1, H, W)
        if target.ndim == input.ndim - 1:
            target = target.unsqueeze(1)  # (B,1,H,W)

        B, C, H, W = input.shape
        if self.from_logits:
            input = F.softmax(input, dim=1)

        # One-hot encode target
        target_idx = target.long().squeeze(1)  # (B,H,W)
        target_oh = F.one_hot(target_idx.clamp(min=0), num_classes=C).permute(0, 3, 1, 2).float()

        # Mask ignore_index
        valid_mask = torch.ones_like(target_oh[:, :1])
        if self.ignore_index is not None:
            ignore_mask = (target_idx == self.ignore_index).unsqueeze(1)
            valid_mask = (~ignore_mask).float()
            target_oh = target_oh * valid_mask  # zero out ignored pixels

        loss = 0.0
        for c in range(C):
            pred_c = input[:, c:c+1]
            gt_c = target_oh[:, c:c+1]

            # Compute distance transforms on CPU (SciPy)
            dt_pred = torch.zeros_like(pred_c)
            dt_gt = torch.zeros_like(gt_c)
            for b in range(B):
                p_bin = (pred_c[b, 0] > 0.5).cpu().numpy()
                g_bin = (gt_c[b, 0] > 0.5).cpu().numpy()
                dt_pred[b, 0] = torch.from_numpy(distance_transform_edt(~p_bin) + distance_transform_edt(p_bin))
                dt_gt[b, 0] = torch.from_numpy(distance_transform_edt(~g_bin) + distance_transform_edt(g_bin))

            weight = dt_pred.pow(self.alpha) + dt_gt.pow(self.alpha)
            sq_err = (pred_c - gt_c) ** 2
            loss += (sq_err * weight * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        return loss / C

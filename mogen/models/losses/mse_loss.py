import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


def gmof(x, sigma):
    """Geman-McClure error function.

    Args:
        x (torch.Tensor): The input tensor.
        sigma (float): The sigma value used in the calculation.

    Returns:
        torch.Tensor: The computed Geman-McClure error.
    """
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


@weighted_loss
def mse_loss(pred, target):
    """Wrapper for Mean Squared Error (MSE) loss.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: MSE loss.
    """
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def smooth_l1_loss(pred, target):
    """Wrapper for Smooth L1 loss.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Smooth L1 loss.
    """
    return F.smooth_l1_loss(pred, target, reduction='none')


@weighted_loss
def l1_loss(pred, target):
    """Wrapper for L1 loss.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: L1 loss.
    """
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss_with_gmof(pred, target, sigma):
    """Extended MSE Loss with Geman-McClure function applied.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values.
        sigma (float): The sigma value for the Geman-McClure function.

    Returns:
        torch.Tensor: The loss value.
    """
    loss = F.mse_loss(pred, target, reduction='none')
    loss = gmof(loss, sigma)
    return loss


@LOSSES.register_module()
class MSELoss(nn.Module):
    """Mean Squared Error (MSE) Loss.

    Args:
        reduction (str, optional): The method to reduce the loss to a scalar.
            Options are 'none', 'mean', and 'sum'. Defaults to 'mean'.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = 'none' if reduction is None else reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function to compute loss.

        Args:
            pred (torch.Tensor): Predictions.
            target (torch.Tensor): Ground truth.
            weight (torch.Tensor, optional): Optional weight per sample.
            avg_factor (int, optional): Factor for averaging the loss.
            reduction_override (str, optional): Option to override reduction method.

        Returns:
            torch.Tensor: Calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * mse_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss


@LOSSES.register_module()
class KinematicLoss(nn.Module):
    """Kinematic Loss for hierarchical motion prediction.
    
    Args:
        reduction (str, optional): Reduction method ('none', 'mean', or 'sum').
        loss_type (str, optional): The type of loss to use ('mse', 'smooth_l1', 'l1').
        loss_weight (list[float], optional): List of weights for each stage of the hierarchy.
    """

    def __init__(self, reduction='mean', loss_type='mse', loss_weight=[1.0]):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = 'none' if reduction is None else reduction
        self.loss_weight = loss_weight
        self.num_stages = len(loss_weight)

        # Select loss function based on loss_type
        if loss_type == 'mse':
            self.loss_func = mse_loss
        elif loss_type == 'smooth_l1':
            self.loss_func = smooth_l1_loss
        elif loss_type == 'l1':
            self.loss_func = l1_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function for hierarchical kinematic loss.

        Args:
            pred (torch.Tensor): The prediction tensor.
            target (torch.Tensor): The target tensor.
            weight (torch.Tensor, optional): Weights for each prediction. Defaults to None.
            avg_factor (int, optional): Factor to average the loss. Defaults to None.
            reduction_override (str, optional): Override reduction method. Defaults to None.

        Returns:
            torch.Tensor: The calculated hierarchical loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        total_loss = 0
        pred_t = pred.clone()
        target_t = target.clone()

        # Apply loss function across stages
        for i in range(self.num_stages):
            stage_loss = self.loss_weight[i] * self.loss_func(
                pred_t, target_t, weight, reduction=reduction, avg_factor=avg_factor)
            total_loss += stage_loss

            # Compute differences between consecutive frames
            pred_t = torch.cat((pred_t[:, :1, :], pred_t[:, 1:] - pred_t[:, :-1]), dim=1)
            target_t = torch.cat((target_t[:, :1, :], target_t[:, 1:] - target_t[:, :-1]), dim=1)

        return total_loss

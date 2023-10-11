"""
Loss function for ESMDynamic model.
"""

import torch
from torch import nn
from torchvision.ops.focal_loss import sigmoid_focal_loss
from .utils import rmsd_vals


def binned_cross_entropy(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        bin_vals: torch.Tensor,
        ignore_indices: torch.Tensor = None,
        reduction: str = "none",
) -> torch.Tensor:
    """Cross entropy loss where incorrectly labeled samples are weighted by how far off the prediction was.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions (unnormalized logits) for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        bin_vals (Tensor): Bin centers for each class.
        ignore_indices (Tensor): Indices to ignore in loss calculation.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")  # Check output
    p = nn.functional.softmax(inputs, dim=-1)  # Check shape is correct
    bin_values = bin_vals[torch.argmax(p, dim=1)]  # Figure out dim for argmax
    target_bin_values = bin_vals[torch.argmax(targets, dim=1)]  # Check output
    loss = torch.abs(bin_values - target_bin_values) / len(bin_vals) * ce_loss  # Ensure shapes are correct
    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )

    return loss


def full_form_loss(
        inputs,
        target,
        alpha=0.25,
        gamma=2,
        reduction="none",
) -> torch.Tensor:
    """Loss for ESMDynamic model.
    """
    dynamic_contact_logits = inputs['dynamic_contact_logits']
    dynamic_contacts_target = target['dynamic_contacts']
    loss_dyn_contact = sigmoid_focal_loss(
        dynamic_contact_logits,
        dynamic_contacts_target,
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
    )

    rmsd_logits = inputs['rmsd_logits']
    rmsd_target = target['rmsd']
    loss_rmsd = binned_cross_entropy(
        rmsd_logits,
        rmsd_target,
        bin_vals=rmsd_vals,
        reduction=reduction,
    )

    return loss_rmsd + loss_dyn_contact

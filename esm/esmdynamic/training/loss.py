"""
Loss function for ESMDynamic model.
"""

import torch
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss


############################################################
# Masks
############################################################

def length_mask_2d(B, L, lengths, device):
    """
    Make a (B, L, L) mask of valid residues.
    """
    m = torch.zeros((B, L, L), dtype=torch.bool, device=device)
    for i, Li in enumerate(lengths):
        m[i, :Li, :Li] = True
    return m


def length_mask_1d(B, L, lengths, device):
    """
    Make a (B, L) 1D mask.
    """
    m = torch.zeros((B, L), dtype=torch.bool, device=device)
    for i, Li in enumerate(lengths):
        m[i, :Li] = True
    return m


############################################################
# 1. Dynamic contact loss
############################################################

def loss_dynamic_logits(logits, target, lengths, alpha=0.25, gamma=2):
    """
    logits: [B, C, L, L] or [B, n_conditions, L, L]
    target: same
    """
    B, C, L, _ = logits.shape
    mask = length_mask_2d(B, L, lengths, logits.device)

    loss = sigmoid_focal_loss(
        logits,
        target,
        reduction="none",
        alpha=alpha,
        gamma=gamma,
    )
    return (loss * mask[:, None]).sum()


def loss_dynamic_conf(conf_pred, conf_target, lengths):
    """
    conf_pred: [B, n_conditions, L]
    """
    B, C, L = conf_pred.shape
    mask = length_mask_1d(B, L, lengths, conf_pred.device)
    loss = (conf_pred - conf_target).pow(2)
    loss = loss * mask[:, None]
    return loss.sum()


############################################################
# 2. Kinetics loss (multiclass)
############################################################

def loss_kinetic_logits(logits, labels, lengths, class_weights):
    """
    logits: [B, C, 2, L, L, K]
    labels: [B, C, 2, L, L]
    class_weights: [2, K]  # weights for off-rate and on-rate
    """
    B, C, R, L, _, K = logits.shape
    assert R == 2, "Expected n_rates = 2"

    # Mask [B, L, L] → [B, 1, 1, L, L]
    mask = length_mask_2d(B, L, lengths, logits.device)
    mask = mask[:, None, None, :, :].bool()   # now broadcastable over [C, R]

    # Apply mask separately to each rate
    # OFF-TIME (rate 0)
    logits_off  = logits[:, :, 0][mask]  # → [N_off, K]
    labels_off  = labels[:, :, 0][mask]  # → [N_off]

    # ON-TIME (rate 1)
    logits_on   = logits[:, :, 1][mask]  # → [N_on, K]
    labels_on   = labels[:, :, 1][mask]  # → [N_on]

    # Compute CE for each
    loss_off = F.cross_entropy(logits_off, labels_off.long(), weight=class_weights[0])
    loss_on  = F.cross_entropy(logits_on,  labels_on.long(),  weight=class_weights[1])

    return loss_off + loss_on


def loss_kinetic_conf(conf_pred, conf_target, lengths):
    """
    conf_pred: [B, n_conditions, n_rates, L]
    """
    B, C, R, L = conf_pred.shape
    mask = length_mask_1d(B, L, lengths, conf_pred.device)
    mask = mask[:, None, None, :]  # [B,1,1,L]

    loss = (conf_pred - conf_target).pow(2)
    return (loss * mask).sum()


############################################################
# 3. Frequency regression losses
############################################################

def loss_frequency(freq_pred, freq_true, lengths):
    """
    freq_pred: [B, n_conditions, L, L]
    """
    B, C, L, _ = freq_pred.shape
    mask = length_mask_2d(B, L, lengths, freq_pred.device)
    loss = (freq_pred - freq_true).pow(2)
    return (loss * mask[:, None]).sum()


def loss_frequency_residual(freq_res_pred, freq_res_target, lengths):
    """
    freq_res_pred: [B, n_conditions, L, L]
    """
    B, C, L, _ = freq_res_pred.shape
    mask = length_mask_2d(B, L, lengths, freq_res_pred.device)
    loss = (freq_res_pred - freq_res_target).pow(2)
    return (loss * mask[:, None]).sum()


############################################################
# Modular wrapper
############################################################

LOSS_FUNCS = {
    "kinetic_logits": loss_kinetic_logits,
    "kinetic_confidence": loss_kinetic_conf,
    "frequency_pred": loss_frequency,
    "frequency_residual_pred": loss_frequency_residual,
    "dynamic_logits": loss_dynamic_logits,
    "dynamic_confidence": loss_dynamic_conf,
}

def esmdynamic_loss(outputs, targets, lengths, active_heads):
    """
    outputs: dict containing only heads that were loaded
    targets: dict containing ground truths for those heads
    lengths: [B] tensor of residue lengths
    active_heads: list or set of head names
    """

    total = 0.0
    count = 0

    for head in active_heads:
        if head not in LOSS_FUNCS:
            continue
        loss_fn = LOSS_FUNCS[head]
        loss = loss_fn(outputs[head], targets[head], lengths)
        total += loss
        count += 1

    if count == 0:
        raise RuntimeError("No valid heads found for loss computation.")

    return total / count

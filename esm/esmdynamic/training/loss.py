"""
Loss function for ESMDynamic model.
"""

import torch
from torch import nn
from torchvision.ops.focal_loss import sigmoid_focal_loss


def filter_dyn_contacts_loss(loss_dyn_contact, protein_lengths):
    for i in range(len(protein_lengths)):
        L = protein_lengths[i]
        loss_dyn_contact[i, :, L:, L:] = 0.
    return loss_dyn_contact


def dyn_contact_loss(
        inputs,
        target,
        alpha=0.25,
        gamma=2,
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
        reduction="none",
    )

    protein_lengths = target['protein_lengths']

    filtered_loss_dyn_contacts = filter_dyn_contacts_loss(loss_dyn_contact, protein_lengths)
    final_loss = filtered_loss_dyn_contacts.sum()
    return final_loss


def get_accuracy_metrics(pred, labels):
    """Useful metrics to track model performance.
    """
    dynamic_contact_pred = pred["dynamic_contact_pred"]
    dynamic_contact_label = labels["dynamic_contacts"]
    prot_lengths = labels["protein_lengths"]

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for p, l, length in zip(dynamic_contact_pred, dynamic_contact_label, prot_lengths):
        tp += torch.sum(torch.logical_and((p[0, :length, :length] == 1), (l[0, :length, :length] == 1)))
        tn += torch.sum(torch.logical_and((p[0, :length, :length] == 0), (l[0, :length, :length] == 0)))
        fp += torch.sum(torch.logical_and((p[0, :length, :length] == 1), (l[0, :length, :length] == 0)))
        fn += torch.sum(torch.logical_and((p[0, :length, :length] == 0), (l[0, :length, :length] == 1)))
    dyn_cont_acc = (tp + tn) / (tp + tn + fp + fn)
    dyn_cont_bal_acc = .5*(tp/(tp+fn) + tn/(tn+fp))
    dyn_cont_tpr = (tp) / (tp + fn)
    dyn_cont_prec = tp / (tp + fp)
    dyn_cont_f1s = (2*tp) / (2*tp + fp + fn)

    return torch.Tensor([dyn_cont_acc, dyn_cont_bal_acc, dyn_cont_tpr, dyn_cont_prec, dyn_cont_f1s])
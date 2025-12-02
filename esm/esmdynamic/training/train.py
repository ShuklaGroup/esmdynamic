"""
New training script for ESMDynamic (works with the uploaded model/loss/dataset).
Saves only trained heads (best val loss and last checkpoint).
"""

import os
import argparse
from datetime import datetime
import math

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# ---------- Package imports ----------
# Assumes the project is installed as a package so these imports resolve
from esm.esmdynamic.esmdynamic import ESMDynamic
import esm.esmdynamic.training.loss as loss_mod
from esm.esmdynamic.training.data_reader import DynContactDataset


# ----------------- Helpers: datasets / loaders -----------------
def init_datasets(
    training_identifiers_file,
    validation_identifiers_file,
    data_dir,
    crop_length=256,
    training_weight_file="../../mdcath_featurization/splits/splits_20_id_cutoff/train_weights.pt",
    validation_weight_file="../../mdcath_featurization/splits/splits_20_id_cutoff/val_weights.pt",
):
    cluster_train = list(np.loadtxt(training_identifiers_file, dtype=str))
    cluster_val = list(np.loadtxt(validation_identifiers_file, dtype=str))

    training_weights = torch.load(training_weight_file) if training_weight_file is not None else None
    validation_weights = torch.load(validation_weight_file) if validation_weight_file is not None else None

    training_set = DynContactDataset(
        data_dir=data_dir,
        identifiers=cluster_train,
        crop_length=crop_length,
    )
    if training_weights is not None:
        training_set.weights = training_weights

    validation_set = DynContactDataset(
        data_dir=data_dir,
        identifiers=cluster_val,
        crop_length=crop_length,
    )
    if validation_weights is not None:
        validation_set.weights = validation_weights

    return training_set, validation_set


def init_data_loaders(training_set, validation_set, batch_size=4, train_samples_per_epoch=10000, val_samples_per_epoch=1000):
    training_sampler = training_set.weighted_random_sampler(num_samples=train_samples_per_epoch)
    training_loader = DataLoader(training_set, batch_size=batch_size, sampler=training_sampler, collate_fn=training_set.custom_collate_fn)

    validation_sampler = validation_set.weighted_random_sampler(num_samples=val_samples_per_epoch)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, sampler=validation_sampler, collate_fn=validation_set.custom_collate_fn)

    return training_loader, validation_loader


# ----------------- Head selection / model init -----------------
def select_prefixes_from_loss_heads(loss_heads):
    prefixes = set()
    for h in loss_heads:
        if "_" in h:
            prefixes.add(h.split("_")[0])
        else:
            prefixes.add(h)
    return prefixes


def init_model(chunk_size=256, device="cuda", pretrained=None, heads_to_load=None):
    model = ESMDynamic(load_esmfold=True, heads_to_load=heads_to_load)
    if pretrained:
        sd = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(sd, strict=False)
    if device == "cuda":
        model.cuda()
    model.set_chunk_size(chunk_size)
    return model


def init_optimizer_for_heads(model, head_prefixes, lr=1e-4):
    params = []
    for h in head_prefixes:
        if h not in model.heads:
            raise RuntimeError(f"Head '{h}' not found. Available: {list(model.heads.keys())}")
        params += list(model.heads[h].parameters())
    if len(params) == 0:
        raise RuntimeError("No parameters selected for optimizer.")
    return torch.optim.Adam(params, lr=lr)


def init_writer(outpath):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(outpath, "runs", f"trainer_{timestamp}"))
    return timestamp, writer


def save_head_state_dicts(model, head_prefixes, outpath, prefix_label, timestamp):
    for h in sorted(head_prefixes):
        fname = os.path.join(outpath, f"{h}_head_{prefix_label}_{timestamp}.pt")
        sd = model.heads[h].state_dict()
        torch.save({f"heads.{h}.{k}": v for k, v in sd.items()}, fname)
        print(f"Saved head '{h}' -> {fname}")


# ----------------- Target creation helpers -----------------
def _length_masks_from_lengths(lengths, Lmax, device):
    """
    Returns:
      mask2d: [B, Lmax, Lmax] boolean mask
      mask1d: [B, Lmax] boolean mask
    """
    B = lengths.shape[0]
    mask2d = torch.zeros((B, Lmax, Lmax), dtype=torch.bool, device=device)
    mask1d = torch.zeros((B, Lmax), dtype=torch.bool, device=device)
    for i, Li in enumerate(lengths):
        Li = int(Li.item())
        mask2d[i, :Li, :Li] = True
        mask1d[i, :Li] = True
    return mask2d, mask1d


def build_confidence_targets_dynamic(pred_pairs, true_pairs, lengths):
    """
    pred_pairs: [B, C, L, L] logits (or probabilities). We'll threshold at 0.5 if logits -> sigmoid.
    true_pairs: [B, C, L, L] (0/1)
    produce conf_target: [B, C, L] per-residue accuracy across partner residues.
    """
    device = true_pairs.device
    B, C, L, _ = true_pairs.shape

    # Convert logits to binary preds if needed
    if pred_pairs.dtype.is_floating_point:
        probs = torch.sigmoid(pred_pairs)
        pred_bin = (probs > 0.5).long()
    else:
        pred_bin = pred_pairs.long()

    conf_target = torch.zeros((B, C, L), dtype=torch.float32, device=device)

    for b in range(B):
        Lb = int(lengths[b].item())
        if Lb == 0:
            continue
        # For each residue i, compute accuracy across j in 0..Lb-1
        tp = (pred_bin[b, :, :Lb, :Lb] == true_pairs[b, :, :Lb, :Lb]).float()  # [C, Lb, Lb]
        # accuracy for residue i is mean over axis=1 (partners), so for residue i: mean over tp[:, i, :]
        # We want conf_target[b, c, i] = mean(tp[c, i, :])
        conf_target[b, :, :Lb] = tp.mean(dim=2)  # tp.shape [C, Lb, Lb] -> mean over last dim -> [C, Lb]
    return conf_target  # [B, C, L]


def build_confidence_targets_kinetic(pred_logits, true_labels, lengths):
    """
    pred_logits: [B, C, R, L, L, K] logits for K classes
    true_labels: [B, C, R, L, L] integer labels
    produce conf_target: [B, C, R, L] per-rate per-residue accuracy across partner residues.
    """
    device = true_labels.device
    B, C, R, L, _, K = pred_logits.shape
    preds = pred_logits.argmax(dim=-1)  # [B, C, R, L, L]
    conf_target = torch.zeros((B, C, R, L), dtype=torch.float32, device=device)

    for b in range(B):
        Lb = int(lengths[b].item())
        if Lb == 0:
            continue
        # For each rate r and residue i: accuracy across j
        for r in range(R):
            # equality map [C, Lb, Lb]
            eq = (preds[b, :, r, :Lb, :Lb] == true_labels[b, :, r, :Lb, :Lb]).float()
            # per residue accuracy for residue i: mean over partners axis=2
            conf_target[b, :, r, :Lb] = eq.mean(dim=2)
    return conf_target  # [B, C, R, L]


def build_frequency_residual_target(freq_pred, freq_true):
    """
    Both freq_pred / freq_true: [B, C, L, L]
    Target for residual head is absolute difference.
    """
    return torch.abs(freq_true - freq_pred)


# ----------------- Metrics -----------------
def safe_div(numer, denom):
    return numer / denom if denom != 0 else 0.0


def metrics_dynamic_batch(pred_logits, true_pairs, lengths):
    """
    Compute accuracy, precision, recall (TPR), F1, balanced acc for the batch.
    - pred_logits: [B, C, L, L] (logits or probabilities)
    - true_pairs: [B, C, L, L] (0/1)
    Returns dict with scalars (batch-averaged).
    """
    device = true_pairs.device
    B, C, L, _ = true_pairs.shape

    # convert logits -> binary
    if pred_logits.dtype.is_floating_point:
        probs = torch.sigmoid(pred_logits)
        preds = (probs > 0.5).long()
    else:
        preds = pred_logits.long()

    total_TP = total_FP = total_FN = total_TN = 0
    total_counts = 0

    for b in range(B):
        Lb = int(lengths[b].item())
        if Lb == 0:
            continue
        t = true_pairs[b, :, :Lb, :Lb].reshape(-1)
        p = preds[b, :, :Lb, :Lb].reshape(-1)
        total_counts += t.numel()
        TP = int(((p == 1) & (t == 1)).sum().item())
        TN = int(((p == 0) & (t == 0)).sum().item())
        FP = int(((p == 1) & (t == 0)).sum().item())
        FN = int(((p == 0) & (t == 1)).sum().item())
        total_TP += TP
        total_TN += TN
        total_FP += FP
        total_FN += FN

    # compute metrics
    accuracy = safe_div((total_TP + total_TN), (total_counts)) if total_counts > 0 else 0.0
    precision = safe_div(total_TP, (total_TP + total_FP)) if (total_TP + total_FP) > 0 else 0.0
    recall = safe_div(total_TP, (total_TP + total_FN)) if (total_TP + total_FN) > 0 else 0.0
    f1 = safe_div(2 * precision * recall, (precision + recall)) if (precision + recall) > 0 else 0.0
    # balanced accuracy = (TPR + TNR)/2
    tpr = recall
    tnr = safe_div(total_TN, (total_TN + total_FP)) if (total_TN + total_FP) > 0 else 0.0
    bal_acc = 0.5 * (tpr + tnr)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "bal_acc": bal_acc,
    }


def metrics_kinetic_batch(logits, true_labels, lengths, n_classes=None):
    """
    logits: [B, C, R, L, L, K]
    true_labels: [B, C, R, L, L] (0..K-1)
    We compute:
      - overall accuracy
      - macro precision, macro recall, macro F1 (average over classes)
      - balanced accuracy (mean recall across classes)
    """
    device = true_labels.device
    B, C, R, L, _, K = logits.shape
    preds = logits.argmax(dim=-1)  # [B, C, R, L, L]

    # accumulate per-class counts
    TP_per_class = torch.zeros(K, device=device)
    P_pred_per_class = torch.zeros(K, device=device)
    P_true_per_class = torch.zeros(K, device=device)
    total_correct = 0
    total_count = 0

    for b in range(B):
        Lb = int(lengths[b].item())
        if Lb == 0:
            continue
        true_flat = true_labels[b, :, :, :Lb, :Lb].reshape(-1)
        pred_flat = preds[b, :, :, :Lb, :Lb].reshape(-1)
        total_count += true_flat.numel()
        total_correct += int((true_flat == pred_flat).sum().item())
        for k in range(K):
            TP_k = int(((pred_flat == k) & (true_flat == k)).sum().item())
            P_pred_k = int((pred_flat == k).sum().item())
            P_true_k = int((true_flat == k).sum().item())
            TP_per_class[k] += TP_k
            P_pred_per_class[k] += P_pred_k
            P_true_per_class[k] += P_true_k

    accuracy = safe_div(total_correct, total_count) if total_count > 0 else 0.0

    precisions = []
    recalls = []
    f1s = []
    for k in range(K):
        tp = TP_per_class[k].item()
        pp = P_pred_per_class[k].item()
        pt = P_true_per_class[k].item()
        prec_k = safe_div(tp, pp) if pp > 0 else 0.0
        rec_k = safe_div(tp, pt) if pt > 0 else 0.0
        f1_k = safe_div(2 * prec_k * rec_k, (prec_k + rec_k)) if (prec_k + rec_k) > 0 else 0.0
        precisions.append(prec_k)
        recalls.append(rec_k)
        f1s.append(f1_k)

    macro_precision = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
    macro_recall = float(np.mean(recalls)) if len(recalls) > 0 else 0.0
    macro_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0
    balanced_acc = macro_recall

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "bal_acc": balanced_acc,
    }


def metrics_frequency_batch(pred, true, lengths):
    """
    pred/true: [B, C, L, L]
    Return RMSE computed across valid entries in batch (single scalar).
    """
    device = true.device
    B, C, L, _ = true.shape
    se_sum = 0.0
    count = 0
    for b in range(B):
        Lb = int(lengths[b].item())
        if Lb == 0:
            continue
        diff = (pred[b, :, :Lb, :Lb] - true[b, :, :Lb, :Lb]).reshape(-1)
        se_sum += float((diff ** 2).sum().item())
        count += diff.numel()
    mse = (se_sum / count) if count > 0 else 0.0
    rmse = math.sqrt(mse)
    return {"rmse": rmse}


# ----------------- Training / validation steps -----------------
def build_outputs_and_targets_for_loss(structure, dyn, kin, freq, lengths, loss_heads, device, kin_class_weights):
    """
    Build outputs_for_loss and targets_for_loss dicts keyed by the exact loss names expected
    by loss_mod.esmdynamic_loss (e.g. 'dynamic_logits', 'kinetic_logits', ...).
    Also construct dynamic confidence targets and frequency residual targets when requested.
    """
    outputs = {}
    targets = {}

    # Move label tensors to device
    dyn = dyn.to(device)
    kin = kin.to(device)
    freq = freq.to(device)
    lengths = lengths.to(device)

    # Extract main logits/preds if present in structure
    # Map expected loss keys to structure keys if needed
    # dynamic_logits -> structure['dynamic_logits']
    # dynamic_confidence -> structure['dynamic_confidence']
    # kinetic_logits -> structure['kinetic_logits']
    # kinetic_confidence -> structure['kinetic_confidence'] (or we will compute)
    # frequency_pred -> structure['frequency_pred'] (model uses <name>_pred)
    # frequency_residual_pred -> structure['frequency_residual_pred']

    for h in loss_heads:
        prefix = h.split("_")[0]
        if h == "dynamic_logits":
            # prefer direct structure key
            if "dynamic_logits" in structure:
                outputs[h] = structure["dynamic_logits"]
            else:
                # try alternate names
                if f"{prefix}_logits" in structure:
                    outputs[h] = structure[f"{prefix}_logits"]
            targets[h] = dyn
        elif h == "dynamic_confidence":
            # Model may provide dynamic_confidence under 'dynamic_confidence'
            if f"{prefix}_confidence" in structure:
                outputs[h] = structure[f"{prefix}_confidence"]  # [B, C, L]
            # build target from pair preds (use logits if available)
            # determine pair predictions (logits or prob)
            if f"{prefix}_logits" in structure:
                pred_pairs = structure[f"{prefix}_logits"]  # [B, C, L, L]
            else:
                raise RuntimeError("Cannot compute dynamic confidence target: pair predictions missing in model output.")
            targets[h] = build_confidence_targets_dynamic(pred_pairs.detach(), dyn, lengths)
        elif h == "kinetic_logits":
            if "kinetic_logits" in structure:
                outputs[h] = structure["kinetic_logits"]  # [B, C, R, L, L, K]
            else:
                if f"{prefix}_logits" in structure:
                    outputs[h] = structure[f"{prefix}_logits"]
            targets[h] = kin
        elif h == "kinetic_confidence":
            # Model may provide kinetic_confidence under 'kinetic_confidence' [B,C,L]
            if f"{prefix}_confidence" in structure:
                outputs[h] = structure[f"{prefix}_confidence"]  # [B, C, L]
            # compute conf_target per rate from kinetic logits/preds vs kin labels
            if "kinetic_logits" in structure:
                kin_logits = structure["kinetic_logits"]  # [B,C,R,L,L,K]
            elif f"{prefix}_logits" in structure:
                kin_logits = structure[f"{prefix}_logits"]
            else:
                raise RuntimeError("Cannot compute kinetic confidence target: kinetic logits missing.")
            conf_per_rate = build_confidence_targets_kinetic(kin_logits.detach(), kin, lengths)
            targets[h] = conf_per_rate.mean(dim=2)
        elif h == "frequency_pred":
            # model provides frequency prediction under 'frequency_pred' (or 'frequency_pred' from model.heads)
            key = f"{prefix}_pred"
            if key in structure:
                outputs[h] = structure[key]
            elif f"{prefix}_value" in structure:
                outputs[h] = structure[f"{prefix}_value"]
            else:
                raise RuntimeError("frequency prediction missing from model outputs.")
            targets[h] = freq
        elif h == "frequency_residual_pred":
            # model provides residual prediction under 'frequency_residual_pred'
            key = f"{prefix}_residual_pred"
            if key in structure:
                outputs[h] = structure[key]
            else:
                raise RuntimeError("frequency residual prediction missing from model outputs.")
            # build target as absolute difference between freq_true and freq_pred
            # freq_pred for this calculation should be outputs['frequency_pred'] if available
            freq_pred = None
            if "frequency_pred" in outputs:
                freq_pred = outputs["frequency_pred"]
            else:
                # try to find in structure
                if f"{prefix}_pred" in structure:
                    freq_pred = structure[f"{prefix}_pred"]
                elif f"{prefix}_value" in structure:
                    freq_pred = structure[f"{prefix}_value"]
            if freq_pred is None:
                raise RuntimeError("Cannot construct frequency_residual target: frequency_pred not found.")
            targets[h] = build_frequency_residual_target(freq_pred.detach(), freq)
        else:
            raise RuntimeError(f"Unsupported loss head requested: {h}")

    return outputs, targets


def train_one_epoch(
    training_loader,
    optimizer,
    model,
    epoch,
    writer,
    device="cuda",
    batch_accum=1,
    alpha=0.25,
    gamma=2,
    outpath="./",
    timestamp="",
    loss_heads=None,
    kin_class_weights=None,
):
    model.train()
    running_loss = 0.0
    loss_norm = 0
    # accumulate metrics per epoch for logging
    epoch_metrics = {h: [] for h in loss_heads}

    autocast_enabled = (device == "cuda")

    for i, data in enumerate(training_loader):
        sequences, dyn, kin, freq, lengths = data
        # forward
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast_enabled):
            structure = model.forward_from_seq(sequences)

        # Build outputs & targets for requested loss heads
        outputs_for_loss, targets_for_loss = build_outputs_and_targets_for_loss(
            structure, dyn, kin, freq, lengths, loss_heads, device, kin_class_weights
        )

        # ensure kin_class_weights in local device and correct dtype
        kin_weights = None
        if kin_class_weights is not None:
            for h in loss_heads:
                if "kinetic_logits" in h and h in outputs_for_loss:
                    logits_dtype = outputs_for_loss[h].dtype   # usually bfloat16 under autocast
                    kin_weights = kin_class_weights.to(device=device, dtype=logits_dtype)
                    break

        loss = loss_mod.esmdynamic_loss(outputs_for_loss, targets_for_loss, lengths.to(device), active_heads=loss_heads, kin_class_weights=kin_weights, alpha=alpha, gamma=gamma)
        loss = loss / batch_accum
        loss.backward()

        if ((i + 1) % batch_accum == 0) or (i + 1 == len(training_loader)):
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            loss_norm += 1

            # compute and log metrics for each requested head
            # dynamic metrics
            if "dynamic_logits" in loss_heads and "dynamic_logits" in outputs_for_loss:
                dyn_metrics = metrics_dynamic_batch(outputs_for_loss["dynamic_logits"], targets_for_loss.get("dynamic_logits"), lengths)
                epoch_metrics["dynamic_logits"].append(dyn_metrics)
                # Log to tensorboard
                step = epoch * len(training_loader) + i
                writer.add_scalar("dynamic/loss_train_batch", loss.item(), step)
                writer.add_scalar("dynamic/accuracy/train", dyn_metrics["accuracy"], step)
                writer.add_scalar("dynamic/precision/train", dyn_metrics["precision"], step)
                writer.add_scalar("dynamic/recall/train", dyn_metrics["recall"], step)
                writer.add_scalar("dynamic/f1/train", dyn_metrics["f1"], step)
                writer.add_scalar("dynamic/bal_acc/train", dyn_metrics["bal_acc"], step)

            # kinetic metrics
            if "kinetic_logits" in loss_heads and "kinetic_logits" in outputs_for_loss:
                # infer K from logits shape
                kin_logits = outputs_for_loss["kinetic_logits"]
                K = kin_logits.shape[-1]
                kin_metrics = metrics_kinetic_batch(kin_logits, targets_for_loss.get("kinetic_logits"), lengths, n_classes=K)
                epoch_metrics["kinetic_logits"].append(kin_metrics)
                step = epoch * len(training_loader) + i
                writer.add_scalar("kinetic/loss_train_batch", loss.item(), step) # Repeated value
                writer.add_scalar("kinetic/accuracy/train", kin_metrics["accuracy"], step)
                writer.add_scalar("kinetic/macro_precision/train", kin_metrics["macro_precision"], step)
                writer.add_scalar("kinetic/macro_recall/train", kin_metrics["macro_recall"], step)
                writer.add_scalar("kinetic/macro_f1/train", kin_metrics["macro_f1"], step)
                writer.add_scalar("kinetic/bal_acc/train", kin_metrics["bal_acc"], step)

            # frequency metrics
            if "frequency_pred" in loss_heads and "frequency_pred" in outputs_for_loss:
                freq_metrics = metrics_frequency_batch(outputs_for_loss["frequency_pred"], targets_for_loss.get("frequency_pred"), lengths)
                epoch_metrics["frequency_pred"].append(freq_metrics)
                step = epoch * len(training_loader) + i
                writer.add_scalar("frequency/loss_training_batch", loss.item(), step) # Repeated value
                writer.add_scalar("frequency/rmse/train", freq_metrics["rmse"], step)

            print(f"[Train] Epoch {epoch+1} batch {i+1}/{len(training_loader)} loss {loss.item():.6f}")

    avg_loss = (running_loss / loss_norm) if loss_norm > 0 else 0.0

    # Average epoch metrics (mean of per-batch metrics)
    aggregated_metrics = {}
    for h in loss_heads:
        list_metrics = epoch_metrics.get(h, [])
        if not list_metrics:
            aggregated_metrics[h] = {}
            continue
        # each item is a dict -> compute mean per-key
        keys = list_metrics[0].keys()
        agg = {}
        for k in keys:
            vals = [m[k] for m in list_metrics]
            agg[k] = float(np.mean(vals))
        aggregated_metrics[h] = agg

    return avg_loss, aggregated_metrics


def compute_validation(
    validation_loader,
    model,
    epoch_number,
    writer,
    training_loss,
    device="cuda",
    alpha=0.25,
    gamma=2,
    loss_heads=None,
    kin_class_weights=None,
):
    model.eval()
    running_vloss = 0.0
    val_batches = 0
    epoch_metrics = {h: [] for h in loss_heads}

    autocast_enabled = (device == "cuda")

    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast_enabled):
        for i, data in enumerate(validation_loader):
            sequences, dyn, kin, freq, lengths = data
            structure = model.forward_from_seq(sequences)

            outputs_for_loss, targets_for_loss = build_outputs_and_targets_for_loss(
                structure, dyn, kin, freq, lengths, loss_heads, model.device, kin_class_weights
            )

             # ensure kin_class_weights in local device and correct dtype
            kin_weights = None
            if kin_class_weights is not None:
                for h in loss_heads:
                    if "kinetic_logits" in h and h in outputs_for_loss:
                        logits_dtype = outputs_for_loss[h].dtype   # usually bfloat16 under autocast
                        kin_weights = kin_class_weights.to(device=device, dtype=logits_dtype)
                        break

            vloss = loss_mod.esmdynamic_loss(outputs_for_loss, targets_for_loss, lengths.to(model.device), active_heads=loss_heads, kin_class_weights=kin_weights, alpha=alpha, gamma=gamma)
            running_vloss += vloss.item()
            val_batches += 1

            # metrics same as train
            if "dynamic_logits" in loss_heads and "dynamic_logits" in outputs_for_loss:
                dyn_metrics = metrics_dynamic_batch(outputs_for_loss["dynamic_logits"], targets_for_loss.get("dynamic_logits"), lengths)
                epoch_metrics["dynamic_logits"].append(dyn_metrics)
            if "kinetic_logits" in loss_heads and "kinetic_logits" in outputs_for_loss:
                K = outputs_for_loss["kinetic_logits"].shape[-1]
                kin_metrics = metrics_kinetic_batch(outputs_for_loss["kinetic_logits"], targets_for_loss.get("kinetic_logits"), lengths, n_classes=K)
                epoch_metrics["kinetic_logits"].append(kin_metrics)
            if "frequency_pred" in loss_heads and "frequency_pred" in outputs_for_loss:
                freq_metrics = metrics_frequency_batch(outputs_for_loss["frequency_pred"], targets_for_loss.get("frequency_pred"), lengths)
                epoch_metrics["frequency_pred"].append(freq_metrics)

    avg_vloss = running_vloss / (val_batches if val_batches > 0 else 1)

    # Aggregate metrics
    aggregated_metrics = {}
    for h in loss_heads:
        list_metrics = epoch_metrics.get(h, [])
        if not list_metrics:
            aggregated_metrics[h] = {}
            continue
        keys = list_metrics[0].keys()
        agg = {}
        for k in keys:
            vals = [m[k] for m in list_metrics]
            agg[k] = float(np.mean(vals))
            # log to tensorboard
            writer.add_scalar(f"{h}/{k}/val", agg[k], epoch_number + 1)
        aggregated_metrics[h] = agg

    # also log combined train vs val loss
    writer.add_scalars('Training vs. Validation Loss', {'Training': training_loss, 'Validation': avg_vloss}, epoch_number + 1)
    print(f"[Val] Epoch {epoch_number+1} train_loss {training_loss:.6f} val_loss {avg_vloss:.6f}")

    return avg_vloss, aggregated_metrics


def save_run_metadata(outpath, args, timestamp):
    metadata_file = os.path.join(outpath, f"run_metadata_{timestamp}.txt")
    script_path = os.path.realpath(__file__)
    with open(metadata_file, "w") as f:
        f.write(f"Run Timestamp: {timestamp}\n")
        f.write(f"Executed Script: {script_path}\n")
        f.write("Parameters:\n")
        for arg, value in vars(args).items():
            f.write(f"  {arg}: {value}\n")
    print(f"Run metadata saved to: {metadata_file}")


def get_args():
    import shlex

    def parse_list(arg):
        return shlex.split(arg.replace(",", " "))

    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--train_identifiers_file", type=str, required=True)
    parser.add_argument("--val_identifiers_file", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--batch_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_samples_per_epoch", type=int, default=10000)
    parser.add_argument("--val_samples_per_epoch", type=int, default=1000)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss_heads", type=parse_list, default=[], required=True, help="Loss heads to use (e.g. dynamic_logits kinetic_logits frequency_pred). Must match keys used by loss.esmdynamic_loss.")
    parser.add_argument("--kin_class_weights", type=str, default=None, help="Optional path to torch-saved kinetics class weights (tensor shape [2,K])")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.outpath, exist_ok=True)

    training_set, validation_set = init_datasets(
        args.train_identifiers_file,
        args.val_identifiers_file,
        args.data_dir,
        crop_length=256,
    )

    training_loader, validation_loader = init_data_loaders(
        training_set,
        validation_set,
        batch_size=args.batch_size,
        train_samples_per_epoch=args.train_samples_per_epoch,
        val_samples_per_epoch=args.val_samples_per_epoch,
    )

    loss_heads = args.loss_heads
    prefixes = sorted(list(select_prefixes_from_loss_heads(loss_heads)))
    print("Requested loss heads:", loss_heads)
    print("Will load model heads (prefixes):", prefixes)

    kin_class_weights = None
    if args.kin_class_weights:
        kin_class_weights = torch.load(args.kin_class_weights, map_location="cpu")

    model = init_model(chunk_size=args.chunk_size, device=args.device, pretrained=args.pretrained, heads_to_load=prefixes)

    # Freeze trunk
    if hasattr(model, "esmfold") and model.load_esmfold:
        model.esmfold.requires_grad_(False)

    optimizer = init_optimizer_for_heads(model, prefixes, lr=args.lr)

    timestamp, writer = init_writer(args.outpath)
    save_run_metadata(args.outpath, args, timestamp)

    best_vloss = float("inf")
    best_saved = False

    for epoch in range(args.epochs):
        print("EPOCH", epoch + 1)
        train_loss, train_metrics = train_one_epoch(
            training_loader,
            optimizer,
            model,
            epoch,
            writer,
            device=args.device,
            batch_accum=args.batch_accum,
            alpha=args.alpha,
            gamma=args.gamma,
            outpath=args.outpath,
            timestamp=timestamp,
            loss_heads=loss_heads,
            kin_class_weights=(kin_class_weights if kin_class_weights is None else kin_class_weights.to(args.device)),
        )

        val_loss, val_metrics = compute_validation(
            validation_loader,
            model,
            epoch,
            writer,
            train_loss,
            device=args.device,
            alpha=args.alpha,
            gamma=args.gamma,
            loss_heads=loss_heads,
            kin_class_weights=(kin_class_weights if kin_class_weights is None else kin_class_weights.to(args.device)),
        )

        # Save best and last for each trained head
        if val_loss < best_vloss:
            best_vloss = val_loss
            save_head_state_dicts(model, prefixes, args.outpath, "best_vloss", timestamp)
            best_saved = True

        save_head_state_dicts(model, prefixes, args.outpath, "chkpt", timestamp)

    print("Training finished. Best validation loss:", best_vloss)
    if not best_saved:
        print("No improvement observed during training; only last checkpoint(s) saved.")


if __name__ == "__main__":
    main()

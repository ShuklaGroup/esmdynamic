#!/usr/bin/env python3
"""
New training script for ESMDynamic (works with the uploaded model/loss/dataset).
Saves only trained heads (best val loss and last checkpoint).
"""

import os
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Use uploaded modules (paths provided by user)
from /mnt/data import esmdynamic as esmdynamic_mod  # path to uploaded esmdynamic.py
from /mnt/data import loss as loss_mod            # path to uploaded loss.py
from /mnt/data import data_reader as dr_mod       # path to uploaded data_reader.py

# NOTE:
# The developer environment will transform the local import path to a usable url.
# If you place this file in the same package/directory, change imports accordingly:
#   from esmdynamic import ESMDynamic
#   from loss import esmdynamic_loss
#   from data_reader import DynContactDataset


def init_datasets(
    training_identifiers_file,
    validation_identifiers_file,
    data_dir,
    crop_length=256,
    training_weight_file=None,
    validation_weight_file=None
):
    cluster_train = list(np.loadtxt(training_identifiers_file, dtype=str))
    cluster_val = list(np.loadtxt(validation_identifiers_file, dtype=str))

    training_weights = torch.load(training_weight_file) if training_weight_file is not None else None
    validation_weights = torch.load(validation_weight_file) if validation_weight_file is not None else None

    training_set = dr_mod.DynContactDataset(
        data_dir=data_dir,
        identifiers=cluster_train,
        crop_length=crop_length,
    )
    # optionally set precomputed weights
    if training_weights is not None:
        training_set.weights = training_weights

    validation_set = dr_mod.DynContactDataset(
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


def select_prefixes_from_loss_heads(loss_heads):
    """
    Given loss head strings like ['dynamic_logits', 'kinetic_logits', 'frequency_pred']
    return unique set of head prefixes: {'dynamic','kinetic','frequency'}
    """
    prefixes = set()
    for h in loss_heads:
        if "_" in h:
            prefixes.add(h.split("_")[0])
        else:
            prefixes.add(h)
    return prefixes


def init_model(chunk_size=256, device="cuda", pretrained=None, heads_to_load=None):
    # heads_to_load is a list/iterable of strings like ['dynamic','kinetic']
    model = esmdynamic_mod.ESMDynamic(load_esmfold=True, heads_to_load=heads_to_load)
    if pretrained:
        # load state dict (non-strict to allow partial keys)
        sd = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(sd, strict=False)
    if device == "cuda":
        model.cuda()
    model.set_chunk_size(chunk_size)
    return model


def init_optimizer_for_heads(model, head_prefixes, lr=1e-4):
    # Only optimize parameters for selected heads
    params = []
    for h in head_prefixes:
        if h not in model.heads:
            raise RuntimeError(f"Head '{h}' not found in model.heads. Available: {list(model.heads.keys())}")
        params += list(model.heads[h].parameters())
    if len(params) == 0:
        raise RuntimeError("No parameters selected for optimization. Check selected heads.")
    optimizer = torch.optim.Adam(params, lr=lr)
    return optimizer


def init_writer(outpath):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(outpath, "runs", f"trainer_{timestamp}"))
    return timestamp, writer


def save_head_state_dicts(model, head_prefixes, outpath, prefix_label, timestamp):
    """
    Save each head's state_dict to a separate file.
    prefix_label should be 'best_vloss' or 'chkpt' to differentiate names.
    """
    for h in sorted(head_prefixes):
        fname = os.path.join(outpath, f"{h}_head_{prefix_label}_{timestamp}.pt")
        torch.save(model.heads[h].state_dict(), fname)
        print(f"Saved head '{h}' -> {fname}")


# --- Simple metrics implementation (conservative; primarily for binary dynamic_logits) ---
def compute_basic_metrics(outputs_for_loss, targets_for_loss, lengths, active_loss_heads):
    """
    Returns (acc, bal_acc, tpr, prec, f1) aggregated across batch.
    - For dynamic_logits (binary) we compute thresholded predictions and compute metrics.
    - For kinetic_logits (multiclass) we compute simple accuracy (per-pair) averaged; other metrics set to 0.
    - If multiple heads active, return metrics from the first available head with meaningful metric (prefers dynamic).
    """
    device = lengths.device
    B = lengths.shape[0]

    # default zeros
    acc = bal_acc = tpr = prec = f1 = 0.0

    # Prefer dynamic_logits metrics
    if "dynamic_logits" in active_loss_heads and "dynamic_logits" in outputs_for_loss and "dynamic_logits" in targets_for_loss:
        logits = outputs_for_loss["dynamic_logits"]  # [B, C, L, L]
        target = targets_for_loss["dynamic_logits"]  # same
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).long()
        # mask per-sample
        total_correct = 0
        total_count = 0
        # We'll compute overall accuracy only for valid residues (simple but meaningful)
        for i in range(B):
            L = int(lengths[i].item())
            t = target[i, :, :L, :L].reshape(-1)
            p = pred[i, :, :L, :L].reshape(-1)
            total_correct += (t == p).sum().item()
            total_count += t.numel()
        acc = (total_correct / total_count) if total_count > 0 else 0.0
        # set other metrics to acc for now (a coarse fallback)
        bal_acc = tpr = prec = f1 = acc
        return acc, bal_acc, tpr, prec, f1

    # Next prefer kinetic logits accuracy
    if "kinetic_logits" in active_loss_heads and "kinetic_logits" in outputs_for_loss and "kinetic_logits" in targets_for_loss:
        logits = outputs_for_loss["kinetic_logits"]  # [B, C, 2, L, L, K]
        target = targets_for_loss["kinetic_logits"]  # [B, C, 2, L, L]
        # compute argmax over classes
        # flatten mask similarly
        total_correct = 0
        total_count = 0
        B, C, R, L1, L2, K = logits.shape
        preds = logits.argmax(dim=-1)  # [B, C, 2, L, L]
        for i in range(B):
            L = int(lengths[i].item())
            t = target[i, :, :, :L, :L].reshape(-1)
            p = preds[i, :, :, :L, :L].reshape(-1)
            total_correct += (t == p).sum().item()
            total_count += t.numel()
        acc = (total_correct / total_count) if total_count > 0 else 0.0
        bal_acc = tpr = prec = f1 = acc
        return acc, bal_acc, tpr, prec, f1

    # Fallback zeros
    return 0.0, 0.0, 0.0, 0.0, 0.0


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
    metrics_accum = torch.zeros(5, device="cpu")

    autocast_enabled = (device == "cuda")

    for i, data in enumerate(training_loader):
        sequences, dyn, kin, freq, lengths = data
        # lengths is CPU tensor -> move to device for loss function usage
        lengths = lengths.to(device)

        # forward
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast_enabled):
            structure = model.forward_from_seq(sequences)

        # Build outputs_for_loss as dict keyed by exactly the loss names used by loss_mod
        outputs_for_loss = {}
        # model stores per-head logits/prob/.. under "<head>_logits" or "<head>_prob" etc.
        # We'll try to extract logits/pred/residual/ confidence keys expected by loss functions:
        # dynamic_logits, dynamic_confidence, kinetic_logits, kinetic_confidence, frequency_pred, frequency_residual_pred
        for h in loss_heads:
            # h is e.g. "dynamic_logits"
            if h in structure:
                outputs_for_loss[h] = structure[h]
            else:
                # try common alternatives using naming in ESMDynamic:
                prefix = h.split("_")[0]
                if h.endswith("_logits"):
                    key = f"{prefix}_logits"
                elif h.endswith("_confidence"):
                    key = f"{prefix}_confidence"
                elif h.endswith("_pred") and prefix == "frequency":
                    key = f"{prefix}_pred"
                elif h.endswith("_residual_pred"):
                    key = f"{prefix}_residual_pred"
                else:
                    key = h
                if key in structure:
                    outputs_for_loss[h] = structure[key]
                else:
                    # Not present - leave missing; will error later if target missing
                    pass

        # Build targets for the requested loss heads
        targets_for_loss = {}
        if "dynamic_logits" in loss_heads:
            targets_for_loss["dynamic_logits"] = dyn.to(device)  # [B, 5, L, L]
        if "dynamic_confidence" in loss_heads:
            raise RuntimeError("dynamic_confidence target not provided by dataset. Remove from --loss_heads or add its target.")
        if "kinetic_logits" in loss_heads:
            targets_for_loss["kinetic_logits"] = kin.to(device)  # [B,5,2,L,L]
        if "kinetic_confidence" in loss_heads:
            raise RuntimeError("kinetic_confidence target not provided by dataset. Remove from --loss_heads or add its target.")
        if "frequency_pred" in loss_heads:
            targets_for_loss["frequency_pred"] = freq.to(device)  # [B,5,L,L]
        if "frequency_residual_pred" in loss_heads:
            raise RuntimeError("frequency_residual_pred target not provided by dataset. Remove from --loss_heads or add its target.")

        # Compute loss
        # Prepare kin_class_weights tensor if necessary
        kin_weights_tensor = kin_class_weights
        if kin_weights_tensor is None and any(["kinetic_logits" == lh for lh in loss_heads]):
            # default: ones with K=6 (as in model definition)
            kin_weights_tensor = torch.ones((2, 6), device=device)

        loss = loss_mod.esmdynamic_loss(outputs_for_loss, targets_for_loss, lengths, active_heads=loss_heads, kin_class_weights=kin_weights_tensor, alpha=alpha, gamma=gamma)
        loss = loss / batch_accum
        loss.backward()

        if ((i + 1) % batch_accum == 0) or (i + 1 == len(training_loader)):
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            loss_norm += 1

            # compute simple metrics
            metrics = compute_basic_metrics(outputs_for_loss, targets_for_loss, lengths, loss_heads)
            metrics_accum += torch.tensor(metrics, device="cpu")

            # logging for batch
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(training_loader) + i)
            acc_avg = (metrics_accum / loss_norm)[0].item()
            writer.add_scalar('(DynCont) Accuracy/train', acc_avg, epoch * len(training_loader) + i)
            print(f"Epoch {epoch+1} batch {i+1}/{len(training_loader)} loss (accum): {loss.item():.6f}")

    avg_loss = (running_loss / loss_norm) if loss_norm > 0 else 0.0
    # finalize metrics
    final_metrics = (metrics_accum / (loss_norm if loss_norm > 0 else 1)).tolist()
    return avg_loss, final_metrics


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
    kin_class_weights=None
):
    model.eval()
    running_vloss = 0.0
    with torch.no_grad():
        metrics_accum = torch.zeros(5, device="cpu")
        count = 0
        for i, data in enumerate(validation_loader):
            sequences, dyn, kin, freq, lengths = data
            lengths = lengths.to(device)

            structure = model.forward_from_seq(sequences)

            outputs_for_loss = {}
            if "dynamic_logits" in loss_heads and f"dynamic_logits" in structure:
                outputs_for_loss["dynamic_logits"] = structure["dynamic_logits"]
            if "kinetic_logits" in loss_heads and f"kinetic_logits" in structure:
                outputs_for_loss["kinetic_logits"] = structure["kinetic_logits"]
            if "frequency_pred" in loss_heads and f"frequency_pred" in structure:
                outputs_for_loss["frequency_pred"] = structure["frequency_pred"]
            # Build targets similarly
            targets_for_loss = {}
            if "dynamic_logits" in loss_heads:
                targets_for_loss["dynamic_logits"] = dyn.to(device)
            if "kinetic_logits" in loss_heads:
                targets_for_loss["kinetic_logits"] = kin.to(device)
            if "frequency_pred" in loss_heads:
                targets_for_loss["frequency_pred"] = freq.to(device)

            kin_weights_tensor = kin_class_weights
            if kin_weights_tensor is None and any([lh == "kinetic_logits" for lh in loss_heads]):
                kin_weights_tensor = torch.ones((2, 6), device=device)

            vloss = loss_mod.esmdynamic_loss(outputs_for_loss, targets_for_loss, lengths, active_heads=loss_heads, kin_class_weights=kin_weights_tensor, alpha=alpha, gamma=gamma)
            running_vloss += vloss.item()
            metrics = compute_basic_metrics(outputs_for_loss, targets_for_loss, lengths, loss_heads)
            metrics_accum += torch.tensor(metrics, device="cpu")
            count += 1

    avg_vloss = running_vloss / (count if count > 0 else 1)
    avg_metrics = (metrics_accum / (count if count > 0 else 1)).tolist()

    # Logging
    writer.add_scalars('Training vs. Validation Loss', {'Training': training_loss, 'Validation': avg_vloss}, epoch_number + 1)
    writer.add_scalar('(DynCont) Accuracy/val', avg_metrics[0], epoch_number + 1)
    print(f"Validation: epoch {epoch_number+1} train_loss {training_loss:.6f} val_loss {avg_vloss:.6f}")

    return avg_vloss, avg_metrics


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
    parser.add_argument("--loss_heads", nargs="+", required=True, help="Loss heads to use (e.g. dynamic_logits kinetic_logits frequency_pred). These names must match those expected by the loss wrapper.")
    parser.add_argument("--kin_class_weights", type=str, default=None, help="Optional path to torch-saved kernel weights for kinetics (shape [2,K])")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.outpath, exist_ok=True)

    # Datasets
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

    # Loss heads and prefixes
    loss_heads = args.loss_heads
    prefixes = sorted(list(select_prefixes_from_loss_heads(loss_heads)))
    print("Requested loss heads:", loss_heads)
    print("Will load model heads (prefixes):", prefixes)

    # Kinetic class weights (optional)
    kin_class_weights = None
    if args.kin_class_weights:
        kin_class_weights = torch.load(args.kin_class_weights, map_location="cpu")
        # ensure on device later

    # Initialize model (load only required heads)
    model = init_model(chunk_size=args.chunk_size, device=args.device, pretrained=args.pretrained, heads_to_load=prefixes)

    # Freeze trunk (should already be frozen by ESMDynamic init) and ensure heads are trainable
    if hasattr(model, "esmfold") and model.load_esmfold:
        model.esmfold.requires_grad_(False)

    # Optimizer over only selected heads
    optimizer = init_optimizer_for_heads(model, prefixes, lr=args.lr)

    # Writer
    timestamp, writer = init_writer(args.outpath)
    save_run_metadata(args.outpath, args, timestamp)

    # Training loop
    best_vloss = float("inf")
    best_saved = False

    # Create placeholder for last checkpoint filenames to allow overwriting easily
    last_prefix = "chkpt"
    best_prefix = "best_vloss"

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
            kin_class_weights=(kin_class_weights.to(args.device) if isinstance(kin_class_weights, torch.Tensor) else kin_class_weights)
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
            kin_class_weights=(kin_class_weights.to(args.device) if isinstance(kin_class_weights, torch.Tensor) else kin_class_weights)
        )

        # Save best and last for each trained head
        # best
        if val_loss < best_vloss:
            best_vloss = val_loss
            save_head_state_dicts(model, prefixes, args.outpath, best_prefix, timestamp)
            best_saved = True

        # always save last (overwrite older last)
        save_head_state_dicts(model, prefixes, args.outpath, last_prefix, timestamp)

    print("Training finished. Best validation loss:", best_vloss)
    if not best_saved:
        print("No improvement observed during training; only last checkpoint(s) saved.")

if __name__ == "__main__":
    main()

import argparse
import csv
import os
import re
from pathlib import Path

import esm
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
from Bio import SeqIO


TEMPERATURES = [320, 348, 379, 413, 450]

KINETIC_ON_CLASS_NAMES = [
    "always_on",
    "1to10ns",
    "10to100ns",
    "100to300ns",
    "gt300ns",
    "never_on",
]

KINETIC_OFF_CLASS_NAMES = [
    "always_off",
    "1to10ns",
    "10to100ns",
    "100to300ns",
    "gt300ns",
    "never_off",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict dynamic contacts, frequency, and kinetics using ESMDynamic."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", type=str, help="Single sequence string.")
    group.add_argument("--fasta", type=str, help="Path to FASTA file with sequences.")
    group.add_argument(
        "--csv",
        type=str,
        help="CSV file with sequences (first column ID, second column sequence).",
    )

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--chunk_size", type=int, default=256, help="Model chunk size.")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to use.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--chain_ids",
        type=str,
        default=None,
        help="Chain IDs to use for labels (e.g. ABCDEF). Default: A-Z.",
    )
    parser.add_argument(
        "--low_memory",
        action="store_true",
        help="Use low-memory inference mode.",
    )
    parser.add_argument(
        "--save_html",
        action="store_true",
        help="Also save interactive HTML heatmaps.",
    )
    parser.add_argument(
        "--save_png",
        action="store_true",
        help="Save PNG heatmaps/plots.",
    )
    parser.add_argument(
        "--save_txt",
        action="store_true",
        help="Save text/CSV outputs.",
    )
    parser.add_argument(
        "--save_raw_pt",
        action="store_true",
        help="Save a .pt bundle with all cropped outputs for each sequence.",
    )
    parser.add_argument(
        "--num_recycles",
        type=int,
        default=None,
        help="Optional number of recycles to pass to the model.",
    )

    args = parser.parse_args()

    # Match original behavior reasonably closely if user does not specify any format flags.
    if not (args.save_html or args.save_png or args.save_txt or args.save_raw_pt):
        args.save_html = True
        args.save_png = True
        args.save_txt = True
        args.save_raw_pt = True

    return args


def load_sequences(args):
    sequences = []
    if args.sequence:
        sequences.append(("output", args.sequence))
    elif args.fasta:
        for record in SeqIO.parse(args.fasta, "fasta"):
            sequences.append((record.id, str(record.seq)))
    elif args.csv:
        with open(args.csv, newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    sequences.append((row[0], row[1]))
    return sequences


def sanitize_id(seq_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", seq_id)


def get_crop_mask_labels_and_boundaries(sequence, insert_len=25, chain_ids=None):
    """
    Build:
      - mask over the model output positions (True for real residues, False for linker)
      - per-residue labels after cropping
      - chain boundary positions in cropped coordinates
    """
    if chain_ids is None:
        chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    chains = sequence.split(":")
    mask = []
    labels = []
    boundaries = []

    cropped_len_so_far = 0
    for chain_idx, chain_seq in enumerate(chains):
        chain_id = chain_ids[chain_idx % len(chain_ids)]

        for res_counter, aa in enumerate(chain_seq, start=1):
            mask.append(True)
            labels.append(f"{chain_id}-{aa}{res_counter}")

        cropped_len_so_far += len(chain_seq)

        # Add a boundary after each chain except the last.
        if chain_idx < len(chains) - 1:
            boundaries.append(cropped_len_so_far)
            mask.extend([False] * insert_len)

    return np.array(mask, dtype=bool), labels, boundaries


def crop_pair_matrix(matrix, mask):
    idx = np.where(mask)[0]
    return matrix[np.ix_(idx, idx)]


def crop_residue_vector(vector, mask):
    idx = np.where(mask)[0]
    return vector[idx]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_matrix_txt(path, matrix, fmt="%.6f"):
    np.savetxt(path, matrix, fmt=fmt)


def save_vector_csv(path, labels, values):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["residue", "value"])
        for label, val in zip(labels, values):
            writer.writerow([label, float(val)])


def save_heatmap_png(path, matrix, title, boundaries=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="viridis", origin="upper", vmin=vmin, vmax=vmax)
    if boundaries:
        for pos in boundaries:
            ax.axhline(pos - 0.5, color="white", linewidth=1)
            ax.axvline(pos - 0.5, color="white", linewidth=1)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_vector_png(path, labels, values, title, ylabel):
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(values))
    ax.plot(x, values)
    ax.set_title(title)
    ax.set_xlabel("Residue index")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, len(values) - 1 if len(values) > 0 else 1)

    # Keep x labels sparse to avoid unreadable plots.
    if len(labels) <= 40:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
    else:
        step = max(1, len(labels) // 20)
        ticks = x[::step]
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_heatmap_html(path, matrix, title, labels, boundaries=None, zmin=None, zmax=None):
    fig = px.imshow(
        matrix,
        x=labels,
        y=labels,
        color_continuous_scale="Viridis",
        labels=dict(x="Residue", y="Residue", color="Value"),
        zmin=zmin,
        zmax=zmax,
    )
    fig.update_layout(title=title)

    if boundaries:
        n = len(labels)
        for pos in boundaries:
            fig.add_shape(
                type="line",
                x0=pos - 0.5,
                x1=pos - 0.5,
                y0=-0.5,
                y1=n - 0.5,
                line=dict(color="white", width=1),
            )
            fig.add_shape(
                type="line",
                y0=pos - 0.5,
                y1=pos - 0.5,
                x0=-0.5,
                x1=n - 0.5,
                line=dict(color="white", width=1),
            )

    fig.write_html(path)


def save_pair_output(
    base_path,
    matrix,
    title,
    labels,
    boundaries,
    args,
    txt_fmt="%.6f",
    png_vmin=None,
    png_vmax=None,
    html_zmin=None,
    html_zmax=None,
):
    if args.save_txt:
        save_matrix_txt(f"{base_path}.txt", matrix, fmt=txt_fmt)
    if args.save_png:
        save_heatmap_png(
            f"{base_path}.png",
            matrix,
            title=title,
            boundaries=boundaries,
            vmin=png_vmin,
            vmax=png_vmax,
        )
    if args.save_html:
        save_heatmap_html(
            f"{base_path}.html",
            matrix,
            title=title,
            labels=labels,
            boundaries=boundaries,
            zmin=html_zmin,
            zmax=html_zmax,
        )


def save_vector_output(base_path, labels, values, title, ylabel, args):
    if args.save_txt:
        save_vector_csv(f"{base_path}.csv", labels, values)
    if args.save_png:
        save_vector_png(
            f"{base_path}.png",
            labels=labels,
            values=values,
            title=title,
            ylabel=ylabel,
        )


def save_kinetics_probabilities_npz(base_path, prob_stack, class_names):
    """
    prob_stack shape: [L, L, n_classes]
    """
    payload = {class_name: prob_stack[:, :, i] for i, class_name in enumerate(class_names)}
    np.savez_compressed(f"{base_path}.npz", **payload)


def save_outputs_for_sequence(output_dir, seq_id, sequence, prediction, seq_idx, args):
    safe_id = sanitize_id(seq_id)
    sample_dir = os.path.join(output_dir, safe_id)
    ensure_dir(sample_dir)

    mask, labels, boundaries = get_crop_mask_labels_and_boundaries(
        sequence,
        insert_len=25,
        chain_ids=args.chain_ids,
    )

    raw_bundle = {
        "sequence": sequence,
        "labels": labels,
        "boundaries": boundaries,
    }

    # -------------------------
    # Dynamic head
    # -------------------------
    if "dynamic_prob" in prediction:
        dynamic_prob = prediction["dynamic_prob"][seq_idx].detach().cpu().numpy()          # [T, L, L]
        dynamic_pred = prediction["dynamic_pred"][seq_idx].detach().cpu().numpy()          # [T, L, L]
        dynamic_conf = prediction["dynamic_confidence"][seq_idx].detach().cpu().numpy()    # [T, L]

        dynamic_dir = os.path.join(sample_dir, "dynamic")
        ensure_dir(dynamic_dir)

        raw_bundle["dynamic_prob"] = []
        raw_bundle["dynamic_pred"] = []
        raw_bundle["dynamic_confidence"] = []

        for t_idx, temp in enumerate(TEMPERATURES):
            prob_map = crop_pair_matrix(dynamic_prob[t_idx], mask)
            pred_map = crop_pair_matrix(dynamic_pred[t_idx], mask)
            conf_vec = crop_residue_vector(dynamic_conf[t_idx], mask)

            raw_bundle["dynamic_prob"].append(prob_map)
            raw_bundle["dynamic_pred"].append(pred_map)
            raw_bundle["dynamic_confidence"].append(conf_vec)

            save_pair_output(
                base_path=os.path.join(dynamic_dir, f"{safe_id}_dynamic_prob_{temp}K"),
                matrix=prob_map,
                title=f"{seq_id} dynamic contact probability ({temp} K)",
                labels=labels,
                boundaries=boundaries,
                args=args,
                txt_fmt="%.6f",
                png_vmin=0.0,
                png_vmax=1.0,
                html_zmin=0.0,
                html_zmax=1.0,
            )

            save_pair_output(
                base_path=os.path.join(dynamic_dir, f"{safe_id}_dynamic_pred_{temp}K"),
                matrix=pred_map.astype(np.int64),
                title=f"{seq_id} dynamic contact prediction ({temp} K)",
                labels=labels,
                boundaries=boundaries,
                args=args,
                txt_fmt="%d",
                png_vmin=0,
                png_vmax=1,
                html_zmin=0,
                html_zmax=1,
            )

            save_vector_output(
                base_path=os.path.join(dynamic_dir, f"{safe_id}_dynamic_confidence_{temp}K"),
                labels=labels,
                values=conf_vec,
                title=f"{seq_id} dynamic confidence ({temp} K)",
                ylabel="Confidence",
                args=args,
            )

    # -------------------------
    # Frequency head
    # -------------------------
    if "frequency_pred" in prediction:
        frequency_pred = prediction["frequency_pred"][seq_idx].detach().cpu().numpy()              # [T, L, L]
        frequency_residual = prediction["frequency_residual_pred"][seq_idx].detach().cpu().numpy() # [T, L, L]

        frequency_dir = os.path.join(sample_dir, "frequency")
        ensure_dir(frequency_dir)

        raw_bundle["frequency_pred"] = []
        raw_bundle["frequency_residual_pred"] = []

        for t_idx, temp in enumerate(TEMPERATURES):
            occ_map = crop_pair_matrix(frequency_pred[t_idx], mask)
            err_map = crop_pair_matrix(frequency_residual[t_idx], mask)

            raw_bundle["frequency_pred"].append(occ_map)
            raw_bundle["frequency_residual_pred"].append(err_map)

            save_pair_output(
                base_path=os.path.join(frequency_dir, f"{safe_id}_frequency_pred_{temp}K"),
                matrix=occ_map,
                title=f"{seq_id} contact frequency / occupancy ({temp} K)",
                labels=labels,
                boundaries=boundaries,
                args=args,
                txt_fmt="%.6f",
                png_vmin=0.0,
                png_vmax=1.0,
                html_zmin=0.0,
                html_zmax=1.0,
            )

            save_pair_output(
                base_path=os.path.join(frequency_dir, f"{safe_id}_frequency_error_{temp}K"),
                matrix=err_map,
                title=f"{seq_id} frequency residual/error prediction ({temp} K)",
                labels=labels,
                boundaries=boundaries,
                args=args,
                txt_fmt="%.6f",
            )

    # -------------------------
    # Kinetics head
    # -------------------------
    if "kinetic_prob" in prediction:
        kinetic_prob = prediction["kinetic_prob"][seq_idx].detach().cpu().numpy()              # [T, 2, L, L, C]
        kinetic_pred_class = prediction["kinetic_pred_class"][seq_idx].detach().cpu().numpy()  # [T, 2, L, L]
        kinetic_conf = prediction["kinetic_confidence"][seq_idx].detach().cpu().numpy()        # [T, L]

        kinetics_dir = os.path.join(sample_dir, "kinetics")
        ensure_dir(kinetics_dir)

        raw_bundle["kinetic_prob"] = []
        raw_bundle["kinetic_pred_class"] = []
        raw_bundle["kinetic_confidence"] = []

        rate_names = ["on", "off"]
        class_name_lookup = {
            "on": KINETIC_ON_CLASS_NAMES,
            "off": KINETIC_OFF_CLASS_NAMES,
        }

        for t_idx, temp in enumerate(TEMPERATURES):
            conf_vec = crop_residue_vector(kinetic_conf[t_idx], mask)
            raw_bundle["kinetic_confidence"].append(conf_vec)

            save_vector_output(
                base_path=os.path.join(kinetics_dir, f"{safe_id}_kinetics_confidence_{temp}K"),
                labels=labels,
                values=conf_vec,
                title=f"{seq_id} kinetics confidence ({temp} K)",
                ylabel="Confidence",
                args=args,
            )

            temp_prob = {}
            temp_pred = {}

            for rate_idx, rate_name in enumerate(rate_names):
                pred_map = crop_pair_matrix(kinetic_pred_class[t_idx, rate_idx], mask)

                # Crop the full probability stack class-by-class.
                full_prob_stack = kinetic_prob[t_idx, rate_idx]  # [L, L, C]
                cropped_prob_stack = np.stack(
                    [crop_pair_matrix(full_prob_stack[:, :, c], mask) for c in range(full_prob_stack.shape[-1])],
                    axis=-1,
                )  # [Lc, Lc, C]

                temp_pred[rate_name] = pred_map
                temp_prob[rate_name] = cropped_prob_stack

                save_pair_output(
                    base_path=os.path.join(
                        kinetics_dir,
                        f"{safe_id}_kinetics_{rate_name}_class_{temp}K",
                    ),
                    matrix=pred_map.astype(np.int64),
                    title=f"{seq_id} kinetics {rate_name}-time predicted class ({temp} K)",
                    labels=labels,
                    boundaries=boundaries,
                    args=args,
                    txt_fmt="%d",
                    png_vmin=0,
                    png_vmax=len(class_name_lookup[rate_name]) - 1,
                    html_zmin=0,
                    html_zmax=len(class_name_lookup[rate_name]) - 1,
                )

                save_kinetics_probabilities_npz(
                    base_path=os.path.join(
                        kinetics_dir,
                        f"{safe_id}_kinetics_{rate_name}_probabilities_{temp}K",
                    ),
                    prob_stack=cropped_prob_stack,
                    class_names=class_name_lookup[rate_name],
                )

                # Save legend for class indices.
                legend_path = os.path.join(
                    kinetics_dir,
                    f"{safe_id}_kinetics_{rate_name}_classes_{temp}K.txt",
                )
                with open(legend_path, "w") as f:
                    for class_idx, class_name in enumerate(class_name_lookup[rate_name]):
                        f.write(f"{class_idx}\t{class_name}\n")

            raw_bundle["kinetic_prob"].append(temp_prob)
            raw_bundle["kinetic_pred_class"].append(temp_pred)

    # -------------------------
    # Optional: native contacts and native-vs-dynamic maps
    # -------------------------
    if "native_contacts" in prediction:
        native_dir = os.path.join(sample_dir, "native")
        ensure_dir(native_dir)

        native_contacts = prediction["native_contacts"][seq_idx].detach().cpu().numpy()
        native_contacts = crop_pair_matrix(native_contacts, mask)
        raw_bundle["native_contacts"] = native_contacts

        save_pair_output(
            base_path=os.path.join(native_dir, f"{safe_id}_native_contacts"),
            matrix=native_contacts.astype(np.int64),
            title=f"{seq_id} ESMFold native contacts",
            labels=labels,
            boundaries=boundaries,
            args=args,
            txt_fmt="%d",
            png_vmin=0,
            png_vmax=1,
            html_zmin=0,
            html_zmax=1,
        )

    if "dynamic_nonnative_contacts" in prediction:
        dnn_dir = os.path.join(sample_dir, "dynamic_nonnative")
        ensure_dir(dnn_dir)
        raw_bundle["dynamic_nonnative_contacts"] = []

        dynamic_nonnative = prediction["dynamic_nonnative_contacts"][seq_idx].detach().cpu().numpy()
        for t_idx, temp in enumerate(TEMPERATURES):
            mat = crop_pair_matrix(dynamic_nonnative[t_idx], mask)
            raw_bundle["dynamic_nonnative_contacts"].append(mat)
            save_pair_output(
                base_path=os.path.join(dnn_dir, f"{safe_id}_dynamic_nonnative_{temp}K"),
                matrix=mat.astype(np.int64),
                title=f"{seq_id} dynamic but non-native contacts ({temp} K)",
                labels=labels,
                boundaries=boundaries,
                args=args,
                txt_fmt="%d",
                png_vmin=0,
                png_vmax=1,
                html_zmin=0,
                html_zmax=1,
            )

    if "native_nondynamic_contacts" in prediction:
        ndn_dir = os.path.join(sample_dir, "native_nondynamic")
        ensure_dir(ndn_dir)
        raw_bundle["native_nondynamic_contacts"] = []

        native_nondynamic = prediction["native_nondynamic_contacts"][seq_idx].detach().cpu().numpy()
        for t_idx, temp in enumerate(TEMPERATURES):
            mat = crop_pair_matrix(native_nondynamic[t_idx], mask)
            raw_bundle["native_nondynamic_contacts"].append(mat)
            save_pair_output(
                base_path=os.path.join(ndn_dir, f"{safe_id}_native_nondynamic_{temp}K"),
                matrix=mat.astype(np.int64),
                title=f"{seq_id} native but non-dynamic contacts ({temp} K)",
                labels=labels,
                boundaries=boundaries,
                args=args,
                txt_fmt="%d",
                png_vmin=0,
                png_vmax=1,
                html_zmin=0,
                html_zmax=1,
            )

    # Save PDB if available.
    if "pdbs" in prediction:
        pdb_path = os.path.join(sample_dir, f"{safe_id}.pdb")
        with open(pdb_path, "w") as f:
            f.write(prediction["pdbs"][seq_idx])
        raw_bundle["pdb_path"] = pdb_path

    # Save a single raw bundle for convenience.
    if args.save_raw_pt:
        torch.save(raw_bundle, os.path.join(sample_dir, f"{safe_id}_all_outputs.pt"))


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = esm.pretrained.esmdynamic()
    model.set_chunk_size(args.chunk_size)
    model = model.to(device)
    model.eval()

    sequences = load_sequences(args)
    if len(sequences) == 0:
        raise ValueError("No sequences were loaded.")

    for start in range(0, len(sequences), args.batch_size):
        batch = sequences[start:start + args.batch_size]
        ids, raw_seqs = zip(*batch)

        with torch.no_grad():
            prediction = model.predict_from_seqs(
                list(raw_seqs),
                low_memory=args.low_memory,
                num_recycles=args.num_recycles,
            )

        for batch_idx, (seq_id, seq) in enumerate(zip(ids, raw_seqs)):
            save_outputs_for_sequence(
                output_dir=args.output_dir,
                seq_id=seq_id,
                sequence=seq,
                prediction=prediction,
                seq_idx=batch_idx,
                args=args,
            )


if __name__ == "__main__":
    main()
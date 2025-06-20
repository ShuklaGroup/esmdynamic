import argparse
import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from esm.pretrained import esmdynamic
from Bio import SeqIO


def parse_args():
    parser = argparse.ArgumentParser(description="Predict dynamic contacts using ESMDynamic.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", type=str, help="Single sequence string.")
    group.add_argument("--fasta", type=str, help="Path to FASTA file with sequences.")
    group.add_argument("--csv", type=str, help="CSV file with sequences (first column ID, second column sequence).")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default 1).")
    parser.add_argument("--chunk_size", type=int, default=256, help="Model chunk size (default 256).")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device (default: cuda).")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--chain_ids", type=str, help="Chain IDs to use (e.g., 'ABCDEF'). Default: A-Z.")

    return parser.parse_args()


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
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    sequences.append((row[0], row[1]))
    return sequences


def get_crop_mask_and_chain_info(sequence, insert_len=25, chain_ids=None):
    """
    Builds crop mask, residue labels with chain IDs, and boundary lines.
    """
    if chain_ids is None:
        chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chain_idx = 0
    mask = []
    labels = []
    boundaries = []
    res_counter = 1

    for res in sequence:
        if res == ":":
            chain_idx += 1
            res_counter = 1
            boundaries.append(len(mask))  # position after crop
            mask.extend([False] * insert_len)
        else:
            mask.append(True)
            chain_id = chain_ids[chain_idx % len(chain_ids)]
            labels.append(f"{chain_id}-{res}{res_counter}")
            res_counter += 1

    return mask, labels, boundaries


def save_outputs(output_dir, id, contact_map, mask, labels, boundaries):
    os.makedirs(output_dir, exist_ok=True)

    idx = [i for i, keep in enumerate(mask) if keep]
    contact_map = contact_map[np.ix_(idx, idx)]

    # Save .npy
    np.save(os.path.join(output_dir, f"{id}.npy"), contact_map)

    # Save .png with chain lines
    plt.figure(figsize=(6, 6))
    plt.matshow(contact_map, cmap='viridis')
    for pos in boundaries:
        plt.axhline(pos - 0.5, color='white', linewidth=1)
        plt.axvline(pos - 0.5, color='white', linewidth=1)
    plt.title(f"{id} Contact Map with Chain Boundaries", y=1.1)
    plt.savefig(os.path.join(output_dir, f"{id}.png"))
    plt.close()

    # Save .html with Plotly
    fig = px.imshow(
        contact_map,
        labels=dict(x="Residue", y="Residue", color="Contact Prob."),
        x=labels,
        y=labels,
        color_continuous_scale="Viridis"
    )
    for pos in boundaries:
        fig.add_shape(type="line", x0=pos - 0.5, x1=pos - 0.5, y0=-0.5, y1=len(labels) - 0.5,
                      line=dict(color="white", width=1))
        fig.add_shape(type="line", y0=pos - 0.5, y1=pos - 0.5, x0=-0.5, x1=len(labels) - 0.5,
                      line=dict(color="white", width=1))

    fig.write_html(os.path.join(output_dir, f"{id}.html"))


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = esmdynamic()
    model.set_chunk_size(args.chunk_size)
    model = model.to(device)

    sequences = load_sequences(args)
    chain_ids = args.chain_ids if args.chain_ids else None

    for i in range(0, len(sequences), args.batch_size):
        batch = sequences[i:i + args.batch_size]
        ids, raw_seqs = zip(*batch)

        with torch.no_grad():
            prediction = model.predict_from_seqs(raw_seqs)

        probs = prediction["dynamic_contact_prob"]  # (B, C, L, L)
        contact_probs = probs[:, 0]  # Use first channel (dynamic contact prob)

        for id_, seq, matrix in zip(ids, raw_seqs, contact_probs):
            matrix_np = matrix.cpu().numpy()
            mask, labels, boundaries = get_crop_mask_and_chain_info(seq, chain_ids=chain_ids)
            save_outputs(args.output_dir, id_, matrix_np, mask, labels, boundaries)


if __name__ == "__main__":
    main()

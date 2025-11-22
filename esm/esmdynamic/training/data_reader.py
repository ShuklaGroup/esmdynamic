"""
Data reader for ESMDynamic training.
"""

import os
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class DynContactDataset(Dataset):
    def __init__(self, data_dir, identifiers, crop_length, weights=None):
        self.data_dir = data_dir
        self.identifiers = identifiers
        self.crop_length = crop_length
        self.weights = weights


    def __len__(self):
        return len(self.identifiers)

    # -----------------------------
    # Loading helpers
    # -----------------------------
    def _load_sequence(self, fpath):
        with open(fpath, "r") as f:
            return f.readlines()[1].strip()  # assume 2-line fasta

    def _load_dynamic(self, fpath):
        # dynamic_contacts.pt has shape (5, L, L)
        arr = torch.load(fpath).float()
        return arr  # [5, L, L]

    def _load_kinetics(self, fpath):
        # kinetics.pt → (5, 2, L, L)
        return torch.load(fpath).float()

    def _load_frequency(self, fpath):
        # frequency.pt → (5, L, L)
        return torch.load(fpath).float()

    # -----------------------------
    # Collate function
    # -----------------------------
    def custom_collate_fn(self, batch):
        """
        batch is list of tuples:
          (sequence, dynamic, kinetics, frequency, length)
        """
        sequences, dynamics, kinetics, freqs, lengths = zip(*batch)
        lengths = list(lengths)
        Lmax = max([len(s) for s in sequences])

        # padded outputs
        dyn_out = []
        kin_out = []
        freq_out = []

        for dyn, kin, freq in zip(dynamics, kinetics, freqs):
            C = dyn.size(0)    # 5
            R = kin.size(1)    # 2

            pad_dyn  = torch.zeros((C, Lmax, Lmax))
            pad_kin  = torch.zeros((C, R, Lmax, Lmax))
            pad_freq = torch.zeros((C, Lmax, Lmax))

            L = dyn.size(1)

            pad_dyn[:,  :L, :L] = dyn
            pad_kin[:, :, :L, :L] = kin
            pad_freq[:, :L, :L] = freq

            dyn_out.append(pad_dyn)
            kin_out.append(pad_kin)
            freq_out.append(pad_freq)

        return (
            sequences,
            torch.stack(dyn_out),   # [B, 5, Lmax, Lmax]
            torch.stack(kin_out),   # [B, 5, 2, Lmax, Lmax]
            torch.stack(freq_out),  # [B, 5, Lmax, Lmax]
            torch.tensor(lengths),
        )

    # -----------------------------
    # Weighted sampler
    # -----------------------------
    def _compute_sampling_weights(self):
        weights = torch.zeros(len(self.identifiers))
        for i, identifier in enumerate(self.identifiers):
            seq_path = os.path.join(self.data_dir, identifier, "consensus.fasta")
            weights[i] = len(self._load_sequence(seq_path))
        self.weights = weights / weights.sum()

    def weighted_random_sampler(self, num_samples):
        if self.weights is None:
            self._compute_sampling_weights()
        return WeightedRandomSampler(
            weights=self.weights,
            num_samples=num_samples,
            replacement=True,
        )

    # -----------------------------
    # __getitem__
    # -----------------------------
    def __getitem__(self, idx):
        identifier = self.identifiers[idx]
        d = os.path.join(self.data_dir, identifier)

        seq = self._load_sequence(os.path.join(d, "consensus.fasta"))
        dyn = self._load_dynamic(os.path.join(d, "dynamic_contacts.pt"))
        kin = self._load_kinetics(os.path.join(d, "kinetics.pt"))
        freq = self._load_frequency(os.path.join(d, "frequency.pt"))

        L = len(seq)

        # Crop positions
        if self.crop_length >= L:
            start, end = 0, L
        else:
            start = torch.randint(0, L - self.crop_length, ()).item()
            end = start + self.crop_length

        # Crop sequence
        seq_crop = seq[start:end]

        # Crop arrays
        dyn_crop = dyn[:, start:end, start:end]          # [5, Lc, Lc]
        kin_crop = kin[:, :, start:end, start:end]       # [5, 2, Lc, Lc]
        freq_crop = freq[:, start:end, start:end]        # [5, Lc, Lc]

        return seq_crop, dyn_crop, kin_crop, freq_crop, len(seq_crop)


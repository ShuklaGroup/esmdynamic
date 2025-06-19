"""
Data reader for ESMDynamic training.
"""

import os
import torch
from scipy.spatial.distance import squareform
from torch.utils.data import Dataset, WeightedRandomSampler
import random


class DynContactDataset(Dataset):
    def __init__(self, data_dir, identifiers, crop_length, weights=None):
        """
        Args:
            data_dir (str): Path to the main data folder containing subdirectories.
            identifiers (list of str): List of identifiers (subdirectory names) to include in the dataset.
            weights (torch.Tensor): Weights for random sampler.
        """
        self.data_dir = data_dir
        self.identifiers = identifiers
        self.crop_length = crop_length
        self.weights = weights


    def __len__(self):
        return len(self.identifiers)


    def _load_dynamic_contacts(self, fpath):
        dynamic_contacts = torch.load(fpath)
        dynamic_contacts = squareform(dynamic_contacts)
        N = dynamic_contacts.shape[0]
        dynamic_contacts = torch.from_numpy(dynamic_contacts).reshape((1, N, N)).float()
        return dynamic_contacts


    def _load_sequence(self, fpath):
        with open(fpath, "r") as f:
            return f.readlines()[1].strip()


    def _compute_sampling_weights(self):
        weights = torch.zeros(len(self.identifiers))
        for i, identifier in enumerate(self.identifiers):
            dir_path = os.path.join(self.data_dir, identifier)
            seq_file = os.path.join(dir_path, "consensus.fasta")
            weights[i] = len(self._load_sequence(seq_file))
        self.weights = weights / weights.sum()


    def custom_collate_fn(self, batch):
        '''Custom collate function, use with data loader.
        '''
        # Extract sequences, arrays, and lengths from the batch (lengths correspond to already-cropped sequences)
        sequences, arrays, lengths = zip(*batch)
        cropped_lengths = [len(s) for s in sequences]
        
        # Determine the maximum length in this batch
        max_length = max(cropped_lengths)
        
        # Initialize list for padded arrays
        padded_arrays = []

        for arr in arrays:
            # Pad the array to (max_length, max_length) with zeros
            padded_arr = torch.zeros((1, max_length, max_length), dtype=arr.dtype)
            padded_arr[:, :arr.size(1), :arr.size(2)] = arr
            padded_arrays.append(padded_arr)
        
        # Stack the padded arrays into a single tensor batch
        batch_arrays = torch.stack(padded_arrays)
        
        # Return sequences as-is and padded arrays
        return sequences, batch_arrays, torch.tensor(lengths)


    def weighted_random_sampler(self, num_samples):
        '''Provides a random sampler where weights are proportional to sequence lengths. Can be passed to a DataLoader.
        '''
        if self.weights is None:
            self._compute_sampling_weights()
        return WeightedRandomSampler(weights=self.weights, num_samples=num_samples, replacement=True)


    def __getitem__(self, idx):
        # Get the subdirectory identifier for this item
        identifier = self.identifiers[idx]
        
        # Construct paths to the sequence and label files
        dir_path = os.path.join(self.data_dir, identifier)
        seq_file = os.path.join(dir_path, "consensus.fasta")
        label_file = os.path.join(dir_path, "dynamic_contacts.pt")
        
        # Load sequence
        sequence = self._load_sequence(seq_file)
        
        # Load label array
        label_array = self._load_dynamic_contacts(label_file)
        
        # Select a crop
        length = len(sequence)
        if self.crop_length >= length:
            start, end = 0, length
        else:
            start = torch.randint(0, length - self.crop_length, (1,)).item()
            end = start + self.crop_length
        
        # Crop the sequence and array
        cropped_sequence = sequence[start:end]
        cropped_array = label_array[:, start:end, start:end]
        
        return cropped_sequence, cropped_array, length


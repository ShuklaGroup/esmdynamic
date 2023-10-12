"""
Data reader for ESMDynamic training.
"""

import os
from glob import glob
from natsort import natsorted
from scipy.spatial.distance import squareform
import torch


# TODO: convert collection of functions to reader object

def access_cached_output(cluster_id, crop_index=0):
    """Access ESMFold cached output for a given cluster index.
    The crop index can be specified for proteins longer than 512 residues.
    """
    basepath = "/mnt/sdb/10-DynamicESM/build_dataset/cached_esm_output/"
    cluster_path = os.path.join(basepath, "{:08d}".format(cluster_id))
    filepaths = natsorted(glob(os.path.join(cluster_path, "cached_*.pt")))
    return torch.load(filepaths[crop_index])


def load_dynamic_contacts(fpath):
    """Load ground-truth dynamic contacts. To be called from access_labels().
    """
    dynamic_contacts = torch.load(fpath)
    dynamic_contacts = squareform(dynamic_contacts)
    N = dynamic_contacts.shape[0]
    dynamic_contacts = torch.from_numpy(dynamic_contacts).reshape((1, 1, N, N)).float()
    return dynamic_contacts


def load_rmsd(fpath, rmsd_bin_boundaries):
    """Load ground-truth RMSDs. To be called from access_labels().
    """
    rmsd = torch.load(fpath)
    N = rmsd.shape[0]
    M = len(rmsd_bin_boundaries) + 1
    binned_output = torch.empty((1, len(rmsd_bin_boundaries) + 1, N))
    for i, r in enumerate(rmsd):
        binned_output[0, :, i] = torch.nn.functional.one_hot(
            torch.bucketize(r, rmsd_bin_boundaries),
            M,
        )
    return binned_output


def access_labels(cluster_id,
                  rmsd_bin_boundaries,
                  basepath="/mnt/sdb/10-DynamicESM/build_dataset/pdb_clusters_new/"
                  ):
    """Load ground-truth labels.
    Args:
        cluster_id (int): cluster index.
        rmsd_bin_boundaries (Tensor): bin boundaries used to discretize RMSD values.
        basepath (str): path to dataset.
    """
    cluster_path = os.path.join(basepath, "{:08d}".format(cluster_id))
    rmsd_fpath = os.path.join(cluster_path, "rmaxsd.pt")
    rmsd = load_rmsd(rmsd_fpath, rmsd_bin_boundaries)
    dyncon_fpath = os.path.join(cluster_path, "dynamic_contacts.pt")
    dynamic_contacts = load_dynamic_contacts(dyncon_fpath)
    return dynamic_contacts, rmsd

def get_crop_start_end(protein_length, crop_idx, max_len=512, overlap=256):
    """
    Calculate the start and end indices for a crop of a protein.
    """
    if (protein_length <= max_len) and (crop_idx == 0): # Single chunk
        start = 0
        end = protein_length
    elif (crop_idx-1) * overlap + max_len >= protein_length: # Out of range
        raise ValueError(f"Protein of length {protein_length} "
                         f"does not have {crop_idx + 1} crops "
                         f"for max length {max_len} and overlap {overlap}.")
    elif crop_idx * overlap + max_len > protein_length: # Last chunk
        start = protein_length - max_len
        end = protein_length
    else: # Intermediate chunk
        start = crop_idx * overlap
        end = crop_idx * overlap + max_len

    return start, end

# Work in progress...

def get_batched_labels(cluster_ids,
                       max_len=512,
                       crop_indices=None,
                       rmsd_bin_boundaries=rmsd_bin_boundaries,
                       ):
    batch_size = len(cluster_ids)
    labels_dyn_contacts = torch.zeros((batch_size, 1, max_len, max_len))
    labels_rmsd = torch.zeros((batch_size, len(rmsd_bin_boundaries) + 1, max_len))
    if crop_indices is None:
        crop_indices = torch.zeros((batch_size))
    for i, (cluster_id, crop_idx) in enumerate(zip(cluster_ids, crop_indices)):
        dyn_contacts, rmsd = access_labels(cluster_id, rmsd_bin_boundaries)
        prot_length = rmsd.shape[-1]
        crop_start, crop_end = get_crop_start_end(prot_length, crop_idx)
        crop_length = crop_end - crop_start  # Might be less than max_len
        labels_dyn_contacts[i, 0, :crop_length, :crop_length] = dyn_contacts[0, 0, crop_start:crop_end,
                                                                crop_start:crop_end]
        labels_rmsd[i, :, :crop_length] = rmsd[0, :, crop_start:crop_end]


def get_batched_data_labels(cluster_ids,
                            max_len=512,
                            crop_indices=None,
                            rmsd_bin_boundaries=rmsd_bin_boundaries,
                            ):
    pass

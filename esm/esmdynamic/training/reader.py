"""
Data reader for ESMDynamic training.
"""

import os
from glob import glob
from natsort import natsorted
from scipy.spatial.distance import squareform
import torch


# TODO: convert collection of functions to reader object

def access_cached_output(cluster_id,
                         crop_index=0,
                         basepath="/mnt/sdb/10-DynamicESM/build_dataset/cached_esm_output/"
                         ):
    """Access ESMFold cached output for a given cluster index.
    The crop index can be specified for proteins longer than 512 residues.
    """
    cluster_path = os.path.join(basepath, "{:08d}".format(cluster_id))
    filepaths = natsorted(glob(os.path.join(cluster_path, "cached_*.pt")))
    return torch.load(filepaths[crop_index])


def get_batched_data(cluster_ids,
                     max_len=512,
                     crop_indices=None,
                     c_s=1024,
                     c_z=128
                     ):
    # Initialize tensors
    batch_size = len(cluster_ids)
    s_s = torch.zeros((batch_size, max_len, c_s))
    s_z = torch.zeros((batch_size, max_len, max_len, c_z))
    aatype = torch.zeros((batch_size, max_len)).long()
    mask = torch.zeros((batch_size, max_len)).long()
    residue_index = torch.zeros((batch_size, max_len)).long()
    frames = torch.zeros((8, batch_size, max_len, 7))
    sidechain_frames = torch.zeros((8, batch_size, max_len, 8, 4, 4))
    unnormalized_angles = torch.zeros((8, batch_size, max_len, 7, 2))
    angles = torch.zeros((8, batch_size, max_len, 7, 2))
    positions = torch.zeros((8, batch_size, max_len, 14, 3))
    states = torch.zeros((8, batch_size, max_len, 384))
    single = torch.zeros((batch_size, max_len, 384))

    if crop_indices is None:
        crop_indices = torch.zeros((batch_size)).long()
    for i, (cluster_id, crop_idx) in enumerate(zip(cluster_ids, crop_indices)):
        cached_data = access_cached_output(cluster_id, crop_idx)
        _, L, _ = cached_data['s_s'].shape
        s_s[i, :L, :] = cached_data['s_s']
        s_z[i, :L, :L, :] = cached_data['s_z']
        aatype[i, :L] = cached_data['aatype']
        mask[i, :L] = cached_data['mask']
        residue_index[i, :L] = cached_data['residue_index']
        frames[:, i, :L, :] = cached_data['frames'][:, 0, :, :]
        sidechain_frames[:, i, :L, :, :, :] = cached_data['sidechain_frames'][:, 0, :, :, :, :]
        unnormalized_angles[:, i, :L, :, :] = cached_data['unnormalized_angles'][:, 0, :, :, :]
        angles[:, i, :L, :, :] = cached_data['angles'][:, 0, :, :, :]
        positions[:, i, :L, :, :] = cached_data['positions'][:, 0, :, :, :]
        states[:, i, :L, :] = cached_data['states'][:, 0, :, :]
        single[i, :L, :] = cached_data['single']

    return dict(
        s_s=s_s,
        s_z=s_z,
        aatype=aatype,
        mask=mask,
        residue_index=residue_index,
        frames=frames,
        sidechain_frames=sidechain_frames,
        unnormalized_angles=unnormalized_angles,
        angles=angles,
        positions=positions,
        states=states,
        single=single,
    )


def load_dynamic_contacts(fpath):
    '''Load ground-truth dynamic contacts. To be called from access_labels().
    '''
    dynamic_contacts = torch.load(fpath)
    dynamic_contacts = squareform(dynamic_contacts)
    N = dynamic_contacts.shape[0]
    dynamic_contacts = torch.from_numpy(dynamic_contacts).reshape((1, 1, N, N)).float()
    return dynamic_contacts


def load_rmsd(fpath, rmsd_bin_boundaries):
    '''Load ground-truth RMSDs. To be called from access_labels().
    '''
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
    '''Load ground-truth labels.
    Args:
        cluster_id (int): cluster index.
        rmsd_bin_boundaries (Tensor): bin boundaries used to discretize RMSD values.
        basepath (str): path to dataset.
    '''
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
    if (protein_length <= max_len) and (crop_idx == 0):  # Single chunk
        start = 0
        end = protein_length
    elif (crop_idx - 1) * overlap + max_len >= protein_length:  # Out of range
        raise ValueError(f"Protein of length {protein_length} "
                         f"does not have {crop_idx + 1} crops "
                         f"for max length {max_len} and overlap {overlap}.")
    elif crop_idx * overlap + max_len > protein_length:  # Last chunk
        start = protein_length - max_len
        end = protein_length
    else:  # Intermediate chunk
        start = crop_idx * overlap
        end = crop_idx * overlap + max_len

    return start, end


def get_batched_labels(cluster_ids,
                       max_len=512,
                       crop_indices=None,
                       rmsd_bin_boundaries=rmsd_bin_boundaries,
                       ):
    batch_size = len(cluster_ids)
    labels_dyn_contacts = torch.zeros((batch_size, 1, max_len, max_len))
    labels_rmsd = torch.zeros((batch_size, len(rmsd_bin_boundaries) + 1, max_len))
    protein_lengths = torch.zeros((batch_size)).long()
    if crop_indices is None:
        crop_indices = torch.zeros((batch_size)).long()
    for i, (cluster_id, crop_idx) in enumerate(zip(cluster_ids, crop_indices)):
        dyn_contacts, rmsd = access_labels(cluster_id, rmsd_bin_boundaries)
        prot_length = rmsd.shape[-1]
        protein_lengths[i] = prot_length
        crop_start, crop_end = get_crop_start_end(prot_length, crop_idx)
        crop_length = crop_end - crop_start  # Might be less than max_len
        labels_dyn_contacts[i, 0, :crop_length, :crop_length] = dyn_contacts[0, 0, crop_start:crop_end,
                                                                crop_start:crop_end]
        labels_rmsd[i, :, :crop_length] = rmsd[0, :, crop_start:crop_end]

    return dict(dynamic_contacts=labels_dyn_contacts,
                rmsd=labels_rmsd,
                protein_lengths=protein_lengths
                )


def get_batched_data_labels(cluster_ids,
                            max_len=512,
                            crop_indices=None,
                            rmsd_bin_boundaries=rmsd_bin_boundaries,
                            ):
    inputs = get_batched_data(cluster_ids,
                              max_len=512,
                              crop_indices=None,
                              c_s=1024,
                              c_z=128
                              )

    targets = get_batched_labels(cluster_ids,
                                 max_len=max_len,
                                 crop_indices=crop_indices,
                                 rmsd_bin_boundaries=rmsd_bin_boundaries
                                 )
    return inputs, targets

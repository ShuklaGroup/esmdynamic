"""
Data reader for ESMDynamic training.
"""

import os
from glob import glob
from natsort import natsorted
from scipy.spatial.distance import squareform
import torch
from torch.utils.data import Dataset
from ..utils import rmsd_bin_boundaries


def access_cached_output(cluster_id,
                         crop_index=0,
                         basepath="/mnt/sdb/10-DynamicESM/build_dataset/cached_esm_output/"
                         ):
    """Access ESMFold cached output for a given cluster index.
    The crop index can be specified for proteins longer than 512 residues.
    """
    cluster_path = os.path.join(basepath, "{:08d}".format(cluster_id))
    filepaths = natsorted(glob(os.path.join(cluster_path, "cached_structure_*.pt")))
    return torch.load(filepaths[crop_index])


def get_input(cluster_id,
              max_len=512,
              crop_id=None,
              c_s=1024,
              c_z=128,
              data_dirpath=""
              ):
    # Initialize tensors
    s_s = torch.zeros((max_len, c_s))
    s_z = torch.zeros((max_len, max_len, c_z))
    # aatype = torch.zeros((max_len)).long()
    mask = torch.zeros((max_len)).long()
    residue_index = torch.zeros((max_len)).long()
    # frames = torch.zeros((8, max_len, 7))
    # sidechain_frames = torch.zeros((8, max_len, 8, 4, 4))
    # unnormalized_angles = torch.zeros((8, max_len, 7, 2))
    # angles = torch.zeros((8, max_len, 7, 2))
    # positions = torch.zeros((8, max_len, 14, 3))
    # states = torch.zeros((8, max_len, 384))
    # single = torch.zeros((max_len, 384))
    lddt_head = torch.zeros((8, max_len, 37, 50)) # 8 = states, 37 = seq_transition, 50 = esmfold.lddt_bins
    lm_logits = torch.zeros((max_len, 23)) # 23 = esmfold.n_tokens_embed
    ptm_logits = torch.zeros((max_len, max_len, 64)) # 64 = esmfold.distogram_bins
    distogram_logits = torch.zeros((max_len, max_len, 64)) # 64 = esmfold.distogram_bins

    if crop_id is None:
        crop_id = 0

    cached_data = access_cached_output(cluster_id, crop_id, basepath=data_dirpath)
    _, L, _ = cached_data['s_s'].shape
    s_s[:L, :] = cached_data['s_s']
    s_z[:L, :L, :] = cached_data['s_z']
    # aatype[:L] = cached_data['aatype']
    mask[:L] = cached_data['mask']
    residue_index[:L] = cached_data['residue_index']
    # frames[:, :L, :] = cached_data['frames'][:, 0, :, :]
    # sidechain_frames[:, :L, :, :, :] = cached_data['sidechain_frames'][:, 0, :, :, :, :]
    # unnormalized_angles[:, :L, :, :] = cached_data['unnormalized_angles'][:, 0, :, :, :]
    # angles[:, :L, :, :] = cached_data['angles'][:, 0, :, :, :]
    # positions[:, :L, :, :] = cached_data['positions'][:, 0, :, :, :]
    # states[:, :L, :] = cached_data['states'][:, 0, :, :]
    # single[:L, :] = cached_data['single']
    lddt_head[:, :L, :, :] = cached_data['lddt_head'][:, 0, :, :, :]
    lm_logits[:L, :] = cached_data['lm_logits'][0, :, :]
    ptm_logits[:L, :L, :] = cached_data['ptm_logits'][0, :, :, :]
    distogram_logits[:L, :L, :] = cached_data['distogram_logits'][0, :, :, :]

    return dict(
        s_s=s_s,
        s_z=s_z,
        mask=mask,
        residue_index=residue_index,
        lddt_head=lddt_head,
        lm_logits=lm_logits,
        ptm_logits=ptm_logits,
        distogram_logits=distogram_logits,
    )


def fix_dim_order(batch_data):
    """For certain keys batch dimension should be dim1 instead dim0.
    """
    # Keys to modify
    change_keys = {"lddt_head"}
    # Dim order change auxiliary function
    change_dims = lambda x: torch.transpose(x, 1, 0)

    batch_data = {
        k: change_dims(v) if k in change_keys else v for k, v in batch_data.items()
    }

    return batch_data


def load_dynamic_contacts(fpath):
    """Load ground-truth dynamic contacts. To be called from access_labels().
    """
    dynamic_contacts = torch.load(fpath)
    dynamic_contacts = squareform(dynamic_contacts)
    N = dynamic_contacts.shape[0]
    dynamic_contacts = torch.from_numpy(dynamic_contacts).reshape((1, 1, N, N)).float()
    return dynamic_contacts


def load_rmsd(fpath, rmsd_bin_boundaries):
    """Load ground-truth RMSDs and bin the values.
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


def load_binned_rmsd(fpath):
    """Load ground-truth RMSDs. To be called from access_labels().
    """
    return torch.load(fpath)


def access_labels(cluster_id,
                  basepath="/mnt/sdb/10-DynamicESM/build_dataset/pdb_clusters_new/"
                  ):
    """Load ground-truth labels.
    Args:
        cluster_id (int): cluster index.
        basepath (str): path to dataset.
    """
    cluster_path = os.path.join(basepath, "{:08d}".format(cluster_id))
    rmsd_fpath = os.path.join(cluster_path, "rmsd_binned.pt")
    rmsd = load_binned_rmsd(rmsd_fpath)
    dyncon_fpath = os.path.join(cluster_path, "dynamic_contacts.pt")
    dynamic_contacts = load_dynamic_contacts(dyncon_fpath)

    return dynamic_contacts, rmsd


def get_crop_start_end(protein_length, crop_id, max_len=512, overlap=256):
    """
    Calculate the start and end indices for a crop of a protein.
    """
    if (protein_length <= max_len) and (crop_id == 0):  # Single chunk
        start = 0
        end = protein_length
    elif (crop_id - 1) * overlap + max_len >= protein_length:  # Out of range
        raise ValueError(f"Protein of length {protein_length} "
                         f"does not have {crop_id + 1} crops "
                         f"for max length {max_len} and overlap {overlap}.")
    elif crop_id * overlap + max_len > protein_length:  # Last chunk
        start = protein_length - max_len
        end = protein_length
    else:  # Intermediate chunk
        start = crop_id * overlap
        end = crop_id * overlap + max_len

    return start, end


def get_labels(cluster_id,
               max_len=512,
               crop_id=None,
               rmsd_bin_number=len(rmsd_bin_boundaries) + 1,
               labels_dirpath=""
               ):
    labels_dyn_contacts = torch.zeros((1, max_len, max_len))
    labels_rmsd = torch.zeros((rmsd_bin_number, max_len))
    protein_length = torch.zeros(1).long()
    if crop_id is None:
        crop_id = 0

    dyn_contacts, rmsd = access_labels(cluster_id, basepath=labels_dirpath)
    prot_length = rmsd.shape[-1]
    protein_length[0] = prot_length
    crop_start, crop_end = get_crop_start_end(prot_length, crop_id)
    crop_length = crop_end - crop_start  # Might be less than max_len
    labels_dyn_contacts[0, :crop_length, :crop_length] = dyn_contacts[0, 0, crop_start:crop_end, crop_start:crop_end]
    labels_rmsd[:, :crop_length] = rmsd[0, :, crop_start:crop_end]

    return dict(dynamic_contacts=labels_dyn_contacts,
                rmsd=labels_rmsd,
                protein_lengths=protein_length
                )


def get_input_labels(cluster_id,
                     max_len=512,
                     crop_id=None,
                     data_dirpath="",
                     labels_dirpath="",
                     ):
    inputs = get_input(cluster_id,
                       max_len=512,
                       crop_id=crop_id,
                       c_s=1024,
                       c_z=128,
                       data_dirpath=data_dirpath
                       )

    labels = get_labels(cluster_id,
                        max_len=max_len,
                        crop_id=crop_id,
                        labels_dirpath=labels_dirpath
                        )

    return inputs, labels


class DynContactDataset(Dataset):
    """Custom Dataset object for dynamic contacts.

    Args:
        data_dir (str): path where input data are stored.
        labels_dir (str): path where ground-truth labels are stored.
        cluster_indices (list[list[int]]): list with cluster and crop indices.
            Indices are read as `cluster_index, crop_index = cluster_indices[index]` for some int `index`.

    Usage:
        Using indices saved in `cluster_indices.pkl`.
        >>> import pickle
        >>> from torch.utils.data import DataLoader
        >>> cluster_indices = pickle.load(open("cluster_indices.pkl", "rb"))
        >>> dataset = DynContactDataset(data_dir="/.", labels_dir="/.", cluster_indices=cluster_indices)
        >>> dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    """

    def __init__(self, data_dir, labels_dir, cluster_indices):
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.cluster_indices = cluster_indices  # Samples (crops) to include in the dataset

    def __len__(self):
        return len(self.cluster_indices)

    def __getitem__(self, index):
        idx, crop_idx = self.cluster_indices[index]
        inputs, labels = get_input_labels(
            idx,
            crop_id=crop_idx,
            data_dirpath=self.data_dir,
            labels_dirpath=self.labels_dir
        )
        return inputs, labels

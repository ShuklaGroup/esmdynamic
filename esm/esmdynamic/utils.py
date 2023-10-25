"""
Utils and constants for ESMDynamic model.
"""

import torch

rmsd_bin_boundaries = torch.tensor(
    [
        0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.,
        2.2, 2.4, 2.6, 2.8, 3., 3.2, 3.4, 3.6, 3.8, 4.,
        5., 6., 7., 8., 9., 10., 15., 20., 25., 30., 35.,
    ]
)

rmsd_vals = torch.zeros(len(rmsd_bin_boundaries) + 1)
rmsd_vals[0] = (0 + rmsd_bin_boundaries[0]) / 2
rmsd_vals[1:-1] = (rmsd_bin_boundaries[:-1] + rmsd_bin_boundaries[1:]) / 2  # Center of bin
rmsd_vals[-1] = rmsd_bin_boundaries[-1]

# def crop_pad_trunk_output(trunk_output, start_res, padded_length):
#     """Crop and pad a protein sequence that has been preprocessed through the ESMFold FoldingTrunk.
#     """
#     _, L, c_s = trunk_output['s_s'].shape  # Corresponds to sequence features
#     _, _, _, c_z = trunk_output['s_z'].shape  # Corresponds to pairwise features
#
#     padding = True if (L - start_res) < padded_length else False
#     end_res = start_res + padded_length
#
#     # Initialize tensors
#     s_s = torch.zeros((1, padded_length, c_s))
#     s_z = torch.zeros((1, padded_length, padded_length, c_z))
#     aatype = torch.zeros((1, padded_length)).long()
#     mask = torch.zeros((1, padded_length)).long()
#     residue_index = torch.zeros((1, padded_length)).long()
#     frames = torch.zeros((8, 1, padded_length, 7))
#     sidechain_frames = torch.zeros((8, 1, padded_length, 8, 4, 4))
#     unnormalized_angles = torch.zeros((8, 1, padded_length, 7, 2))
#     angles = torch.zeros((8, 1, padded_length, 7, 2))
#     positions = torch.zeros((8, 1, padded_length, 14, 3))
#     states = torch.zeros((8, 1, padded_length, 384))
#     single = torch.zeros((1, padded_length, 384))
#
#     if padding:
#         s_s[:, :L, :] = trunk_output['s_s'][:, start_res:, :]
#         s_z[:, :L, :L, :] = trunk_output['s_z'][:, start_res:, start_res:, :]
#         aatype[:, :L] = trunk_output['aatype'][:, start_res:]
#         mask[:, :L] = trunk_output['mask'][:, start_res:]
#         residue_index[:, :L] = trunk_output['residue_index'][:, start_res:]
#         frames[:, :, :L, :] = trunk_output['frames'][:, :, start_res:, :]
#         sidechain_frames[:, :, :L, :, :, :] = trunk_output['sidechain_frames'][:, :, start_res:, :, :, :]
#         unnormalized_angles[:, :, :L, :, :] = trunk_output['unnormalized_angles'][:, :, start_res:, :, :]
#         angles[:, :, :L, :, :] = trunk_output['angles'][:, :, start_res:, :, :]
#         positions[:, :, :L, :, :] = trunk_output['positions'][:, :, start_res:, :, :]
#         states[:, :, :L, :] = trunk_output['states'][:, :, start_res:, :]
#         single[:, :L, :] = trunk_output['single'][:, start_res:, :]
#     else:
#         s_s = trunk_output['s_s'][:, start_res:end_res, :]
#         s_z = trunk_output['s_z'][:, start_res:end_res, start_res:end_res, :]
#         aatype = trunk_output['aatype'][:, start_res:end_res]
#         mask = trunk_output['mask'][:, start_res:end_res]
#         residue_index = trunk_output['residue_index'][:, start_res:end_res]
#         frames = trunk_output['frames'][:, :, start_res:end_res, :]
#         sidechain_frames = trunk_output['sidechain_frames'][:, :, start_res:end_res, :, :, :]
#         unnormalized_angles = trunk_output['unnormalized_angles'][:, :, start_res:end_res, :, :]
#         angles = trunk_output['angles'][:, :, start_res:end_res, :, :]
#         positions = trunk_output['positions'][:, :, start_res:end_res, :, :]
#         states = trunk_output['states'][:, :, start_res:end_res, :]
#         single = trunk_output['single'][:, start_res:end_res, :]
#
#     return dict(
#         s_s=s_s,
#         s_z=s_z,
#         aatype=aatype,
#         mask=mask,
#         residue_index=residue_index,
#         frames=frames,
#         sidechain_frames=sidechain_frames,
#         unnormalized_angles=unnormalized_angles,
#         angles=angles,
#         positions=positions,
#         states=states,
#         single=single,
#     )

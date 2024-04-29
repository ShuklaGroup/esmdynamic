"""
Exclusively convolutional model for dynamic contact identification.
"""

import typing as T

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from openfold.data.data_transforms import make_atom14_masks
from openfold.utils.loss import compute_predicted_aligned_error, compute_tm
from einops.layers.torch import Rearrange

import esm
from esm.esmfold.v1.categorical_mixture import categorical_lddt
from esm.esmfold.v1.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
)

from .resnet import SymmetricResNet, ResNetConfig


class ConvNet(nn.Module):
    def __init__(self, cfg=None, load_esmfold=True, esmfold_config=None, **kwargs):
        super().__init__()

        self.register_buffer('dummy_buffer', torch.zeros(1))

        #  Load ESMFold
        self.load_esmfold = load_esmfold
        if self.load_esmfold is True:
            self.esmfold = esm.pretrained.esmfold_v1()
            self.esmfold.requires_grad_(False)

        # Define some handy constants in case ESMFold model is not loaded
        self.esmfold_distogram_bins = 64
        self.esmfold_cfg_trunk_pairwise_state_dim = 128

        # This layer creates a bias term for the pair representation from the output of the distogram head and ptm head
        self.pair_transition_input_dim = 2 * self.esmfold_distogram_bins
        self.pair_transition = nn.Sequential(  #
            nn.LayerNorm(self.pair_transition_input_dim),
            nn.Linear(self.pair_transition_input_dim, self.esmfold_cfg_trunk_pairwise_state_dim),
            nn.Linear(self.esmfold_cfg_trunk_pairwise_state_dim, self.esmfold_cfg_trunk_pairwise_state_dim),
        )  # Output dimensions must match c_z

        # Change from B x L x L x C -> B x C x L x L
        self.rearrange_pair = Rearrange('b l1 l2 c -> b c l1 l2')
        self.rearrange_pair_reverse = Rearrange('b c l1 l2 -> b l1 l2 c')

        # ResNet for dynamic contact prediction
        self.cfg = cfg
        if self.cfg is None:
            self.cfg = ResNetConfig()
        self.resnet_dynamic_contacts = SymmetricResNet(**self.cfg)

    def set_chunk_size(self, chunk_size: T.Optional[int]):
        if self.load_esmfold is True:
            self.esmfold.set_chunk_size(chunk_size)

    def forward(
            self,
            aa: T.Optional[torch.Tensor] = None,  # Must be provided unless using precomputed output
            mask: T.Optional[torch.Tensor] = None,
            residx: T.Optional[torch.Tensor] = None,
            masking_pattern: T.Optional[torch.Tensor] = None,
            num_recycles: T.Optional[int] = None,
            precomputed: T.Optional[dict] = None,  # ESMFold output --> Only used at training time
    ):

        if self.load_esmfold is False:
            try:
                assert(precomputed is not None)
            except AssertionError:
                raise RuntimeError("If load_esmfold=False, the model can only be called with precomputed input.")

        with torch.no_grad():
            if precomputed is None:
                structure = self.esmfold(aa, mask, residx, masking_pattern, num_recycles)
            if precomputed:
                structure = precomputed
                # structure = self._structure_from_trunk_output(structure) --> No longer necessary

        # Combine ptm_logits and distogram_logits to bias s_z
        ptm_logits = structure['ptm_logits']
        distogram_logits = structure['distogram_logits']
        pair_transition_input = torch.cat((ptm_logits, distogram_logits), dim=3)  # Concatenate along dim dimension
        s_z_0 = structure['s_z'] + self.pair_transition(pair_transition_input)

        # Predict dynamic contact probability from pair features
        dynamic_contact_logits = self.resnet_dynamic_contacts(self.rearrange_pair(dynamic_module_output['s_z']))
        structure['dynamic_contact_logits'] = dynamic_contact_logits
        dynamic_contact_prob = torch.sigmoid(dynamic_contact_logits)
        structure['dynamic_contact_prob'] = dynamic_contact_prob
        structure['dynamic_contact_pred'] = torch.where(structure['dynamic_contact_prob'] > 0.5, 1, 0).long()  # 0.5 threshold

        return structure

    @property
    def device(self):
        return self.dummy_buffer.device

        
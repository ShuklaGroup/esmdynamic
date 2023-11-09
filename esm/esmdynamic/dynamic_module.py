"""
DynamicModule based on ESMFold's FoldingTrunk.
"""

import typing as T
from dataclasses import dataclass
from contextlib import ExitStack

import torch
import torch.nn as nn
from esm.esmfold.v1.trunk import RelativePosition

# from openfold.model.structure_module import StructureModule

from esm.esmfold.v1.tri_self_attn_block import TriangularSelfAttentionBlock


@dataclass
class DynamicModuleConfig:
    _name: str = "DynamicModuleConfig"
    num_blocks: int = 2

    # Identical to FoldingTrunkConfig
    sequence_state_dim: int = 1024
    pairwise_state_dim: int = 128
    sequence_head_width: int = 32
    pairwise_head_width: int = 32
    position_bins: int = 32
    dropout: float = 0
    layer_drop: float = 0
    cpu_grad_checkpoint: bool = False

    max_recycles: int = 4
    chunk_size: T.Optional[int] = None


class DynamicModule(nn.Module):
    """
    Modified version of ESMFold's FoldingTrunk for dynamic contact prediction.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # Very similar to FoldingTrunk but omits unnecessary parts
        self.cfg = DynamicModuleConfig(**kwargs)
        assert self.cfg.max_recycles > 0

        c_s = self.cfg.sequence_state_dim
        c_z = self.cfg.pairwise_state_dim

        assert c_s % self.cfg.sequence_head_width == 0
        assert c_z % self.cfg.pairwise_head_width == 0
        block = TriangularSelfAttentionBlock
        self.pairwise_positional_embedding = RelativePosition(self.cfg.position_bins, c_z)
        self.blocks = nn.ModuleList(
            [
                block(
                    sequence_state_dim=c_s,
                    pairwise_state_dim=c_z,
                    sequence_head_width=self.cfg.sequence_head_width,
                    pairwise_head_width=self.cfg.pairwise_head_width,
                    dropout=self.cfg.dropout,
                )
                for i in range(self.cfg.num_blocks)
            ]
        )

        # Is recycling necessary for DynamicModule?
        self.recycle_bins = 15
        self.recycle_s_norm = nn.LayerNorm(c_s)
        self.recycle_z_norm = nn.LayerNorm(c_z)
        self.recycle_disto = nn.Embedding(self.recycle_bins, c_z)
        self.recycle_disto.weight[0].detach().zero_()

        self.chunk_size = self.cfg.chunk_size

    def set_chunk_size(self, chunk_size):
        self.chunk_size = chunk_size

    def forward(self, seq_feats, pair_feats, residx, mask, no_recycles: T.Optional[int] = None):
        """
        Inputs:
          seq_feats:     B x L x C            tensor of sequence features
          pair_feats:    B x L x L x C        tensor of pair features
          residx:        B x L                long tensor giving the position in the sequence
          mask:          B x L                boolean tensor indicating valid residues

        Output:
          dynamic_contacts_logits: B x L x L x C prediction tensor for dynamic contacts
        """
        device = seq_feats.device
        s_s_0 = seq_feats
        s_z_0 = pair_feats

        if no_recycles is None:
            no_recycles = self.cfg.max_recycles
        else:
            assert (no_recycles >= 0, "Number of recycles must not be negative.")
            no_recycles += 1  # First 'recycle' is just the standard forward pass through the model.

        def trunk_iter(s, z, residx, mask):
            z = z + self.pairwise_positional_embedding(residx, mask=mask)

            for block in self.blocks:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            return s, z

        s_s = s_s_0
        s_z = s_z_0
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

        assert no_recycles > 0
        for recycle_idx in range(no_recycles):
            with ExitStack() if recycle_idx == no_recycles - 1 else torch.no_grad():
                # === Recycling ===
                recycle_s = self.recycle_s_norm(recycle_s.detach())
                recycle_z = self.recycle_z_norm(recycle_z.detach())
                recycle_z += self.recycle_disto(recycle_bins.detach())
                s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask)
                recycle_s = s_s
                recycle_z = s_z

        return dict(s_s=s_s, s_z=s_z)

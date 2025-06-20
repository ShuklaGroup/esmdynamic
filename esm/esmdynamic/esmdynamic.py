f"""
ESMFold fine-tuning for dynamic contact prediction - Diego E. Kleiman (Shukla Group, UIUC).
"""
import dataclasses
import typing as T
from dataclasses import dataclass

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

from .utils import rmsd_vals
from .dynamic_module import DynamicModule, DynamicModuleConfig
from .resnet import SymmetricResNet, ResNetConfig
from .dilated_convnet import DilatedConvNet, DilatedConvNetConfig


@dataclass
class ESMDynamicConfig:
    dynamic_module: T.Any = DynamicModuleConfig()


class ESMDynamic(nn.Module):
    """Model for prediction of dynamic contact maps from protein sequences.

    Minimal usage:
    >>> from esm.pretrained import esmdynamic
    >>> model = esmdynamic()
    >>> prediction = model.predict_from_seqs(["SEQVENCE"]) # "dynamic_contact_prob" key contains dynamic contact maps
    """
    def __init__(self, load_esmfold=True, esmdynamic_config=None, esmfold_config=None, **kwargs):
        super().__init__()

        self.register_buffer('dummy_buffer', torch.zeros(1))

        esmdynamic_config = esmdynamic_config if esmdynamic_config else OmegaConf.structured(ESMDynamicConfig(**kwargs))
        self.cfg = esmdynamic_config

        #  Load ESMFold
        self.load_esmfold = load_esmfold
        if self.load_esmfold is True:
            self.esmfold = esm.pretrained.esmfold_v1()
            self.esmfold.requires_grad_(False)

        # Define some handy constants in case ESMFold model is not loaded
        self.esmfold_n_tokens_embed = 23
        self.esmfold_lddt_bins = 50
        self.esmfold_cfg_trunk_sequence_state_dim = 1024
        self.esmfold_distogram_bins = 64
        self.esmfold_cfg_trunk_pairwise_state_dim = 128

        # This layer creates a bias term for the sequence representation from the output of lddt_head and lm_logits
        self.seq_transition_input_dim = self.esmfold_n_tokens_embed + 37 * self.esmfold_lddt_bins
        self.seq_transition = nn.Sequential(
            nn.LayerNorm(self.seq_transition_input_dim),
            nn.Linear(self.seq_transition_input_dim, self.esmfold_cfg_trunk_sequence_state_dim),
            nn.Linear(self.esmfold_cfg_trunk_sequence_state_dim, self.esmfold_cfg_trunk_sequence_state_dim),
        )  # Output dimensions must match c_s

        # This layer creates a bias term for the pair representation from the output of the distogram head and ptm head
        self.pair_transition_input_dim = 2 * self.esmfold_distogram_bins
        self.pair_transition = nn.Sequential(  #
            nn.LayerNorm(self.pair_transition_input_dim),
            nn.Linear(self.pair_transition_input_dim, self.esmfold_cfg_trunk_pairwise_state_dim),
            nn.Linear(self.esmfold_cfg_trunk_pairwise_state_dim, self.esmfold_cfg_trunk_pairwise_state_dim),
        )  # Output dimensions must match c_z

        # DynamicModule based on evoformer block (from FoldingTrunk)
        self.dynamic_module = DynamicModule(**self.cfg.dynamic_module)

        # Linear module for final prediction
        self.prediction_layer = nn.Sequential(
            nn.LayerNorm(self.esmfold_cfg_trunk_pairwise_state_dim),
            nn.Linear(self.esmfold_cfg_trunk_pairwise_state_dim, 1)
            )

        self.rearrange_pair = Rearrange('b l1 l2 c -> b c l1 l2')

    def set_chunk_size(self, chunk_size: T.Optional[int]):
        if self.load_esmfold is True:
            self.esmfold.set_chunk_size(chunk_size)
        self.dynamic_module.set_chunk_size(chunk_size)

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

        if (mask is None) and (precomputed is None):
            mask = torch.ones_like(aa)
            structure['mask'] = mask
        elif (precomputed is None):
            structure['mask'] = mask

        # Combine output from lddt_head and lm_logits to bias s_s
        lddt_logits = structure['lddt_head'][-1]  # Use last state
        lddt_logits = lddt_logits.reshape(*lddt_logits.shape[:2],
                                          37 * self.esmfold_lddt_bins)  # Reshape into (B, L, 37 * self.lddt_bins)
        lm_logits = structure['lm_logits']  # Shape (B, L, self.n_tokens_embed)
        seq_transition_input = torch.cat((lddt_logits, lm_logits), dim=2)  # Concatenate along dim dimension
        s_s_0 = structure['s_s'] + self.seq_transition(seq_transition_input)

        # Combine ptm_logits and distogram_logits to bias s_z
        ptm_logits = structure['ptm_logits']
        distogram_logits = structure['distogram_logits']
        pair_transition_input = torch.cat((ptm_logits, distogram_logits), dim=3)  # Concatenate along dim dimension
        s_z_0 = structure['s_z'] + self.pair_transition(pair_transition_input)
        
        # Run through dynamic contact module
        dynamic_module_output = self.dynamic_module(s_s_0,
                                                    s_z_0,
                                                    structure['residue_index'],
                                                    structure['mask'],
                                                    no_recycles=num_recycles
                                                    )

        structure['dynamic_module_output'] = dynamic_module_output

        # Predict dynamic contact probability from pair features
        dynamic_contact_logits = self.rearrange_pair(self.prediction_layer(dynamic_module_output['s_z']))
        structure['dynamic_contact_logits'] = dynamic_contact_logits
        dynamic_contact_prob = torch.sigmoid(dynamic_contact_logits)
        structure['dynamic_contact_prob'] = (dynamic_contact_prob + torch.transpose(dynamic_contact_prob, 2, 3)) / 2 # Symmetrize output
        structure['dynamic_contact_pred'] = torch.where(structure['dynamic_contact_prob'] > 0.5, 1, 0).long()  # 0.5 threshold

        return structure

    def forward_from_seq(
            self, 
            sequences: T.Union[str, T.List[str]],
            residx: T.Optional[torch.Tensor] = None,
            masking_pattern: T.Optional[torch.Tensor] = None,
            num_recycles: T.Optional[int] = None,
            residue_index_offset: T.Optional[int] = 512,
            chain_linker: T.Optional[str] = "G" * 25
    ):
        """Feed example from sequence directly. Gradients are computed! Use self.predict_from_seqs() during inference.

        Args:
            sequences (Union[str, List[str]]): amino acid sequences.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the
                same size as `aa`.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
            residue_index_offset (int): Residue index separation between chains if predicting a multimer. Has no effect on
                single chain predictions. Default: 512.
            chain_linker (str): Linker to use between chains if predicting a multimer. Has no effect on single chain
                predictions. Default: length-25 poly-G ("G" * 25).

        Returns:
            structure (dict): dictionary containing all predictions.
        """
        
        if isinstance(sequences, str):
            sequences = [sequences]

        aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
            sequences, residue_index_offset, chain_linker
        )

        if residx is None:
            residx = _residx
        elif not isinstance(residx, torch.Tensor):
            residx = collate_dense_tensors(residx)

        aatype, mask, residx, linker_mask = map(
            lambda x: x.to(self.device), (aatype, mask, residx, linker_mask)
        )

        return self.forward(aa=aatype, mask=mask, residx=residx, masking_pattern=masking_pattern, num_recycles=num_recycles)

    @torch.no_grad()
    def predict_from_seqs(
            self, 
            sequences: T.Union[str, T.List[str]],
            residx: T.Optional[torch.Tensor] = None,
            masking_pattern: T.Optional[torch.Tensor] = None,
            num_recycles: T.Optional[int] = None,
            residue_index_offset: T.Optional[int] = 512,
            chain_linker: T.Optional[str] = "G" * 25
    ):
        """Predict from sequences directly. Gradient is not computed. Use for inference.

        Args:
            sequences (Union[str, List[str]]): amino acid sequences.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the
                same size as `aa`.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
            residue_index_offset (int): Residue index separation between chains if predicting a multimer. Has no effect on
                single chain predictions. Default: 512.
            chain_linker (str): Linker to use between chains if predicting a multimer. Has no effect on single chain
                predictions. Default: length-25 poly-G ("G" * 25).

        Returns:
            structure (dict): dictionary containing all predictions.
        """

        return self.forward_from_seq(
            sequences,
            residx,
            masking_pattern,
            num_recycles,
            residue_index_offset,
            chain_linker
        )
        
        

    @torch.no_grad()
    def _trunk_input_from_seqs(self, sequences: T.Union[str, T.List[str]],
                               filename: str = None,
                               residx: T.Optional[torch.Tensor] = None,
                               masking_pattern: T.Optional[torch.Tensor] = None,
                               num_recycles: T.Optional[int] = None,
                               residue_index_offset: T.Optional[int] = 512,
                               chain_linker: T.Optional[str] = "G" * 25,
                               ):
        """Computes the input for the FoldingTrunk from a batch of sequences.
        This allows to save time in the forward pass, since the language model (3B ESM-2, 48 folding blocks) doesn't
        need to be called for every batch of inputs. When cropping and batching intermediate inputs, zero-padding is
        used as opposed to a padding token.

        This function is based on ESMFold.infer() and ESMFold.forward().

        Args:
            sequences (Union[str, List[str]]): amino acid sequences.
            filename (str): output path/name. If provided, saves results here as a dictionary (.pkl file).
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the
                same size as `aa`.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
            residue_index_offset (int): Residue index separation between chains if predicting a multimer. Has no effect on
                single chain predictions. Default: 512.
            chain_linker (str): Linker to use between chains if predicting a multimer. Has no effect on single chain
                predictions. Default: length-25 poly-G ("G" * 25).
        """
        if self.load_esmfold is False:
            raise RuntimeError("The method _trunk_input_from_seqs cannot be called if ESMFold is not loaded.")

        if isinstance(sequences, str):
            sequences = [sequences]

        aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
            sequences, residue_index_offset, chain_linker
        )

        if residx is None:
            residx = _residx
        elif not isinstance(residx, torch.Tensor):
            residx = collate_dense_tensors(residx)

        aatype, mask, residx, linker_mask = map(
            lambda x: x.to(self.device), (aatype, mask, residx, linker_mask)
        )

        # if mask is None:
        #     mask = torch.ones_like(aa)

        aa = aatype
        B = aa.shape[0]
        L = aa.shape[1]
        device = aa.device

        # if residx is None:
        #     residx = torch.arange(L, device=device).expand_as(aa)

        # === ESM ===
        esmaa = self.esmfold._af2_idx_to_esm_idx(aa, mask)

        if masking_pattern is not None:
            esmaa = self.esmfold._mask_inputs_to_esm(esmaa, masking_pattern)

        esm_s = self.esmfold._compute_language_model_representations(esmaa)

        # Convert esm_s to the precision used by the trunk and
        # the structure module. These tensors may be a lower precision if, for example,
        # we're running the language model in fp16 precision.
        esm_s = esm_s.to(self.esmfold.esm_s_combine.dtype)

        esm_s = esm_s.detach()

        # === preprocessing ===
        esm_s = (self.esmfold.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)

        s_s_0 = self.esmfold.esm_s_mlp(esm_s)
        s_z_0 = s_s_0.new_zeros(B, L, L, self.esmfold.cfg.trunk.pairwise_state_dim)

        s_s_0 += self.esmfold.embedding(aa)

        results = dict(seq_feats=s_s_0,
                       pair_feats=s_z_0,  # It doesn't make sense to store a tensor of zeros,
                       # but the trunk requires the input
                       true_aa=aa,
                       residx=residx,
                       mask=mask,
                       no_recycles=num_recycles)

        if filename is not None:
            extension = '.pt'  # Enforce this extension
            if filename[-len(extension):] != extension:
                filename += extension

            torch.save(results, filename)

        return results

    @torch.no_grad()
    def _trunk_output_from_seqs(self, sequences: T.Union[str, T.List[str]], filename: str = None):
        """This function produces the FoldingTrunk output from ESMFold. It is useful to pre-compute the ESMFold output
        before training the DynamicModule.

        Args:
            sequences (Union[str, List[str]]): amino acid sequences.
            filename (str): output path/name. If provided, saves results here as a dictionary (.pkl file).
        """
        if self.load_esmfold is False:
            raise RuntimeError("The method _trunk_output_from_seqs cannot be called if ESMFold is not loaded.")

        trunk_arguments = self._trunk_input_from_seqs(sequences)
        results = self.esmfold.trunk(**trunk_arguments)

        # I'm adding additional keys because they're needed for downstream tasks
        results["aatype"] = trunk_arguments["true_aa"]
        results["residue_index"] = trunk_arguments["residx"]
        results["mask"] = trunk_arguments["mask"]

        if filename is not None:
            extension = '.pt'  # Enforce this extension
            if filename[-len(extension):] != extension:
                filename += extension

            torch.save(results, filename)

        return results

    @torch.no_grad()
    def _structure_from_trunk_output(self, structure: dict, pdb_path: T.Optional[str] = None):
        """Compute final predicted structure from trunk output. I will use this function to check effects of cropping
        and padding at a downstream portion of the network.

        This function is a slightly modified version of ESMFold.forward().

        Args:
            structure (dict): output of FoldingTrunk.
            pdb_path (str): output path/name. If provided, saves results here as a PDB file.
        """
        if self.load_esmfold is False:
            raise RuntimeError("The method _structure_from_trunk_output cannot be called if ESMFold is not loaded.")

        mask = structure["mask"]
        B, L = structure["aatype"].shape

        structure = {  # Added some extra keys here because they must be conserved
            k: v
            for k, v in structure.items()
            if k
               in [
                   "s_z",
                   "s_s",
                   "frames",
                   "sidechain_frames",
                   "unnormalized_angles",
                   "angles",
                   "positions",
                   "states",
                   "aatype",
                   "residue_index",
                   "mask",
               ]
        }

        disto_logits = self.esmfold.distogram_head(structure["s_z"])
        disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
        structure["distogram_logits"] = disto_logits

        lm_logits = self.esmfold.lm_head(structure["s_s"])
        structure["lm_logits"] = lm_logits

        # structure["aatype"] = aa
        make_atom14_masks(structure)

        for k in [
            "atom14_atom_exists",
            "atom37_atom_exists",
        ]:
            structure[k] *= mask.unsqueeze(-1)

        lddt_head = self.esmfold.lddt_head(structure["states"]).reshape(
            structure["states"].shape[0], B, L, -1, self.esmfold.lddt_bins
        )
        structure["lddt_head"] = lddt_head
        plddt = categorical_lddt(lddt_head[-1], bins=self.esmfold.lddt_bins)
        structure["plddt"] = (
                100 * plddt
        )  # we predict plDDT between 0 and 1, scale to be between 0 and 100.

        ptm_logits = self.esmfold.ptm_head(structure["s_z"])

        seqlen = mask.type(torch.int64).sum(1)
        structure["ptm_logits"] = ptm_logits
        structure["ptm"] = torch.stack(
            [
                compute_tm(
                    batch_ptm_logits[None, :sl, :sl], max_bins=31, no_bins=self.esmfold.distogram_bins
                )
                for batch_ptm_logits, sl in zip(ptm_logits, seqlen)
            ]
        )
        structure.update(
            compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=self.esmfold.distogram_bins)
        )

        if pdb_path is not None:
            extension = '.pdb'  # Enforce this extension
            if pdb_path[-len(extension):] != extension:
                pdb_path += extension

            for i, pdb in enumerate(self.esmfold.output_to_pdb(structure)):
                pdb_fname = pdb_path[:-len(extension)] + "_" + str(i) + pdb_path[-len(extension):]
                with open(pdb_fname, 'w') as outfile:
                    outfile.write(pdb)

        return structure

    @property
    def device(self):
        return self.dummy_buffer.device
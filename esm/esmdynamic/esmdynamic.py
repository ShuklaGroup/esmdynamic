f"""
ESMFold fine-tuning for dynamic contact prediction - Diego E. Kleiman (Shukla Group, UIUC).
"""
import dataclasses
import typing as T
from dataclasses import dataclass, field
import tempfile

import numpy as np
import torch
import torch.nn as nn
import mdtraj as md
from omegaconf import OmegaConf

import esm
from esm.esmfold.v1.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
)

from .dynamic_module import DynamicModule, DynamicModuleConfig


@dataclass
class ESMDynamicConfig:
    dynamic_module: T.Any = field(default_factory=DynamicModuleConfig) # Predict dynamic contacts
    kinetic_module: T.Any = field(default_factory=DynamicModuleConfig) # Predict kinetics for contacts
    frequency_module: T.Any = field(default_factory=DynamicModuleConfig) # Predict contact frequency



class DynamicHead(nn.Module):
    """
    Unified head supporting:
      - task_type: "classification", "regression", "multiclass", or "kinetics"
      - multi-temperature outputs (n_conditions)
      - multiclass via n_classes (for 'multiclass' or 'kinetics')
      - classification confidence head (per-residue, per-temp) -> [B, n_conditions, L]
      - regression pairwise residual head -> [B, n_conditions, L, L]
    For kinetics:
      - outputs logits shaped [B, L, L, n_conditions, n_classes, 2] (on/off)
      - confidence_head predicts per-residue per-temp accuracy averaged across on/off: [B, n_conditions, L]
    """
    def __init__(
        self,
        name: str,
        task_type: str,
        seq_input_dim: int,
        seq_state_dim: int,
        pair_input_dim: int,
        pair_state_dim: int,
        dynamic_cfg,
        n_conditions: int = 5,
        n_classes: T.Optional[int] = None,
        use_confidence_head: bool = False,
        use_residual_head: bool = False,
    ):
        super().__init__()
        self.name = name
        self.task_type = task_type
        self.n_conditions = n_conditions
        self.n_classes = n_classes
        self.use_confidence_head = use_confidence_head
        self.use_residual_head = use_residual_head

        # --- Transitions (bias terms) ---
        self.seq_transition = nn.Sequential(
            nn.LayerNorm(seq_input_dim),
            nn.Linear(seq_input_dim, seq_state_dim),
            nn.Linear(seq_state_dim, seq_state_dim),
        )
        self.pair_transition = nn.Sequential(
            nn.LayerNorm(pair_input_dim),
            nn.Linear(pair_input_dim, pair_state_dim),
            nn.Linear(pair_state_dim, pair_state_dim),
        )

        # --- Dynamic module ---
        self.dynamic_module = DynamicModule(**dynamic_cfg)

        # --- Main prediction linear: determine out_dim ---
        # For kinetics: out_dim = n_conditions * n_classes * n_rates (n_rates == 2 for on/off)
        if self.task_type == "kinetics":
            assert n_classes is not None and n_classes >= 1, "n_classes must be set for kinetics"
            self.n_rates = 2
            out_dim = n_conditions * n_classes * self.n_rates
        elif self.task_type == "multiclass":
            assert n_classes is not None and n_classes >= 2, "n_classes must be >=2 for multiclass"
            out_dim = n_conditions * n_classes
        else:
            # binary classification or regression -> n_conditions outputs per pair
            out_dim = n_conditions

        self.prediction_linear = nn.Linear(pair_state_dim, out_dim)

        # --- Confidence head for classification or kinetics: per-residue per-temp -> [B, n_conditions, L]
        if use_confidence_head and self.task_type in ("classification", "multiclass", "kinetics"):
            self.confidence_head = nn.Sequential(
                nn.LayerNorm(seq_state_dim),
                nn.Linear(seq_state_dim, seq_state_dim // 2),
                nn.ReLU(),
                nn.Linear(seq_state_dim // 2, n_conditions),  # one value per temperature
            )
        else:
            self.confidence_head = None

        # --- Regression residual head (pairwise) ---
        if use_residual_head and self.task_type == "regression":
            # maps pair_state_dim -> n_conditions residuals per pair
            self.residual_head = nn.Sequential(
                nn.LayerNorm(pair_state_dim),
                nn.Linear(pair_state_dim, pair_state_dim // 2),
                nn.ReLU(),
                nn.Linear(pair_state_dim // 2, n_conditions),  # one value per temperature
            )
        else:
            self.residual_head = None

    def forward(self, structure: dict, num_recycles: T.Optional[int] = None):
        # --- Build bias inputs (same procedure as before) ---
        lddt_logits = structure["lddt_head"][-1]
        lddt_logits = lddt_logits.reshape(*lddt_logits.shape[:2], 37 * lddt_logits.shape[-1])
        lm_logits = structure["lm_logits"]
        seq_transition_input = torch.cat((lddt_logits, lm_logits), dim=2)
        s_s_0 = structure["s_s"] + self.seq_transition(seq_transition_input)

        ptm_logits = structure["ptm_logits"]
        distogram_logits = structure["distogram_logits"]
        pair_transition_input = torch.cat((ptm_logits, distogram_logits), dim=3)
        s_z_0 = structure["s_z"] + self.pair_transition(pair_transition_input)

        # --- Dynamic module ---
        dynamic_out = self.dynamic_module(
            s_s_0, s_z_0, structure["residue_index"], structure["mask"], no_recycles=num_recycles
        )

        # pairwise features from dynamic module: [B, L, L, Cz]
        pair_feats = dynamic_out["s_z"]
        B, L1, L2, Cz = pair_feats.shape

        # --- Main prediction linear and reshape depending on task ---
        pred_raw = self.prediction_linear(pair_feats)  # [B, L, L, out_dim]

        if self.task_type == "kinetics":
            # pred_raw: [B, L, L, n_conditions * n_classes * n_rates]
            pred = pred_raw.view(B, L1, L2, self.n_conditions, self.n_classes, self.n_rates)
            # canonical order: [B, n_conditions, n_rates, L, L, n_classes]
            pred = pred.permute(0, 3, 5, 1, 2, 4).contiguous()
            structure[f"{self.name}_logits"] = pred

            # Compute probabilities (softmax over classes)
            probs = torch.softmax(pred, dim=-1)
            # symmetrize across residue pairs (axes 3 and 4 are L, L)
            probs = (probs + probs.transpose(3, 4)) / 2
            structure[f"{self.name}_prob"] = probs

            # Predicted class index for each condition & rate
            structure[f"{self.name}_pred_class"] = probs.argmax(dim=-1)  # [B, n_conditions, n_rates, L, L]

        elif self.task_type == "multiclass":
            # pred_raw: [B, L, L, n_conditions * n_classes]
            pred = pred_raw.view(B, L1, L2, self.n_conditions, self.n_classes)
            # canonical: [B, n_conditions, L, L, n_classes]
            pred = pred.permute(0, 3, 1, 2, 4).contiguous()
            structure[f"{self.name}_logits"] = pred

            probs = torch.softmax(pred, dim=-1)
            # symmetrize across residue pairs (axes 2 and 3 are L, L)
            probs = (probs + probs.transpose(2, 3)) / 2
            structure[f"{self.name}_prob"] = probs
            structure[f"{self.name}_pred_class"] = probs.argmax(dim=-1)  # [B, n_conditions, L, L]

        else:
            # Regression or binary classification
            pred = pred_raw.view(B, L1, L2, self.n_conditions)
            # canonical: [B, n_conditions, L, L]
            pred = pred.permute(0, 3, 1, 2).contiguous()

            if self.task_type == "classification":
                prob = torch.sigmoid(pred)
                # symmetrize across residue pairs (axes 2 and 3 are L, L)
                prob = (prob + prob.transpose(2, 3)) / 2
                structure[f"{self.name}_logits"] = pred
                structure[f"{self.name}_prob"] = prob
                structure[f"{self.name}_pred"] = (prob > 0.5).long()
            else:  # regression
                pred_clipped = torch.sigmoid((pred + pred.transpose(2, 3)) / 2)
                structure[f"{self.name}_value"] = pred
                structure[f"{self.name}_pred"] = pred_clipped

        # --- Confidence head ---
        if self.confidence_head is not None:
            s_s = dynamic_out["s_s"]  # [B, L, seq_state_dim]
            conf_raw = self.confidence_head(s_s)  # [B, L, n_conditions]
            # canonical: [B, n_conditions, L]
            conf = conf_raw.permute(0, 2, 1).contiguous()
            structure[f"{self.name}_confidence"] = conf

        # --- Residual head (regression) ---
        if self.residual_head is not None:
            res_raw = self.residual_head(pair_feats)  # [B, L, L, n_conditions]
            res_sym = (res_raw + res_raw.transpose(1, 2)) / 2
            # canonical: [B, n_conditions, L, L]
            res = res_sym.permute(0, 3, 1, 2).contiguous()
            structure[f"{self.name}_residual_pred"] = res


        structure[f"{self.name}_output"] = dynamic_out
        return structure



class ESMDynamic(nn.Module):
    """Model for prediction of dynamic contact maps from protein sequences.

    Minimal usage:
    >>> from esm.pretrained import esmdynamic
    >>> model = esmdynamic(heads_to_load=["dynamic", "kinetic"]) # Can also select "frequency"
    >>> prediction = model.predict_from_seqs(["SEQVENCE"]) # "dynamic_contact_prob" key contains dynamic contact maps
    """
    def __init__(
        self,
        load_esmfold=True,
        esmdynamic_config=None,
        esmfold_config=None,
        head_definitions=None,
        heads_to_load=None,
        **kwargs,
    ):
        super().__init__()
        self.register_buffer("dummy_buffer", torch.zeros(1))
        esmdynamic_config = esmdynamic_config or OmegaConf.structured(ESMDynamicConfig(**kwargs))
        self.cfg = esmdynamic_config

        # --- ESMFold ---
        self.load_esmfold = load_esmfold
        if self.load_esmfold:
            self.esmfold = esm.pretrained.esmfold_v1()
            self.esmfold.requires_grad_(False)

        # --- Dim constants ---
        self.esmfold_n_tokens_embed = 23
        self.esmfold_lddt_bins = 50
        self.esmfold_cfg_trunk_sequence_state_dim = 1024
        self.esmfold_distogram_bins = 64
        self.esmfold_cfg_trunk_pairwise_state_dim = 128

        seq_input_dim = self.esmfold_n_tokens_embed + 37 * self.esmfold_lddt_bins
        pair_input_dim = 2 * self.esmfold_distogram_bins


        # --- Check for mutually exclusive configuration ---
        if head_definitions is not None and heads_to_load is not None:
            raise ValueError(
                "Arguments 'head_definitions' and 'heads_to_load' are mutually exclusive. "
                "Use 'head_definitions' to define custom heads OR 'heads_to_load' to select from defaults."
            )

        # --- Define heads with per-head configs ---
        default_head_definitions = [
            dict(
                name="dynamic",
                task_type="classification",
                n_conditions=5,
                n_classes=None,
                dynamic_cfg=self.cfg.dynamic_module,
                use_confidence_head=True,   # per-residue per-temp confidence
                use_residual_head=False
            ),
            dict(
                name="kinetic",
                task_type="kinetics",
                n_conditions=5,
                n_classes=6,               # {always observed, 1-10 ns, 10-100 ns, 100-300 ns, 300+ ns, never observed}
                dynamic_cfg=self.cfg.kinetic_module,
                use_confidence_head=True,  # per-residue per-temp accuracy averaged across on/off
                use_residual_head=False
            ),
            dict(
                name="frequency",
                task_type="regression",
                n_conditions=5,
                n_classes=None,
                dynamic_cfg=self.cfg.frequency_module,
                use_confidence_head=False,
                use_residual_head=True     # residuals predicted pairwise for frequency
            ),
        ]

        # --- Determine which heads to instantiate ---
        if head_definitions is not None:
            selected_heads = head_definitions
        else:
            if heads_to_load is not None:
                heads_to_load = set(heads_to_load)
                selected_heads = [
                    hd for hd in default_head_definitions if hd["name"] in heads_to_load
                ]
            else:
                selected_heads = default_head_definitions

        # --- Build head modules ---
        self.heads = nn.ModuleDict()
        for hd in selected_heads:
            name = hd["name"]
            task_type = hd["task_type"]
            n_conditions = hd.get("n_conditions", 5)
            n_classes = hd.get("n_classes", None)
            dynamic_cfg = hd.get("dynamic_cfg", self.cfg.dynamic_module)
            use_conf = hd.get("use_confidence_head", False)
            use_res = hd.get("use_residual_head", False)

            self.heads[name] = DynamicHead(
                name=name,
                task_type=task_type,
                seq_input_dim=seq_input_dim,
                seq_state_dim=self.esmfold_cfg_trunk_sequence_state_dim,
                pair_input_dim=pair_input_dim,
                pair_state_dim=self.esmfold_cfg_trunk_pairwise_state_dim,
                dynamic_cfg=dynamic_cfg,
                n_conditions=n_conditions,
                n_classes=n_classes,
                use_confidence_head=use_conf,
                use_residual_head=use_res,
            )

    def set_chunk_size(self, chunk_size: T.Optional[int]):
        if self.load_esmfold is True:
            self.esmfold.set_chunk_size(chunk_size)
        for head in self.heads.values():
            if hasattr(head.dynamic_module, "set_chunk_size"):
                head.dynamic_module.set_chunk_size(chunk_size)

    def forward(
            self,
            aa: T.Optional[torch.Tensor] = None,  # Must be provided unless using precomputed output
            mask: T.Optional[torch.Tensor] = None,
            residx: T.Optional[torch.Tensor] = None,
            masking_pattern: T.Optional[torch.Tensor] = None,
            num_recycles: T.Optional[int] = None,
            precomputed: T.Optional[dict] = None,  # ESMFold output --> Only used at training time
    ):

        if not self.load_esmfold and precomputed is None:
            raise RuntimeError("If load_esmfold=False, must call with precomputed structure.")

        with torch.no_grad():
            structure = (
                precomputed if precomputed is not None
                else self.esmfold(aa, mask, residx, masking_pattern, num_recycles)
            )

        if mask is None and precomputed is None:
            mask = torch.ones_like(aa)
        structure["mask"] = mask

        for head in self.heads.values():
            structure = head(structure, num_recycles=num_recycles)

        # Get native contacts from ESMFold and find the set "dynamic - native" and "native - dynamic"
        if "dynamic" in self.heads:

            struct_cpu = {
                k: (
                    v.float().cpu()
                    if v.is_floating_point()
                    else v.cpu()
                )
                for k, v in structure.items()
                if isinstance(v, torch.Tensor)
            }

            structure["pdbs"] = self.esmfold.output_to_pdb(struct_cpu)

            native_contacts_list = self.compute_native_contacts(structure["pdbs"])

            dynamic_pred = structure["dynamic_pred"] 
            B, N, L_dyn, _ = dynamic_pred.shape

            native_contacts = torch.zeros(
                (B, L_dyn, L_dyn),
                dtype=torch.long,
                device=self.device,
            )

            for b, native in enumerate(native_contacts_list):
                L_nat = native.shape[0]

                if L_nat > L_dyn:
                    raise ValueError(
                        f"Native contact map ({L_nat}) larger than dynamic L ({L_dyn})."
                    )

                native_contacts[b, :L_nat, :L_nat] = native

            structure["native_contacts"] = native_contacts

            # Broadcast across N
            native_expanded = native_contacts.unsqueeze(1)

            # dynamic AND NOT native
            structure["dynamic_nonnative_contacts"] = (
                dynamic_pred * (1 - native_expanded)
            )

            # native AND NOT dynamic
            structure["native_nondynamic_contacts"] = (
                native_expanded * (1 - dynamic_pred)
            )

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

    def forward_from_seq_low_memory(
            self, 
            sequences: T.Union[str, T.List[str]],
            residx: T.Optional[torch.Tensor] = None,
            masking_pattern: T.Optional[torch.Tensor] = None,
            num_recycles: T.Optional[int] = None,
            residue_index_offset: T.Optional[int] = 512,
            chain_linker: T.Optional[str] = "G" * 25
    ):
        """Feed example from sequence directly.
        This low memory implementation loads the model's modules 'just in time' and then offloads them.
        Cannot be used during training.

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

        # Free gpu
        if self.device == "cuda":
            self.to("cpu")
            torch.cuda.empty_cache()

        # Compute ESMFold output
        self.esmfold.to(self.device)

        trunk_out = self.esmfold(aatype, mask, residx, masking_pattern, num_recycles)
        trunk_detached = {k: v.detach().cpu() for k, v in trunk_out.items()}
        trunk_detached["mask"] = mask.detach().cpu()

        self.esmfold.to("cpu")
        del trunk_out
        torch.cuda.empty_cache()

        combined_outputs = {}
        combined_outputs.update(trunk_detached)
        for head_name, head in self.heads.items():
            # Move head to device
            head.to(self.device)
            # Move trunk to same device
            trunk_device = {k: v.to(self.device) for k, v in trunk_detached.items()}

            # Forward pass for this head
            head_out = head(trunk_device, num_recycles=num_recycles)
            combined_outputs.update({
                k: (
                    {kk: vv.detach().cpu() for kk, vv in v.items()}  # if v is a dict of tensors
                    if isinstance(v, dict)
                    else v.detach().cpu()                            # if v is a tensor
                )
                for k, v in head_out.items()
            })

            # Free memory
            head.to("cpu")
            del trunk_device
            del head_out
            torch.cuda.empty_cache()

        self.to(self.device)

        return combined_outputs

    @torch.no_grad()
    def predict_from_seqs(
            self, 
            sequences: T.Union[str, T.List[str]],
            low_memory: bool = None,
            residx: T.Optional[torch.Tensor] = None,
            masking_pattern: T.Optional[torch.Tensor] = None,
            num_recycles: T.Optional[int] = None,
            residue_index_offset: T.Optional[int] = 512,
            chain_linker: T.Optional[str] = "G" * 25
    ):
        """Predict from sequences directly. Gradient is not computed. Use for inference.

        Args:
            sequences (Union[str, List[str]]): amino acid sequences.
            low_memory (bool): use low memory forward evaluation (slower).
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

        if low_memory:
            return self.forward_from_seq_low_memory(
                sequences,
                residx,
                masking_pattern,
                num_recycles,
                residue_index_offset,
                chain_linker
            )
        else:
            return self.forward_from_seq(
                sequences,
                residx,
                masking_pattern,
                num_recycles,
                residue_index_offset,
                chain_linker
            )
            
    @property
    def device(self):
        return self.dummy_buffer.device


    def compute_native_contacts(self, pdb_strings, threshold=8.0):

        contact_maps = []

        for pdb_str in pdb_strings:

            with tempfile.NamedTemporaryFile(mode="w+", suffix=".pdb") as tmp:
                tmp.write(pdb_str)
                tmp.flush()
                traj = md.load(tmp.name)

            distances_nm, residue_pairs = md.compute_contacts(
                traj,
                contacts="all",
                scheme="ca",
            )

            distances_angstrom = distances_nm[0] * 10.0 # to angstroms
            n_residues = traj.n_residues

            contact_matrix = np.zeros((n_residues, n_residues), dtype=np.int64)

            contact_mask = distances_angstrom < threshold
            contacting_pairs = residue_pairs[contact_mask]

            contact_matrix[contacting_pairs[:, 0], contacting_pairs[:, 1]] = 1
            contact_matrix[contacting_pairs[:, 1], contacting_pairs[:, 0]] = 1

            np.fill_diagonal(contact_matrix, 0)

            contact_maps.append(
                torch.tensor(contact_matrix, dtype=torch.int64, device=self.device)
            )

        return contact_maps
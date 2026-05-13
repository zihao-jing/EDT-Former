"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch
import torch.nn as nn
import logging
from torch.cuda.amp import autocast as autocast
from peft import get_peft_model, LoraConfig, TaskType

from models.configuration import MolLLaMAConfig
from models.edt_former_encoder import EDTFormerEncoder
from models.mol_llama_encoder import MolLLaMAEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, GenerationMixin, BitsAndBytesConfig, LlamaForCausalLM

from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel
from torch_geometric.data import Data, Batch
from data_provider.mol_dataset import smiles2graph, get_unimol_data
from data_provider.collaters import Mol3DCollater
import numpy as np
from safetensors.torch import load_file as load_safetensors
from pathlib import Path
import json
import glob

logger = logging.getLogger(__name__)
# Set to ERROR level to suppress warnings (INFO < WARNING < ERROR < CRITICAL)
logger.setLevel(logging.ERROR)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def unlock_new_token_embeddings(embedding_layer, new_token_ids, init="mean"):
    """
    Unfreeze the rows in embedding_layer corresponding to new_token_ids,
    and optionally initialize the embedding vectors for those tokens.

    Args:
        embedding_layer (nn.Embedding): from self.llm.get_input_embeddings()
        new_token_ids (List[int]): List of new token IDs to unfreeze
        init (str or None): Initialization strategy. Options:
            - "mean": initialize with the mean of the original vocabulary
            - "zero": initialize to 0 vector
            - None: do not initialize (keep default random)
    """
    # 1. Freeze all embeddings
    embedding_layer.weight.requires_grad = False

    # 2. Optionally initialize the new rows
    with torch.no_grad():
        if init == "mean":
            old_vocab_size = embedding_layer.weight.shape[0] - len(new_token_ids)
            avg_vec = embedding_layer.weight[:old_vocab_size].mean(dim=0)
            for idx in new_token_ids:
                embedding_layer.weight[idx].copy_(avg_vec)
        elif init == "zero":
            for idx in new_token_ids:
                embedding_layer.weight[idx].zero_()

    # 3. Unfreeze only the new rows
    for idx in new_token_ids:
        embedding_layer.weight[idx].requires_grad = True

    logger.info(f"Unfrozen {len(new_token_ids)} tokens: {new_token_ids}")


class MolLLaMAPreTrainedModel(PreTrainedModel):
    config_class = MolLLaMAConfig
    base_model_prefix = 'mllm'
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"encoder.graph_encoder",
        r"llm."
    ]

class EDTFormer(MolLLaMAPreTrainedModel):
    def __init__(
        self,
        config: MolLLaMAConfig,
        vocab_size=None,
        torch_dtype="float16",
        enable_flash=True,
        add_ids=None,
        local_q_only=False,
        freeze_llm=False,
        brics_gids_enable=False,
        entropy_gids_enable=False,
        enable_blending=False,
        load_ckpt_before_peft=False,
        ckpt_path=None,
        llm_only=False,  # New parameter: skip encoder for text-only tasks
        global_q_budget=None,
        local_q_budget=None,
    ):
        super().__init__(config)
        self.config = config
        self.llm_only = llm_only
        
        ## Initialize encoder (skip if llm_only mode)
        if not llm_only:
            if enable_blending:
                config.graph_encoder_config.encoder_types = ['unimol', 'moleculestm']
            self.num_query_tokens = config.qformer_config.num_query_tokens
            self.encoder = EDTFormerEncoder(
                graph_encoder_config = config.graph_encoder_config,
                blending_module_config = config.blending_module_config,
                qformer_config = config.qformer_config,
                brics_gids_enable = brics_gids_enable,
                entropy_gids_enable = entropy_gids_enable,
                enable_blending = enable_blending,
                global_q_budget=global_q_budget,
                local_q_budget=local_q_budget,
            )
            self.local_q_only = local_q_only
            self.brics_gids_enable = brics_gids_enable
            self.entropy_gids_enable = entropy_gids_enable
            self.postprocess_encoder()
            logger.info("✅ Encoder initialized for molecule-text tasks")
        else:
            # LLM-only mode: no encoder needed
            self.encoder = None
            self.num_query_tokens = None
            self.local_q_only = False
            self.brics_gids_enable = False
            self.entropy_gids_enable = False
            logger.info("🚀 LLM-only mode: Skipping encoder initialization (saves ~20-25 GB GPU memory)")
        ## Initialize LLM
        if torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32

        # -------------------------- train llm ----------------------------------
        if not freeze_llm:
            logger.info(f"Loading LLM model: {config.llm_config.llm_model}")
            if enable_flash:
                try:
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        config.llm_config.llm_model,
                        torch_dtype=torch_dtype,
                        attn_implementation="flash_attention_2",
                    )
                    logger.info("Using flash attention")
                except TypeError:
                    # Some architectures may not accept attn_implementation
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        config.llm_config.llm_model,
                        torch_dtype=torch_dtype,
                    )
            else:
                self.llm = AutoModelForCausalLM.from_pretrained(
                    config.llm_config.llm_model,
                    torch_dtype=torch_dtype,
                )
            self.llm.resize_token_embeddings(vocab_size)
            
            # Create llm_proj BEFORE loading checkpoint so it can be loaded properly
            # Skip if llm_only mode (no encoder)
            if not llm_only:
                self.llm_proj = nn.Linear(self.encoder.Qformer.config.hidden_size, 
                                            self.llm.config.hidden_size)
            else:
                self.llm_proj = None
            
            # Load checkpoint before PEFT if requested
            if load_ckpt_before_peft and ckpt_path:
                logger.info(f"🔧 Loading checkpoint BEFORE PEFT model creation: {ckpt_path}")
                self._load_checkpoint_before_peft(ckpt_path)
            
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                        inference_mode=False,
                                        r=config.llm_config.lora_config.r,
                                        lora_alpha=config.llm_config.lora_config.lora_alpha,
                                        lora_dropout=config.llm_config.lora_config.lora_dropout,
                                        target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
            self.peft_config = peft_config
            
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()

        # -------------------------- frozen llm ----------------------------------

        else:
            logger.info(f"Loading LLM model: {config.llm_config.llm_model}")
            self.llm = LlamaForCausalLM.from_pretrained(
                config.llm_config.llm_model,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2" if enable_flash else None,
            )

            self.llm.resize_token_embeddings(vocab_size)

            # Freeze LLM parameters
            self.llm.eval()
            for p in self.llm.parameters():
                p.requires_grad = False

            if add_ids is not None:
                embed = self.llm.get_input_embeddings()
                unlock_new_token_embeddings(embed, add_ids, init="mean")
            
            # Create llm_proj for frozen LLM case too
            # Skip if llm_only mode (no encoder)
            if not llm_only:
                self.llm_proj = nn.Linear(self.encoder.Qformer.config.hidden_size, 
                                            self.llm.config.hidden_size)
            else:
                self.llm_proj = None

    def postprocess_encoder(self):
        self.encoder.Qformer.cls = None
        self.encoder.Qformer.bert.embeddings.word_embeddings = None
        self.encoder.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.encoder.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.encoder.graph_proj = None
        self.encoder.text_proj = None
        self.encoder.gtm_head = None

    def inject_queries_multi_molecule(
        self,
        query_output: torch.Tensor,          # [B, Q_total, D]
        text_embeds: torch.Tensor,           # [B, L, D]
        mol_token_flag: torch.Tensor,        # [B, L]  bool
        attention_mask: torch.Tensor,        # [B, L]  0/1, 左 padding
        labels: torch.Tensor,                # [B, L]  int  (可含 -100)
        max_pos: int,                        # llm.config.max_position_embeddings
        inner_cluster: torch.Tensor,         # Not used in current simplified version
        inner_cluster_batch: torch.Tensor,   # Not used in current simplified version
        num_global_tokens: int,              # number of global tokens per molecule
        local_q_only: bool = False,
        entropy_gids: torch.Tensor = None,   # For multi-molecule support
    ):
        """
        Multi-molecule version of inject_queries.
        
        Logic:
        1. Detect continuous <mol> sequences (e.g., <mol> <mol> <mol> for 3 molecules)
        2. Each molecule group gets the SAME global queries (first num_global_tokens)
           - Within each molecule, <mol> tokens cycle through global queries by position
           - This allows all molecules to share the same global representation
        3. Insert remaining local queries after the last <mol> token
        
        Example with 3 molecules and 8 global tokens:
            Molecule 1: <mol>[0] <mol>[1] ... <mol>[7]
            Molecule 2: <mol>[0] <mol>[1] ... <mol>[7]  (same pattern)
            Molecule 3: <mol>[0] <mol>[1] ... <mol>[7]  (same pattern)
        
        Args:
            inner_cluster: Molecule tracking info (used during encoding, not injection)
            inner_cluster_batch: Batch tracking info (used during encoding, not injection)
            num_global_tokens: number of global query tokens (e.g., 8 or 32)
        """
        ignore_index = -100
        B, _, D = query_output.shape
        query_output = query_output.to(text_embeds.dtype)

        embeds_list, mask_list, label_list, new_lengths = [], [], [], []

        for i in range(B):
            flag_i = mol_token_flag[i].nonzero(as_tuple=False).squeeze()  # <mol> positions
            q_i = query_output[i]  # [Q_total, D]
            n_true = flag_i.numel()  # number of <mol> tokens
            n_q = q_i.size(0)  # total number of queries
            pad_left = (attention_mask[i] == 0).sum().item()
            
            # Detect continuous <mol> sequences for molecule grouping
            mol_positions = flag_i if flag_i.dim() > 0 else flag_i.unsqueeze(0)
            if mol_positions.numel() == 0:
                # No <mol> tokens - shouldn't happen but handle gracefully
                embeds_list.append(text_embeds[i])
                mask_list.append(attention_mask[i])
                if labels is not None:
                    label_list.append(labels[i])
                else:
                    label_list.append(None)
                new_lengths.append(text_embeds[i].size(0))
                continue
            
            # Group consecutive <mol> tokens into molecule sequences
            mol_groups = []  # List of lists of positions for each molecule
            current_group = [mol_positions[0].item()]
            
            for j in range(1, mol_positions.numel()):
                pos = mol_positions[j].item()
                prev_pos = mol_positions[j-1].item()
                # Consecutive or nearly consecutive (allowing 0-2 tokens in between)
                if pos - prev_pos <= 1:
                    current_group.append(pos)
                else:
                    mol_groups.append(current_group)
                    current_group = [pos]
            mol_groups.append(current_group)
            
            num_molecules = len(mol_groups)
            
            # --- 1. Inject global queries to EVERY <mol> token (shared across all molecules) ---
            x = text_embeds[i].clone()
            if labels is not None:
                l = labels[i].clone()
            
            if not local_q_only:
                # Each <mol> token gets the SAME global queries (first num_global_tokens)
                global_queries = q_i[:num_global_tokens]  # [num_global_tokens, D]
                
                for mol_idx, mol_group in enumerate(mol_groups):
                    # Inject the same global queries to each <mol> in this molecule
                    for idx_in_group, mol_pos in enumerate(mol_group):
                        # Each <mol> position gets ONE global query token
                        # Cycle through global queries if more <mol> tokens than global queries
                        query_idx = idx_in_group % num_global_tokens
                        x[mol_pos] = global_queries[query_idx]
                        if labels is not None:
                            l[mol_pos] = ignore_index
            
            # Mark all <mol> positions as ignore in labels
            if labels is not None:
                l[mol_positions] = ignore_index
            
            # --- 2. Insert remaining local queries ---
            # For multi-molecule, insert remaining queries after the LAST <mol> token
            # (Cannot reliably separate local queries by molecule since encoder aggregates them)
            local_q = q_i[num_global_tokens:]
            if local_q.numel() > 0:
                insert_pos = mol_positions[-1].item() + 1
                x = torch.cat([x[:insert_pos], local_q, x[insert_pos:]], dim=0)
                if labels is not None:
                    local_lbl = torch.full((local_q.size(0),), ignore_index, dtype=l.dtype, device=l.device)
                    l = torch.cat([l[:insert_pos], local_lbl, l[insert_pos:]], dim=0)
            
            # --- 3. Generate attention mask ---
            cur_len = x.size(0)
            ones_len = cur_len - pad_left
            cur_mask = torch.cat([
                torch.zeros(pad_left, dtype=torch.long, device=x.device),
                torch.ones(ones_len, dtype=torch.long, device=x.device)
            ], dim=0)
            
            embeds_list.append(x)
            mask_list.append(cur_mask)
            if labels is not None:
                label_list.append(l)
            else:
                label_list.append(None)
            new_lengths.append(cur_len)

        # --- 4. Pad/truncate to max length ---
        max_len = min(max(new_lengths), max_pos)
        padded_embeds, padded_mask, padded_labels = [], [], []

        for emb, m, l in zip(embeds_list, mask_list, label_list):
            emb = emb[:max_len]
            m = m[:max_len]
            if labels is not None:
                l = l[:max_len]

            if emb.size(0) < max_len:
                pad_len = max_len - emb.size(0)
                emb_pad = torch.zeros(pad_len, D, dtype=emb.dtype, device=emb.device)
                m_pad = torch.zeros(pad_len, dtype=m.dtype, device=m.device)
                if labels is not None:
                    l_pad = torch.full((pad_len,), ignore_index, dtype=l.dtype, device=l.device)

                emb = torch.cat([emb, emb_pad], dim=0)
                m = torch.cat([m, m_pad], dim=0)
                if labels is not None:
                    l = torch.cat([l, l_pad], dim=0)

            padded_embeds.append(emb)
            padded_mask.append(m)
            if labels is not None:
                padded_labels.append(l)
            else:
                padded_labels.append(None)

        text_embeds = torch.stack(padded_embeds, dim=0)
        attention_mask = torch.stack(padded_mask, dim=0)
        if labels is not None:
            labels = torch.stack(padded_labels, dim=0)

        return text_embeds, attention_mask, labels, max_len

    def inject_queries(
        self,
        query_output: torch.Tensor,          # [B, Q_total, D]
        text_embeds: torch.Tensor,           # [B, L, D]
        mol_token_flag: torch.Tensor,        # [B, L]  bool
        attention_mask: torch.Tensor,        # [B, L]  0/1, left padding
        labels: torch.Tensor,                # [B, L]  int  (may contain -100)
        max_pos: int,                        # llm.config.max_position_embeddings
        local_q_only: bool = False,          # whether to use only local Q tokens
        inner_cluster: torch.Tensor = None,  # For multi-molecule support
        inner_cluster_batch: torch.Tensor = None,  # For multi-molecule support
        num_global_tokens: int = None,       # For multi-molecule support
        entropy_gids: torch.Tensor = None,   # For multi-molecule support
    ):
        # Detect multi-molecule mode and dispatch to appropriate function
        # if inner_cluster is not None and inner_cluster_batch is not None and num_global_tokens is not None:
        #     return self.inject_queries_multi_molecule(
        #         query_output=query_output,
        #         text_embeds=text_embeds,
        #         mol_token_flag=mol_token_flag,
        #         attention_mask=attention_mask,
        #         labels=labels,
        #         max_pos=max_pos,
        #         inner_cluster=inner_cluster,
        #         inner_cluster_batch=inner_cluster_batch,
        #         num_global_tokens=num_global_tokens,
        #         local_q_only=local_q_only,
        #         entropy_gids=entropy_gids,
        #     )
        
        # Original single-molecule logic
        ignore_index = -100
        B, _, D = query_output.shape
        query_output = query_output.to(text_embeds.dtype)

        embeds_list, mask_list, label_list, new_lengths = [], [], [], []

        for i in range(B):
            flag_i = mol_token_flag[i].nonzero(as_tuple=False).squeeze()  # True 位置
            q_i    = query_output[i]                                     # [Q_total, D]
            n_true = flag_i.numel()                                      # number of global Q tokens
            n_q    = q_i.size(0)                                         # total Q tokens
            pad_left = (attention_mask[i] == 0).sum().item()             # left padding count

            assert n_q >= n_true, f"Sample {i}: Q count {n_q} < True count {n_true}"

            # --- 1. Write global Q tokens (replace <mol> positions) ---
            x = text_embeds[i]
            if labels is not None:
                l = labels[i]
                if not local_q_only:
                    x[flag_i] = q_i[:n_true]
                l[flag_i] = ignore_index
            else:
                if not local_q_only:
                    x[flag_i] = q_i[:n_true]

            # --- 2. Insert local Q tokens (after the last <mol> position) ---
            local_q = q_i[n_true:]                                       # may be empty
            if local_q.numel():
                insert_pos = flag_i[-1].item() + 1
                x = torch.cat([x[:insert_pos], local_q, x[insert_pos:]], dim=0)
                if labels is not None:
                    local_lbl = torch.full((local_q.size(0),), ignore_index, dtype=l.dtype, device=l.device)
                    l = torch.cat([l[:insert_pos], local_lbl, l[insert_pos:]], dim=0)
            # --- 3. Generate corresponding attention mask ---
            cur_len = x.size(0)
            ones_len = cur_len - pad_left
            cur_mask = torch.cat([
                torch.zeros(pad_left, dtype=torch.long, device=x.device),
                torch.ones(ones_len, dtype=torch.long, device=x.device)
            ], dim=0)

            embeds_list.append(x)
            mask_list.append(cur_mask)
            if labels is not None:
                label_list.append(l)
            else:
                label_list.append(None)
            new_lengths.append(cur_len)

        # --- 4. Pad / truncate to batch max length and max_pos ---
        max_len = min(max(new_lengths), max_pos)
        padded_embeds, padded_mask, padded_labels = [], [], []

        for emb, m, l in zip(embeds_list, mask_list, label_list):
            emb = emb[:max_len]
            m   = m[:max_len]
            if labels is not None:
                l   = l[:max_len]

            if emb.size(0) < max_len:        # Right-pad
                pad_len = max_len - emb.size(0)
                emb_pad = torch.zeros(pad_len, D, dtype=emb.dtype, device=emb.device)
                m_pad   = torch.zeros(pad_len,     dtype=m.dtype,   device=m.device)
                if labels is not None:
                    l_pad = torch.full((pad_len,), ignore_index, dtype=l.dtype, device=l.device)

                emb = torch.cat([emb, emb_pad], dim=0)
                m   = torch.cat([m,   m_pad],   dim=0)
                if labels is not None:
                    l   = torch.cat([l,   l_pad],   dim=0)

            padded_embeds.append(emb)
            padded_mask.append(m)
            if labels is not None:
                padded_labels.append(l)
            else:
                padded_labels.append(None)


        text_embeds = torch.stack(padded_embeds, dim=0)
        attention_mask = torch.stack(padded_mask, dim=0)
        if labels is not None:
            labels = torch.stack(padded_labels, dim=0)

        return text_embeds, attention_mask, labels, max_len



    def forward(self, graph_batch, text_batch):
        # Support different modes:
        # 1. LLM-only mode: no encoder at all (set via llm_only=True during init)
        # 2. Text-only mode: encoder exists but no graph data provided
        # 3. Molecule-text mode: graph data provided (single or multi-molecule)
        
        # Check if we're in LLM-only or text-only mode
        is_text_only = (self.llm_only or 
                        'unimol' not in graph_batch or 
                        graph_batch is None or 
                        (isinstance(graph_batch, dict) and len(graph_batch) == 0))
        
        if is_text_only:
            # Text-only mode: no molecular encoder
            inputs_embeds = self.llm.get_input_embeddings()(text_batch.input_ids)
            attention_mask = text_batch.attention_mask
            labels = text_batch.labels if hasattr(text_batch, 'labels') else None
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=labels,
                use_cache=False,
            )
            return outputs
        
        # Process graph batch (single or multiple molecules)
        # brics_gids and entropy_gids are now in graph_batch
        # The encoder will extract them automatically
        _, _, query_output = self.encoder(graph_batch)
        query_output = self.llm_proj(query_output.last_hidden_state) #[batch_size,num_query_token,dim]

        inputs_embeds = self.llm.get_input_embeddings()(text_batch.input_ids) # [batch_size, max_len, dim]

        if hasattr(text_batch, 'labels'):
            labels = text_batch.labels
        else:
            labels = None

        # Extract inner_cluster information for multi-molecule support
        # inner_cluster is assigned during batching, before encoders (encoder-agnostic)
        inner_cluster = graph_batch.get('inner_cluster', None)
        inner_cluster_batch = graph_batch.get('inner_cluster_batch', None)
        entropy_gids = graph_batch.get('entropy_gids', None)
        num_global_tokens = self.num_query_tokens if inner_cluster is not None else None
        

        inputs_embeds, attention_mask, labels, _ = self.inject_queries(
            query_output=query_output,
            text_embeds=inputs_embeds,
            mol_token_flag=text_batch.mol_token_flag,
            max_pos=self.llm.config.max_position_embeddings,
            labels=labels,
            attention_mask=text_batch.attention_mask,
            local_q_only=self.local_q_only,
            inner_cluster=inner_cluster,
            inner_cluster_batch=inner_cluster_batch,
            num_global_tokens=num_global_tokens,
            entropy_gids=entropy_gids,
        )

        # Align dtypes (e.g. Half vs BFloat16) to avoid runtime errors when using quantized models
        # inputs_embeds[text_batch.mol_token_flag] = query_output.flatten(0, 1) # [batch_size, max_len, dim]
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
            use_cache=False,
        )
        
        return outputs

    @torch.no_grad()
    def generate(
        self,
        graph_batch,
        text_batch,
        do_sample=False,
        num_beams=1,
        max_length=None,
        min_length=1,
        max_new_tokens=512,
        min_new_tokens=None,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_return_sequences=1,
        top_p=None,
        temperature=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        # Support different modes (same logic as forward())
        # 1. LLM-only mode: no encoder at all (set via llm_only=True during init)
        # 2. Text-only mode: encoder exists but no graph data provided
        # 3. Molecule-text mode: graph data provided
        
        is_text_only = (self.llm_only or 
                        'unimol' not in graph_batch or 
                        graph_batch is None or 
                        (isinstance(graph_batch, dict) and len(graph_batch) == 0))
        
        if is_text_only:
            inputs_embeds = self.llm.get_input_embeddings()(text_batch.input_ids)
            attention_mask = text_batch.attention_mask
            
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=do_sample,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
            )
            return outputs
        
        # 1. Graph -> Query
        # brics_gids and entropy_gids are already in graph_batch, no need to pass separately
        _, _, query_output = self.encoder(graph_batch)
        query_output = self.llm_proj(query_output.last_hidden_state)  # [B,Q,D]

        # 2. Text embeddings
        inputs_embeds = self.llm.get_input_embeddings()(text_batch.input_ids)

        # 3. Inject queries into text embeddings
        if hasattr(text_batch, 'labels'):
            labels = text_batch.labels
        else:
            labels = None

        # Extract inner_cluster information for multi-molecule support
        # inner_cluster is assigned during batching, before encoders (encoder-agnostic)
        inner_cluster = graph_batch.get('inner_cluster', None)
        inner_cluster_batch = graph_batch.get('inner_cluster_batch', None)
        num_global_tokens = self.num_query_tokens if inner_cluster is not None else None

        inputs_embeds, attention_mask, _, _ = self.inject_queries(
            query_output=query_output,
            text_embeds=inputs_embeds,
            mol_token_flag=text_batch.mol_token_flag,
            attention_mask=text_batch.attention_mask,
            labels=labels,
            max_pos=self.llm.config.max_position_embeddings,
            local_q_only=self.local_q_only,
            inner_cluster=inner_cluster,
            inner_cluster_batch=inner_cluster_batch,
            num_global_tokens=num_global_tokens,
        )

        # 4. Generate
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
        )
        return outputs

    @torch.no_grad()
    def generate_with_smiles(
        self,
        smiles_list,
        text_batch,
        do_sample=False,
        num_beams=1,
        max_length=None,
        min_length=1,
        max_new_tokens=1024,
        min_new_tokens=None,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_return_sequences=1,
        top_p=None,
        temperature=None,
        pad_token_id=None,
        eos_token_id=None,
        brics_gids=None,
        entropy_gids=None,
    ):
        # This method requires encoder for molecule processing
        if self.llm_only:
            raise RuntimeError(
                "generate_with_smiles() is not available in LLM-only mode. "
                "Use generate() with text_batch for text-only generation."
            )
        
        graph_batch = get_mol_graphs(smiles_list, self.encoder.unimol_dictionary, self.device)
        outputs = self.generate(
            graph_batch=graph_batch,
            text_batch=text_batch,
            do_sample=do_sample,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            brics_gids=brics_gids,
            entropy_gids=entropy_gids,
        )
        return outputs

    def _load_checkpoint_before_peft(self, ckpt_path):
        """
        Internal method to load checkpoint before PEFT model creation.
        Only loads encoder and llm_proj weights, skipping LLM weights.
        
        Args:
            ckpt_path: Path to checkpoint file (.ckpt, .pt, .pth for PyTorch or .safetensors for HuggingFace)
        """
        # Skip checkpoint loading in LLM-only mode (no encoder to load)
        if self.llm_only:
            logger.info(f"⚠️  Skipping checkpoint loading in LLM-only mode (no encoder)")
            return
        
        logger.info(f"Loading encoder and projector from checkpoint: {ckpt_path}")
        
        path = Path(ckpt_path)
        
        # Detect file type and load accordingly
        if path.suffix == '.safetensors':
            logger.info("Detected safetensors format")
            state_dict_raw = load_safetensors(ckpt_path)
        else:
            logger.info("Detected PyTorch checkpoint format")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state_dict_raw = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Determine prefix from keys
        first_key = list(state_dict_raw.keys())[0]
        if 'mol_llama.' in first_key:
            prefix_len = 10
            prefix = "mol_llama."
        elif 'model.' in first_key:
            prefix_len = 6
            prefix = "model."
        else:
            prefix_len = 0
            prefix = ""
        
        # Extract only encoder and llm_proj weights (skip LLM)
        state_dict = {}
        for k, v in state_dict_raw.items():
            if k.startswith(prefix):
                k_stripped = k[prefix_len:]
                state_dict[k_stripped] = v
        
        logger.info(f"Found {len(state_dict)} encoder/projector parameters to load")

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        # Filter expected missing keys
        expected_missing = []
        for k in missing_keys:
            if 'position_ids' in k or k.startswith("encoder.graph_encoder.") or k.startswith("encoder.static_q_mask"):
                expected_missing.append(k)
            else:
                assert False, f"Unexpected missing key: {k}"
        if len(unexpected_keys) > 0:
            logger.warning(f"Unexpected missing keys: {unexpected_keys}")
            assert False, f"Unexpected missing keys: {unexpected_keys}"
        logger.info(f"✅ Successfully loaded encoder and projector weights (LLM will be initialized separately)")

    def load_from_ckpt(self, ckpt_path, lora_init=False):
        """
        Load checkpoint from either PyTorch checkpoint or HuggingFace safetensors.
        Supports both single files and directories with multiple safetensors shards.
        
        Args:
            ckpt_path: Path to checkpoint file (.ckpt, .pt, .pth for PyTorch or .safetensors for HuggingFace)
                      or directory containing model.safetensors.index.json and shard files
        """
        logger.info(f"Loading from checkpoint: {ckpt_path}")
        
        path = Path(ckpt_path)

        if lora_init:
            # Check if PEFT is already applied to avoid double wrapping
            if hasattr(self.llm, 'peft_config'):
                logger.warning("⚠️  PEFT is already applied to the model. Skipping lora_init to avoid double PEFT wrapping.")
            else:
                peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                            inference_mode=False,
                                            r=self.config.llm_config.lora_config.r,
                                            lora_alpha=self.config.llm_config.lora_config.lora_alpha,
                                            lora_dropout=self.config.llm_config.lora_config.lora_dropout,
                                            target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
                self.llm = get_peft_model(self.llm, peft_config)
        
        # Check if input is a directory with multiple safetensors shards
        if path.is_dir():
            # Look for safetensors index file
            index_files = glob.glob(str(path / "*.safetensors.index.json"))
            if index_files:
                # Load from index file
                index_path = sorted(index_files)[0]
                logger.info(f"Detected safetensors directory with index file: {index_path}")
                
                # Parse index file to get shard files
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                weight_map = index_data.get('weight_map', {})
                shard_files = sorted(set(weight_map.values()))
                
                # Load all shards and merge
                state_dict_raw = {}
                for shard_file in shard_files:
                    shard_path = path / shard_file
                    if not shard_path.exists():
                        raise FileNotFoundError(f"Shard file not found: {shard_path}")
                    logger.info(f"Loading shard: {shard_file}")
                    shard_dict = load_safetensors(str(shard_path))
                    state_dict_raw.update(shard_dict)
                logger.info(f"Loaded {len(shard_files)} shard files, total {len(state_dict_raw)} parameters")
            else:
                # No index file, try to find safetensors files directly
                safetensors_files = glob.glob(str(path / "*.safetensors"))
                if safetensors_files:
                    logger.info(f"Detected safetensors directory without index, found {len(safetensors_files)} files")
                    state_dict_raw = {}
                    for shard_file in sorted(safetensors_files):
                        logger.info(f"Loading shard: {Path(shard_file).name}")
                        shard_dict = load_safetensors(shard_file)
                        state_dict_raw.update(shard_dict)
                    logger.info(f"Loaded {len(safetensors_files)} shard files, total {len(state_dict_raw)} parameters")
                else:
                    raise ValueError(f"No safetensors files found in directory: {ckpt_path}")
        elif path.suffix == '.safetensors':
            # Load single HuggingFace safetensor file
            logger.info("Detected safetensors format")
            state_dict_raw = load_safetensors(ckpt_path)
        else:
            # Load PyTorch checkpoint format
            logger.info("Detected PyTorch checkpoint format")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            # Some checkpoints save state_dict directly, others wrap it
            state_dict_raw = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Determine prefix from keys
        first_key = list(state_dict_raw.keys())[0]
        if 'mol_llama.' in first_key:
            prefix_len = 10
            prefix = "mol_llama."
        elif 'model.' in first_key:
            prefix_len = 6
            prefix = "model."
        else:
            prefix_len = 0
            prefix = ""
        
        # Extract relevant state dict with prefix removal
        state_dict = {k[prefix_len:]:v for k,v in state_dict_raw.items() if k.startswith(prefix)}
        
        # Filter out encoder keys in LLM-only mode
        if self.llm_only:
            logger.info(f"LLM-only mode: Filtering out encoder weights from checkpoint")
            state_dict = {k:v for k,v in state_dict.items() 
                         if not k.startswith("encoder.") and not k.startswith("llm_proj.")}
            logger.info(f"Found {len(state_dict)} LLM parameters to load (encoder weights skipped)")
        else:
            logger.info(f"Found {len(state_dict)} parameters to load")

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        assert len(unexpected_keys) == 0, f"unexpected keys: {unexpected_keys}"

        for k in missing_keys:
            if 'position_ids' in k: continue
            # In LLM-only mode, encoder keys are expected to be missing
            if self.llm_only and (k.startswith("encoder.") or k.startswith("llm_proj.")):
                continue
            if not (k.startswith("encoder.graph_encoder.") or k.startswith("llm.")) or k.startswith("encoder.static_q_mask"):
                logger.warning(f"❌ Unexpected missing key: {k}")
            else:
                logger.warning(f"Key: {k}, make sure this key is loaded before.")
            assert k.startswith("encoder.graph_encoder.") or \
                k.startswith("llm.") or k.startswith("encoder.static_q_mask") or \
                (self.llm_only and (k.startswith("encoder.") or k.startswith("llm_proj.")))
        
        logger.info(f"✅ Successfully loaded weights from {ckpt_path}")
        
    
    def load_from_stage1_ckpt(self, ckpt_path):
        """
        Load stage1 checkpoint from either PyTorch Lightning checkpoint or HuggingFace safetensors.
        Only loads encoder weights (not applicable for LLM-only mode).
        
        Args:
            ckpt_path: Path to checkpoint file (.ckpt, .pt, .pth for PyTorch or .safetensors for HuggingFace)
        """
        # Stage1 checkpoints only contain encoder weights - skip in LLM-only mode
        if self.llm_only:
            logger.warning(f"⚠️  Cannot load Stage1 checkpoint in LLM-only mode (Stage1 only contains encoder weights)")
            return
        
        logger.info(f"Loading from stage1 checkpoint: {ckpt_path}")
        
        path = Path(ckpt_path)
        
        # Detect file type and load accordingly
        if path.suffix == '.safetensors':
            # Load HuggingFace safetensor format
            logger.info("Detected safetensors format")
            state_dict_raw = load_safetensors(ckpt_path)
        else:
            # Load PyTorch Lightning checkpoint format
            logger.info("Detected PyTorch checkpoint format")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            # Some checkpoints save state_dict directly, others wrap it
            state_dict_raw = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Extract encoder parameters
        # Handle different possible prefixes: "encoder.", "model.encoder.", etc.
        state_dict = {}
        for k, v in state_dict_raw.items():
            if k.startswith("encoder."):
                # Remove "encoder." prefix (8 chars)
                state_dict[k[8:]] = v
            elif k.startswith("model.encoder."):
                # Remove "model.encoder." prefix (14 chars)
                state_dict[k[14:]] = v
        
        if not state_dict:
            logger.warning(f"No encoder weights found. Available keys: {list(state_dict_raw.keys())[:5]}...")
        
        logger.info(f"Found {len(state_dict)} encoder parameters to load")
        
        
        # Load state dict into encoder
        missing_keys, unexpected_keys = self.encoder.load_state_dict(state_dict, strict=False, assign=True)
        
        assert len(unexpected_keys) == 0, f"Unexpected keys found: {unexpected_keys}"
        
        # Validate missing keys - only graph_encoder keys are allowed to be missing
        for k in missing_keys:
            assert k.startswith("graph_encoder."), f"Missing unexpected key: {k}"
        
        logger.info(f"✅ Successfully loaded encoder weights from {ckpt_path}")

 
def gen_3d_conformation_from_rdkit(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        num_atoms = mol.GetNumAtoms()

        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=1, numThreads=8, pruneRmsThresh=1, maxAttempts=10000, useRandomCoords=False)
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=8)
        except:
            pass
        mol = Chem.RemoveHs(mol)
    except:
        return None, None
    if mol.GetNumConformers() == 0:
        return None, None

    if num_atoms != mol.GetNumAtoms():
        return None, None

    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coordinates = np.array(mol.GetConformer().GetPositions())
    return atoms, coordinates


def gen_3d_conformation_from_openbabel(smiles):
    mol = pybel.readstring('smi', smiles)
    mol.make3D(forcefield='mmff94', steps=10000)
    mol.OBMol.DeleteHydrogens()

    atomic_nums = [atom.atomicnum for atom in mol.atoms]
    pt = Chem.GetPeriodicTable()
    atoms = [pt.GetElementSymbol(atomic_num) for atomic_num in atomic_nums]
    coordinates = np.array([atom.coords for atom in mol.atoms])
    return atoms, coordinates


def gen_3d_conformation_from_libraries(smiles):
    atoms, coordinates = gen_3d_conformation_from_rdkit(smiles)
    if atoms is None or coordinates is None:
        atoms, coordinates = gen_3d_conformation_from_openbabel(smiles)

    return atoms, coordinates


def get_mol_graphs(smiles_list, dictionary, device):
    data_graphs = defaultdict(list)
    for idx, smiles in enumerate(tqdm(smiles_list, desc='Processing Molecules...')):
        atoms, coordinates = gen_3d_conformation_from_libraries(smiles)

        if atoms is None or coordinates is None:
            logger.warning(f"Invalid SMILES for {idx}-th SMILES: {smiles}")
            continue

        data_graphs['unimol'].append(
            get_unimol_data(atoms, coordinates, dictionary, remove_Hs=True))

        graph = smiles2graph(smiles)
        data_graphs['moleculestm'].append(Data(x=graph['node_feat'], 
                                        edge_index=graph['edge_index'], 
                                        edge_attr=graph['edge_feat']))

    d3_collater = Mol3DCollater(dictionary.pad())
    graph_batch = {}
    graph_batch['unimol'] = d3_collater(data_graphs['unimol'])
    graph_batch['moleculestm'] = Batch.from_data_list(data_graphs['moleculestm'])

    for key in graph_batch.keys():
        if key == 'unimol':
            for key_ in graph_batch[key].keys():
                graph_batch[key][key_] = graph_batch[key][key_].to(device)
        elif key == 'moleculestm':
            graph_batch[key] = graph_batch[key].to(device)
        
    return graph_batch


def build_local_q_assignment(entropy_gids, inner_cluster, inner_cluster_batch):
    """
    Compute local_q start offsets and cluster assignments based on entropy_gids.

    Args:
        entropy_gids: list of list, len = B
            entropy_gids[b]: atom-level patch IDs of length N_b (0..P_b-1)
        inner_cluster: LongTensor [N_total_atoms]
            inner_cluster ID for each atom (0-indexed within each sample)
        inner_cluster_batch: LongTensor [N_total_atoms]
            batch index for each atom (0..B-1)

    Returns:
        local_q_starts: list[int], len = B
            Starting offset of each sample's local_q in the global local_q tensor.
            Sample b's local_q indices span [local_q_starts[b], local_q_starts[b] + num_patches_b).

        local_q_cluster: list[LongTensor], len = B
            local_q_cluster[b]: shape [num_patches_b]
            The inner_cluster ID for each local_q (patch) in sample b.
    """
    import torch

    B = len(entropy_gids)
    device = inner_cluster.device
    dtype = inner_cluster.dtype

    local_q_starts = []
    local_q_cluster = []

    offset = 0  # global local_q offset

    inner_cluster = inner_cluster.to(device)
    inner_cluster_batch = inner_cluster_batch.to(device)

    for b in range(B):
        gids_b = entropy_gids[b]  # list[int], len N_b (number of atoms)
        if len(gids_b) == 0:
            # No atoms / no local patches for this sample
            local_q_starts.append(offset)
            local_q_cluster.append(torch.empty(0, dtype=dtype, device=device))
            continue

        gids_b_tensor = torch.tensor(gids_b, device=device, dtype=torch.long)

        # Slice out atoms for this sample from inner_cluster
        atom_mask_b = (inner_cluster_batch == b)
        clusters_b = inner_cluster[atom_mask_b]  # [N_b]
        assert clusters_b.numel() == gids_b_tensor.numel(), \
            f"Batch {b}: entropy_gids length {gids_b_tensor.numel()} != atom count in inner_cluster_batch {clusters_b.numel()}"

        # Number of patches for this sample (use unique for safety)
        patch_ids = torch.unique(gids_b_tensor).tolist()
        patch_ids = sorted(patch_ids)
        num_patches_b = len(patch_ids)

        local_q_starts.append(offset)

        # Map each patch -> one inner_cluster (one segment / molecule)
        cluster_per_patch = []

        for p in patch_ids:
            mask_p = (gids_b_tensor == p)       # atoms belonging to this patch
            patch_clusters = clusters_b[mask_p] # their inner_cluster IDs

            # Each patch should belong to exactly one inner_cluster
            uniq = torch.unique(patch_clusters)
            if uniq.numel() != 1:
                raise ValueError(
                    f"Batch {b}, patch {p} spans multiple inner_clusters: {uniq.tolist()}"
                )

            cluster_per_patch.append(uniq.item())

        local_q_cluster.append(
            torch.tensor(cluster_per_patch, dtype=dtype, device=device)
        )

        # Advance the global offset by this sample's patch count
        offset += num_patches_b

    return local_q_starts, local_q_cluster

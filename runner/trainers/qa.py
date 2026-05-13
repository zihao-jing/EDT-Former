import os
from typing import Any, Dict, Optional, Union, Tuple, List
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch import optim, nn
from transformers import (
    BertTokenizerFast, 
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    Trainer,
    get_cosine_schedule_with_warmup,
)
from transformers.utils import logging
from peft import LoraConfig, get_peft_model, TaskType
from torch_geometric.data import Batch
from safetensors.torch import load_file as load_safetensors

from models.mol_llama import EDTFormer

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np
from Levenshtein import distance as levenshtein_distance
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import DataStructs
from bert_score import score as bert_score

logger = logging.get_logger(__name__)


class MoleculeQATrainer(Trainer):
    """Trainer for molecule QA tasks (multiple choice questions)."""
    def __init__(self, vocab_size, model_config, train_config, tokenizer, use_dq_encoder=False, torch_dtype=None, **kwargs):
        self.train_config = train_config
        
        if torch_dtype is None:
            if train_config.precision == 'bf16-mixed':
                torch_dtype = "bfloat16"
            elif train_config.precision == '16':
                torch_dtype = "float16"
            elif train_config.precision == '32':
                torch_dtype = "float32"
        
        self.use_dq_encoder = use_dq_encoder
        logger.info(f"use_dq_encoder: {use_dq_encoder}")

        if train_config.get('llm_only', False) and train_config.get('llm_backbone', None) is not None:
            # LLM baseline - only use language model without molecular encoders
            logger.info("Using LLM baseline: ", train_config.llm_model_path)
            
            # Check if using T5 model (encoder-decoder architecture) by inspecting model config
            use_t5 = False
            try:
                from transformers import AutoConfig
                model_cfg = AutoConfig.from_pretrained(train_config.llm_model_path)
                if hasattr(model_cfg, 'architectures') and model_cfg.architectures:
                    use_t5 = any('T5ForConditionalGeneration' in arch for arch in model_cfg.architectures)
                    logger.info(f"Model architectures: {model_cfg.architectures}")
            except Exception as e:
                logger.warning(f"Could not load model config, falling back to path-based detection: {e}")
                use_t5 = getattr(train_config, 'use_t5', False) or 't5' in str(train_config.llm_model_path).lower()
            
            if use_t5:
                # T5 is a seq2seq model
                logger.info("Detected T5 model - using T5ForConditionalGeneration")
                model = T5ForConditionalGeneration.from_pretrained(
                    train_config.llm_model_path,
                    torch_dtype=torch_dtype,
                )
                model.resize_token_embeddings(vocab_size)
                
                # Apply LoRA if not freezing LLM
                if not getattr(train_config, 'freeze_llm', False):
                    peft_config = LoraConfig(
                        task_type=TaskType.SEQ_2_SEQ_LM,
                        inference_mode=False,
                        r=model_config.llm_config.lora_config.r,
                        lora_alpha=model_config.llm_config.lora_config.lora_alpha,
                        lora_dropout=model_config.llm_config.lora_config.lora_dropout,
                        target_modules=["q", "v"]  # T5 uses different attention module names
                    )
                    model = get_peft_model(model, peft_config)
                    logger.info("Applied LoRA to T5 baseline")
                
                self.is_t5_baseline = True
            else:
                # Use AutoModelForCausalLM to support Gemma, Qwen, Mistral, LLaMA, etc.
                if train_config.enable_flash:
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            train_config.llm_model_path,
                            torch_dtype=torch_dtype,
                            attn_implementation="flash_attention_2",
                        )
                        logger.info("Using flash attention for LLM baseline")
                    except TypeError:
                        # Some architectures may not accept attn_implementation
                        model = AutoModelForCausalLM.from_pretrained(
                            train_config.llm_model_path,
                            torch_dtype=torch_dtype,
                        )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        train_config.llm_model_path,
                        torch_dtype=torch_dtype,
                    )
                
                model.resize_token_embeddings(vocab_size)
                
                # Apply LoRA if not freezing LLM
                if not getattr(train_config, 'freeze_llm', False):
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=model_config.llm_config.lora_config.r,
                        lora_alpha=model_config.llm_config.lora_config.lora_alpha,
                        lora_dropout=model_config.llm_config.lora_config.lora_dropout,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    )
                    model = get_peft_model(model, peft_config)
                    logger.info("Applied LoRA to LLM baseline")
                
                self.is_t5_baseline = False
            
            self.is_llm_baseline = True
        else:
            # Use DQ encoder (default molecular encoder)
            self.is_llm_baseline = False
            # Override LLM backbone if explicitly specified (e.g., via llm_backbone parameter)
            # For edt_former, this is typically None, using the default LLM from config
            if hasattr(train_config, 'llm_model_path') and train_config.llm_model_path is not None:
                model_config.llm_config.llm_model = train_config.llm_model_path
            model = EDTFormer(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = getattr(train_config, 'freeze_llm', False),
                brics_gids_enable = train_config.brics_gids_enable,
                entropy_gids_enable = train_config.entropy_gids_enable,
                enable_blending = getattr(train_config, 'enable_blending', False),
                load_ckpt_before_peft = getattr(train_config, 'load_ckpt_before_peft', False),
                ckpt_path = getattr(train_config, 'ckpt_path', None),
                llm_only = getattr(train_config, 'llm_only', False),  # Skip encoder for text-only tasks
            )

        self.test_step_outputs = []
        
        # Initialize parent Trainer
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
    
    @property
    def tokenizer(self):
        """Access tokenizer via processing_class to avoid deprecation warning."""
        return self.processing_class

    def _get_eos_token_ids(self):
        ids = []
        try:
            if getattr(self.tokenizer, 'eos_token_id', None) is not None:
                ids.append(self.tokenizer.eos_token_id)
        except Exception:
            pass
        # Try a few common end-of-turn markers across model families
        for tok in ["<|eot_id|>", "<eos_token>", "<end_of_turn>", "<|endoftext|>", "<eos>", "</s>"]:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    ids.append(tid)
            except Exception:
                continue
        # De-duplicate while preserving order
        ids = list(dict.fromkeys(ids))
        return ids if len(ids) > 0 else None

    def load_from_ckpt(self, ckpt_path, lora_init=False):
        """
        Load checkpoint from either PyTorch checkpoint or HuggingFace safetensors.
        
        Args:
            ckpt_path: Path to checkpoint file (.ckpt, .pt, .pth for PyTorch or .safetensors for HuggingFace)
            lora_init: Whether to initialize LoRA when loading checkpoint (used for mol-llama baseline)
        """
        if hasattr(self.model, 'load_from_ckpt'):
            self.model.load_from_ckpt(ckpt_path, lora_init=lora_init)
        else:
            # Load checkpoint manually
            path = Path(ckpt_path)
            if path.suffix == '.safetensors':
                logger.info("Detected safetensors format")
                state_dict = load_safetensors(ckpt_path)
            else:
                logger.info("Detected PyTorch checkpoint format")
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"✅ Successfully loaded weights from {ckpt_path}")

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        Override to use custom optimizer/scheduler from train_config.
        """
        if self.optimizer is None:
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.train_config.init_lr, 
                weight_decay=self.train_config.weight_decay
            )
            self.optimizer = optimizer
        
        if self.lr_scheduler is None:
            if self.train_config.scheduler == 'linear_warmup_cosine_lr':
                warmup_steps = min(num_training_steps, self.train_config.warmup_steps)
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps
                )
            elif self.train_config.scheduler == 'None':
                self.lr_scheduler = None
        
        return self.optimizer, self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for training.
        
        Args:
            model: The model to compute loss for
            inputs: Dict with 'graph_batch', 'text_batch', 'brics_gids', 'entropy_gids', 'other_infos'
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items in batch (for newer transformers versions)
        """
        graph_batch = inputs.get('graph_batch', {})
        text_batch = inputs['text_batch']
        # brics_gids and entropy_gids are now in graph_batch and will be extracted by the model
        
        if self.is_llm_baseline:
            # For LLM baseline, only use text_batch for forward pass
            if getattr(self, 'is_t5_baseline', False):
                # T5 is encoder-decoder, needs labels to be prepared
                output = model(
                    input_ids=text_batch.input_ids,
                    attention_mask=text_batch.attention_mask,
                    labels=text_batch.input_ids
                )
            else:
                # Causal LM
                output = model(
                    input_ids=text_batch.input_ids,
                    attention_mask=text_batch.attention_mask,
                    labels=text_batch.input_ids
                )
            loss = output.loss
        else:
            # Standard molecular + text training
            output = model(graph_batch, text_batch)
            loss = output['loss'] if isinstance(output, dict) else output.loss
        
        return (loss, output) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation/prediction step.
        """
        inputs = self._prepare_inputs(inputs)
        
        graph_batch = inputs.get('graph_batch', {})
        text_batch = inputs['text_batch']
        brics_gids = inputs.get('brics_gids', None)
        entropy_gids = inputs.get('entropy_gids', None)
        other_infos = inputs.get('other_infos', {})
        
        # Check if text_batch has labels attribute
        has_labels = hasattr(text_batch, 'labels') or ('labels' in text_batch if isinstance(text_batch, dict) else False)
        
        with torch.no_grad():
            if self.is_llm_baseline:
                # For LLM baseline, only use text for generation
                if getattr(self, 'is_t5_baseline', False):
                    # T5 generation
                    responses = model.generate(
                        input_ids=text_batch.input_ids,
                        attention_mask=text_batch.attention_mask,
                        max_length=512,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    # Causal LM generation
                    eos_ids = self._get_eos_token_ids()
                    gen_kwargs = {
                        'input_ids': text_batch.input_ids,
                        'attention_mask': text_batch.attention_mask,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'max_new_tokens': 512,
                        'do_sample': True,
                        'temperature': 0.7,
                    }
                    if eos_ids is not None:
                        gen_kwargs['eos_token_id'] = eos_ids
                    responses = model.generate(**gen_kwargs)
            else:
                # Standard molecular + text generation
                responses = model.generate(
                    graph_batch, 
                    text_batch,
                    pad_token_id = self.tokenizer.pad_token_id,
                    eos_token_id = [self.tokenizer.eos_token_id],
                )
            
            generated_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
            original_texts = self.tokenizer.batch_decode(text_batch['input_ids'], skip_special_tokens=False)
            pattern = r"[Aa]nswer:"

            # Generate further if the output does not contain "Answer:"
            no_format_indices = []
            new_texts = []
            for idx, (original_text, generated_text) in enumerate(zip(original_texts, generated_texts)):
                if not re.search(pattern, generated_text):
                    no_format_indices.append(idx)
                    new_texts.append(original_text + generated_text + "\n\nAnswer: ")
            
            if len(no_format_indices) > 0 and not self.is_llm_baseline:
                new_graph_batch = {"unimol": {}, "moleculestm": {}}
                for k, v in graph_batch['unimol'].items():
                    new_graph_batch['unimol'][k] = v[no_format_indices]
                new_graph_batch['moleculestm'] = Batch.from_data_list(graph_batch['moleculestm'].index_select(no_format_indices))
                
                # Copy brics_gids and entropy_gids if they exist
                if 'brics_gids' in graph_batch and graph_batch['brics_gids'] is not None:
                    new_graph_batch['brics_gids'] = [graph_batch['brics_gids'][i] for i in no_format_indices]
                else:
                    new_graph_batch['brics_gids'] = None
                    
                if 'entropy_gids' in graph_batch and graph_batch['entropy_gids'] is not None:
                    new_graph_batch['entropy_gids'] = [graph_batch['entropy_gids'][i] for i in no_format_indices]
                else:
                    new_graph_batch['entropy_gids'] = None

                new_text_batch = self.tokenizer(
                    new_texts,
                    truncation=False,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    add_special_tokens=False,
                ).to(self.args.device)
                new_text_batch.mol_token_flag = (new_text_batch.input_ids == self.tokenizer.mol_token_id).to(self.args.device)

                new_responses = model.generate(
                    new_graph_batch, 
                    new_text_batch,
                    pad_token_id = self.tokenizer.pad_token_id,
                    eos_token_id = [self.tokenizer.eos_token_id],
                )
                new_generated_texts = self.tokenizer.batch_decode(new_responses, skip_special_tokens=True)

                for _, i in enumerate(no_format_indices):
                    generated_texts[i] += "\n\nAnswer: " + new_generated_texts[_]

            # Store outputs for metrics computation
            for response, answer, task in zip(generated_texts, other_infos['answer'], other_infos['task']):
                self.test_step_outputs.append({
                    'response': response,
                    'answer': answer,
                    'task': task
                })
        
        # Compute loss only if we have valid labels (not all -100)
        # Note: In inference mode (do_infer=True), labels are all -100, so loss will be NaN
        loss = None
        if has_labels:
            try:
                with torch.no_grad():
                    # Check if labels contain any valid (non -100) values
                    if hasattr(text_batch, 'labels'):
                        valid_labels = (text_batch.labels != -100).any()
                        if valid_labels:
                            loss = self.compute_loss(model, inputs, return_outputs=False)
                    else:
                        loss = self.compute_loss(model, inputs, return_outputs=False)
            except Exception as e:
                logger.warning(f"Could not compute loss during evaluation: {e}")
                loss = None
        
        return (loss, None, None)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to compute custom metrics after prediction loop.
        Note: eval_loss will be None/NaN for inference-mode evaluation (do_infer=True)
        because evaluation datasets don't include labels. Use accuracy instead.
        """
        self.test_step_outputs = []  # Reset outputs
        
        # Run standard evaluation
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Log a helpful message if loss is NaN
        if self.args.local_rank in [-1, 0] and (f"{metric_key_prefix}_loss" not in output or output.get(f"{metric_key_prefix}_loss") is None or (isinstance(output.get(f"{metric_key_prefix}_loss"), float) and output[f"{metric_key_prefix}_loss"] != output[f"{metric_key_prefix}_loss"])):  # NaN check
            logger.info(f"ℹ️  {metric_key_prefix}_loss is None/NaN (expected for inference-mode evaluation). Using accuracy metric instead.")
        
        # Compute metrics on all ranks from their local outputs
        corrects, results = self.compute_metrics_qa(self.test_step_outputs)
        
        # Add accuracy to output metrics (all ranks)
        if 'overall' in corrects:
            output[f"{metric_key_prefix}_accuracy"] = corrects['overall']['accuracy']
        else:
            # If no outputs collected (can happen on some ranks), set to 0
            output[f"{metric_key_prefix}_accuracy"] = 0.0
        
        # Save results only on main process
        if self.args.local_rank in [-1, 0]:
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Include global step in filename to avoid overwriting
            step_suffix = f"_step{self.state.global_step}" if self.state.global_step > 0 else ""
            
            with open(os.path.join(output_dir, f"{metric_key_prefix}_results{step_suffix}.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open(os.path.join(output_dir, f"{metric_key_prefix}_metrics{step_suffix}.json"), "w") as f:
                json.dump(corrects, f, indent=4)
        
        return output
    
    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        """
        Override predict to compute custom metrics after prediction loop.
        """
        self.test_step_outputs = []  # Reset outputs
        
        # Run standard prediction
        output = super().predict(test_dataset, ignore_keys, metric_key_prefix)
        
        # Gather outputs from all processes
        if self.args.local_rank in [-1, 0]:
            corrects, results = self.compute_metrics_qa(self.test_step_outputs)
            
            # Save results
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, f"{metric_key_prefix}_results.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open(os.path.join(output_dir, f"{metric_key_prefix}_metrics.json"), "w") as f:
                json.dump(corrects, f, indent=4)
        
        return output
                

    def compute_metrics_qa(self, outputs):
        """Compute metrics for QA task (multiple choice)."""
        results = defaultdict(list)
        corrects = {}

        # Handle empty outputs (can happen on some ranks in distributed training)
        if not outputs:
            return corrects, results

        for output in outputs:
            task = output['task']
            response = output['response']
            answer = output['answer'].replace("Answer: ", "")
            prediction = response.split("Answer: ")[-1].strip()
            if 'A' in prediction:
                prediction = 'A'
            elif 'B' in prediction:
                prediction = 'B'
            elif 'C' in prediction:
                prediction = 'C'
            elif 'D' in prediction:
                prediction = 'D'
            else:
                prediction = 'None'

            correct = 1 if prediction == answer else 0
            results[task].append({
                'response': response,
                'answer': answer,
                'prediction': prediction,
                'correct': correct,
            })

            if task not in corrects:
                corrects[task] = {'correct': 0, 'total': 0}
            corrects[task]['correct'] += correct
            corrects[task]['total'] += 1

        tasks = list(corrects.keys())
        for task in tasks:
            correct = corrects[task]['correct']
            total = corrects[task]['total']
            accuracy = correct / total * 100
            corrects[task]['accuracy'] = accuracy

        # Calculate overall accuracy
        if tasks:
            overall_correct = sum([corrects[task]['correct'] for task in tasks])
            overall_total = sum([corrects[task]['total'] for task in tasks])
            overall_accuracy = overall_correct / overall_total * 100
            corrects['overall'] = {'correct': overall_correct, 'total': overall_total, 'accuracy': overall_accuracy}

        return corrects, results


class MoleculeGENQATrainer(Trainer):
    """Trainer for molecule generation/captioning tasks."""
    def __init__(self, vocab_size, model_config, train_config, tokenizer, use_dq_encoder=False, torch_dtype=None, **kwargs):
        self.train_config = train_config
        
        if torch_dtype is None:
            if train_config.precision == 'bf16-mixed':
                torch_dtype = "bfloat16"
            elif train_config.precision == '16':
                torch_dtype = "float16"
            elif train_config.precision == '32':
                torch_dtype = "float32"
        
        self.use_dq_encoder = use_dq_encoder
        logger.info(f"use_dq_encoder: {use_dq_encoder}")

        if train_config.get('llm_only', False) and train_config.get('llm_backbone', None) is not None:
            logger.info("Using LLM baseline: ", train_config.llm_model_path)
            
            # Check if using T5 model (encoder-decoder architecture) by inspecting model config
            use_t5 = False
            try:
                from transformers import AutoConfig
                model_cfg = AutoConfig.from_pretrained(train_config.llm_model_path)
                if hasattr(model_cfg, 'architectures') and model_cfg.architectures:
                    use_t5 = any('T5ForConditionalGeneration' in arch for arch in model_cfg.architectures)
                    logger.info(f"Model architectures: {model_cfg.architectures}")
            except Exception as e:
                logger.warning(f"Could not load model config, falling back to path-based detection: {e}")
                use_t5 = getattr(train_config, 'use_t5', False) or 't5' in str(train_config.llm_model_path).lower()
            
            if use_t5:
                # T5 is a seq2seq model
                logger.info("Detected T5 model - using T5ForConditionalGeneration")
                model = T5ForConditionalGeneration.from_pretrained(
                    train_config.llm_model_path,
                    torch_dtype=torch_dtype,
                )
                model.resize_token_embeddings(vocab_size)
                
                # Apply LoRA if not freezing LLM
                if not getattr(train_config, 'freeze_llm', False):
                    peft_config = LoraConfig(
                        task_type=TaskType.SEQ_2_SEQ_LM,
                        inference_mode=False,
                        r=model_config.llm_config.lora_config.r,
                        lora_alpha=model_config.llm_config.lora_config.lora_alpha,
                        lora_dropout=model_config.llm_config.lora_config.lora_dropout,
                        target_modules=["q", "v"]  # T5 uses different attention module names
                    )
                    model = get_peft_model(model, peft_config)
                    logger.info("Applied LoRA to T5 baseline")
                
                self.is_t5_baseline = True
            else:
                if train_config.enable_flash:
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            train_config.llm_model_path,
                            torch_dtype=torch_dtype,
                            attn_implementation="flash_attention_2",
                        )
                        logger.info("Using flash attention for LLM baseline")
                    except TypeError:
                        model = AutoModelForCausalLM.from_pretrained(
                            train_config.llm_model_path,
                            torch_dtype=torch_dtype,
                        )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        train_config.llm_model_path,
                        torch_dtype=torch_dtype,
                    )
                model.resize_token_embeddings(vocab_size)
                if not getattr(train_config, 'freeze_llm', False):
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=model_config.llm_config.lora_config.r,
                        lora_alpha=model_config.llm_config.lora_config.lora_alpha,
                        lora_dropout=model_config.llm_config.lora_config.lora_dropout,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    )
                    model = get_peft_model(model, peft_config)
                    logger.info("Applied LoRA to LLM baseline")
                
                self.is_t5_baseline = False
            
            self.is_llm_baseline = True
        else:
            # Use DQ encoder (default molecular encoder)
            self.is_llm_baseline = False
            # Override LLM backbone if explicitly specified (e.g., via llm_backbone parameter)
            # For edt_former, this is typically None, using the default LLM from config
            if hasattr(train_config, 'llm_model_path') and train_config.llm_model_path is not None:
                model_config.llm_config.llm_model = train_config.llm_model_path
            model = EDTFormer(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = getattr(train_config, 'freeze_llm', False),
                brics_gids_enable = train_config.brics_gids_enable,
                entropy_gids_enable = train_config.entropy_gids_enable,
                enable_blending = getattr(train_config, 'enable_blending', False),
                load_ckpt_before_peft = getattr(train_config, 'load_ckpt_before_peft', False),
                ckpt_path = getattr(train_config, 'ckpt_path', None),
                llm_only = getattr(train_config, 'llm_only', False),  # Skip encoder for text-only tasks
            )

        self.test_step_outputs = []
        
        # Initialize parent Trainer
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
    
    @property
    def tokenizer(self):
        """Access tokenizer via processing_class to avoid deprecation warning."""
        return self.processing_class

    def load_from_ckpt(self, ckpt_path, lora_init=False):
        """
        Load checkpoint from either PyTorch checkpoint or HuggingFace safetensors.
        
        Args:
            ckpt_path: Path to checkpoint file (.ckpt, .pt, .pth for PyTorch or .safetensors for HuggingFace)
            lora_init: Whether to initialize LoRA when loading checkpoint (used for mol-llama baseline)
        """
        if hasattr(self.model, 'load_from_ckpt'):
            self.model.load_from_ckpt(ckpt_path, lora_init=lora_init)
        else:
            path = Path(ckpt_path)
            if path.suffix == '.safetensors':
                logger.info("Detected safetensors format")
                state_dict = load_safetensors(ckpt_path)
            else:
                logger.info("Detected PyTorch checkpoint format")
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"✅ Successfully loaded weights from {ckpt_path}")

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and scheduler for HF Trainer."""
        if self.optimizer is None:
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.train_config.init_lr, 
                weight_decay=self.train_config.weight_decay
            )
            self.optimizer = optimizer
        
        if self.lr_scheduler is None:
            if self.train_config.scheduler == 'linear_warmup_cosine_lr':
                warmup_steps = min(num_training_steps, self.train_config.warmup_steps)
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps
                )
            elif self.train_config.scheduler == 'None':
                self.lr_scheduler = None
        
        return self.optimizer, self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss for training."""
        graph_batch = inputs.get('graph_batch', {})
        text_batch = inputs['text_batch']
        # brics_gids and entropy_gids are now in graph_batch and will be extracted by the model
        
        if getattr(self, 'is_llm_baseline', False):
            output = model(
                input_ids=text_batch.input_ids,
                attention_mask=text_batch.attention_mask,
                labels=text_batch.input_ids
            )
            loss = output.loss
        else:
            output = model(graph_batch, text_batch)
            loss = output['loss'] if isinstance(output, dict) else output.loss

        return (loss, output) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Perform an evaluation/prediction step for generation."""
        inputs = self._prepare_inputs(inputs)
        
        graph_batch = inputs.get('graph_batch', {})
        text_batch = inputs['text_batch']
        brics_gids = inputs.get('brics_gids', None)
        entropy_gids = inputs.get('entropy_gids', None)
        other_infos = inputs.get('other_infos', {})
        
        # Check if text_batch has labels attribute
        has_labels = hasattr(text_batch, 'labels') or ('labels' in text_batch if isinstance(text_batch, dict) else False)
        
        with torch.no_grad():
            if getattr(self, 'is_llm_baseline', False):
                if getattr(self, 'is_t5_baseline', False):
                    # T5 generation
                    responses = model.generate(
                        input_ids=text_batch.input_ids,
                        attention_mask=text_batch.attention_mask,
                        max_length=512,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    # Causal LM generation
                    eos_ids = self._get_eos_token_ids()
                    gen_kwargs = {
                        'input_ids': text_batch.input_ids,
                        'attention_mask': text_batch.attention_mask,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'max_new_tokens': 512,
                        'do_sample': True,
                        'temperature': 0.7,
                    }
                    if eos_ids is not None:
                        gen_kwargs['eos_token_id'] = eos_ids
                    responses = model.generate(**gen_kwargs)
            else:
                responses = model.generate(
                    graph_batch,
                    text_batch,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=[self.tokenizer.eos_token_id],
                )
            generated_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

            for pred_text, gt_text in zip(generated_texts, other_infos['answer']):
                self.test_step_outputs.append({
                    'prediction': pred_text,
                    'ground_truth': gt_text,
                })
        
        # Compute loss only if we have valid labels (not all -100)
        # Note: In inference mode (do_infer=True), labels are all -100, so loss will be NaN
        loss = None
        if has_labels:
            try:
                with torch.no_grad():
                    # Check if labels contain any valid (non -100) values
                    if hasattr(text_batch, 'labels'):
                        valid_labels = (text_batch.labels != -100).any()
                        if valid_labels:
                            loss = self.compute_loss(model, inputs, return_outputs=False)
                    else:
                        loss = self.compute_loss(model, inputs, return_outputs=False)
            except Exception as e:
                logger.warning(f"Could not compute loss during evaluation: {e}")
                loss = None
        
        return (loss, None, None)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to compute custom metrics.
        Note: eval_loss is automatically computed by HuggingFace Trainer.
        """
        self.test_step_outputs = []
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Compute metrics on all ranks from their local outputs
        metrics, per_sample = self.compute_metrics_gen(self.test_step_outputs)
        
        # Add metrics to output (all ranks need these for best model selection)
        for k, v in metrics.items():
            output[f"{metric_key_prefix}_{k}"] = v
        
        # Save results only on main process
        if self.args.local_rank in [-1, 0]:
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Include global step in filename to avoid overwriting
            step_suffix = f"_step{self.state.global_step}" if self.state.global_step > 0 else ""
            
            with open(os.path.join(output_dir, f"{metric_key_prefix}_caption_results{step_suffix}.json"), "w") as f:
                json.dump(per_sample, f, indent=4)
            with open(os.path.join(output_dir, f"{metric_key_prefix}_caption_metrics{step_suffix}.json"), "w") as f:
                json.dump(metrics, f, indent=4)
        
        return output

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        """Override predict to compute custom metrics."""
        self.test_step_outputs = []
        output = super().predict(test_dataset, ignore_keys, metric_key_prefix)
        
        if self.args.local_rank in [-1, 0]:
            metrics, per_sample = self.compute_metrics_gen(self.test_step_outputs)
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, f"{metric_key_prefix}_caption_results.json"), "w") as f:
                json.dump(per_sample, f, indent=4)
            with open(os.path.join(output_dir, f"{metric_key_prefix}_caption_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
        
        return output
                

    def compute_metrics_gen(self, outputs):
        """Compute metrics for generation/captioning task."""
        # Handle empty outputs (can happen on some ranks in distributed training)
        if not outputs:
            return {
                'bleu2': 0.0,
                'bleu4': 0.0,
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'meteor': 0.0,
            }, []
        
        # Prepare pairs
        pairs = [(o['ground_truth'], o['prediction']) for o in outputs]
        per_sample = [{'ground_truth': gt, 'prediction': pred} for gt, pred in pairs]

        # Tokenizer as in reference implementation
        text_tokenizer = BertTokenizerFast.from_pretrained('allenai/scibert_scivocab_uncased')

        references = []
        hypotheses = []
        meteor_scores = []

        for gt, pred in pairs:
            gt_tokens = text_tokenizer.tokenize(gt, truncation=True, max_length=512, padding='max_length')
            gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
            gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
            gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

            pred_tokens = text_tokenizer.tokenize(pred, truncation=True, max_length=512, padding='max_length')
            pred_tokens = list(filter(('[PAD]').__ne__, pred_tokens))
            pred_tokens = list(filter(('[CLS]').__ne__, pred_tokens))
            pred_tokens = list(filter(('[SEP]').__ne__, pred_tokens))

            references.append([gt_tokens])
            hypotheses.append(pred_tokens)

            mscore = meteor_score([gt_tokens], pred_tokens)
            meteor_scores.append(mscore)

        bleu2 = corpus_bleu(references, hypotheses, weights=(.5, .5))
        bleu4 = corpus_bleu(references, hypotheses, weights=(.25, .25, .25, .25))

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        rouge_scores = []
        for gt, pred in pairs:
            rs = scorer.score(pred, gt)
            rouge_scores.append(rs)

        rouge_1 = float(np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]))
        rouge_2 = float(np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]))
        rouge_l = float(np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]))
        meteor_avg = float(np.mean(meteor_scores)) if len(meteor_scores) > 0 else 0.0

        metrics = {
            'BLEU-2': float(bleu2),
            'BLEU-4': float(bleu4),
            'ROUGE-1': rouge_1,
            'ROUGE-2': rouge_2,
            'ROUGE-L': rouge_l,
            'METEOR': meteor_avg,
        }

        return metrics, per_sample

    def _get_eos_token_ids(self):
        ids = []
        try:
            if getattr(self.tokenizer, 'eos_token_id', None) is not None:
                ids.append(self.tokenizer.eos_token_id)
        except Exception:
            pass
        for tok in ["<|eot_id|>", "<eos_token>", "<end_of_turn>", "<|endoftext|>", "<eos>", "</s>"]:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    ids.append(tid)
            except Exception:
                continue
        ids = list(dict.fromkeys(ids))
        return ids if len(ids) > 0 else None


class MoleculePropertyQATrainer(Trainer):
    """Trainer for molecule property prediction/regression tasks."""
    def __init__(self, vocab_size, model_config, train_config, tokenizer, use_dq_encoder=False, torch_dtype=None, **kwargs):
        self.train_config = train_config
        
        if torch_dtype is None:
            if train_config.precision == 'bf16-mixed':
                torch_dtype = "bfloat16"
            elif train_config.precision == '16':
                torch_dtype = "float16"
            elif train_config.precision == '32':
                torch_dtype = "float32"
        
        self.use_dq_encoder = use_dq_encoder
        logger.info(f"use_dq_encoder: {use_dq_encoder}")

        if train_config.get('llm_only', False) and train_config.get('llm_backbone', None) is not None:
            logger.info("Using LLM baseline: ", train_config.llm_model_path)
            
            # Check if using T5 model (encoder-decoder architecture) by inspecting model config
            use_t5 = False
            try:
                from transformers import AutoConfig
                model_cfg = AutoConfig.from_pretrained(train_config.llm_model_path)
                if hasattr(model_cfg, 'architectures') and model_cfg.architectures:
                    use_t5 = any('T5ForConditionalGeneration' in arch for arch in model_cfg.architectures)
                    logger.info(f"Model architectures: {model_cfg.architectures}")
            except Exception as e:
                logger.warning(f"Could not load model config, falling back to path-based detection: {e}")
                use_t5 = getattr(train_config, 'use_t5', False) or 't5' in str(train_config.llm_model_path).lower()
            
            if use_t5:
                # T5 is a seq2seq model
                logger.info("Detected T5 model - using T5ForConditionalGeneration")
                model = T5ForConditionalGeneration.from_pretrained(
                    train_config.llm_model_path,
                    torch_dtype=torch_dtype,
                )
                model.resize_token_embeddings(vocab_size)
                
                # Apply LoRA if not freezing LLM
                if not getattr(train_config, 'freeze_llm', False):
                    peft_config = LoraConfig(
                        task_type=TaskType.SEQ_2_SEQ_LM,
                        inference_mode=False,
                        r=model_config.llm_config.lora_config.r,
                        lora_alpha=model_config.llm_config.lora_config.lora_alpha,
                        lora_dropout=model_config.llm_config.lora_config.lora_dropout,
                        target_modules=["q", "v"]  # T5 uses different attention module names
                    )
                    model = get_peft_model(model, peft_config)
                    logger.info("Applied LoRA to T5 baseline")
                
                self.is_t5_baseline = True
            else:
                if train_config.enable_flash:
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            train_config.llm_model_path,
                            torch_dtype=torch_dtype,
                            attn_implementation="flash_attention_2",
                        )
                        logger.info("Using flash attention for LLM baseline")
                    except TypeError:
                        model = AutoModelForCausalLM.from_pretrained(
                            train_config.llm_model_path,
                            torch_dtype=torch_dtype,
                        )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        train_config.llm_model_path,
                        torch_dtype=torch_dtype,
                    )
                model.resize_token_embeddings(vocab_size)
                if not getattr(train_config, 'freeze_llm', False):
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=model_config.llm_config.lora_config.r,
                        lora_alpha=model_config.llm_config.lora_config.lora_alpha,
                        lora_dropout=model_config.llm_config.lora_config.lora_alpha,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    )
                    model = get_peft_model(model, peft_config)
                    logger.info("Applied LoRA to LLM baseline")
                
                self.is_t5_baseline = False
            
            self.is_llm_baseline = True
        else:
            # Use DQ encoder (default molecular encoder)
            self.is_llm_baseline = False
            # Override LLM backbone if explicitly specified (e.g., via llm_backbone parameter)
            # For edt_former, this is typically None, using the default LLM from config
            if hasattr(train_config, 'llm_model_path') and train_config.llm_model_path is not None:
                model_config.llm_config.llm_model = train_config.llm_model_path
            model = EDTFormer(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = getattr(train_config, 'freeze_llm', False),
                brics_gids_enable = train_config.brics_gids_enable,
                entropy_gids_enable = train_config.entropy_gids_enable,
                enable_blending = getattr(train_config, 'enable_blending', False),
                load_ckpt_before_peft = getattr(train_config, 'load_ckpt_before_peft', False),
                ckpt_path = getattr(train_config, 'ckpt_path', None),
                llm_only = getattr(train_config, 'llm_only', False),  # Skip encoder for text-only tasks
            )

        self.test_step_outputs = []
        
        # Initialize parent Trainer
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
    
    @property
    def tokenizer(self):
        """Access tokenizer via processing_class to avoid deprecation warning."""
        return self.processing_class

    def load_from_ckpt(self, ckpt_path, lora_init=False):
        """
        Load checkpoint from either PyTorch checkpoint or HuggingFace safetensors.
        
        Args:
            ckpt_path: Path to checkpoint file (.ckpt, .pt, .pth for PyTorch or .safetensors for HuggingFace)
            lora_init: Whether to initialize LoRA when loading checkpoint (used for mol-llama baseline)
        """
        if hasattr(self.model, 'load_from_ckpt'):
            self.model.load_from_ckpt(ckpt_path, lora_init=lora_init)
        else:
            path = Path(ckpt_path)
            if path.suffix == '.safetensors':
                logger.info("Detected safetensors format")
                state_dict = load_safetensors(ckpt_path)
            else:
                logger.info("Detected PyTorch checkpoint format")
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"✅ Successfully loaded weights from {ckpt_path}")

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and scheduler for HF Trainer."""
        if self.optimizer is None:
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.train_config.init_lr, 
                weight_decay=self.train_config.weight_decay
            )
            self.optimizer = optimizer
        
        if self.lr_scheduler is None:
            if self.train_config.scheduler == 'linear_warmup_cosine_lr':
                warmup_steps = min(num_training_steps, self.train_config.warmup_steps)
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps
                )
            elif self.train_config.scheduler == 'None':
                self.lr_scheduler = None
        
        return self.optimizer, self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss for training."""
        graph_batch = inputs.get('graph_batch', {})
        text_batch = inputs['text_batch']
        # brics_gids and entropy_gids are now in graph_batch and will be extracted by the model
        
        if getattr(self, 'is_llm_baseline', False):
            output = model(
                input_ids=text_batch.input_ids,
                attention_mask=text_batch.attention_mask,
                labels=text_batch.input_ids
            )
            loss = output.loss
        else:
            output = model(graph_batch, text_batch)
            loss = output['loss'] if isinstance(output, dict) else output.loss

        return (loss, output) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Perform an evaluation/prediction step for property prediction."""
        inputs = self._prepare_inputs(inputs)
        
        graph_batch = inputs.get('graph_batch', {})
        text_batch = inputs['text_batch']
        brics_gids = inputs.get('brics_gids', None)
        entropy_gids = inputs.get('entropy_gids', None)
        other_infos = inputs.get('other_infos', {})
        
        # Check if text_batch has labels attribute
        has_labels = hasattr(text_batch, 'labels') or ('labels' in text_batch if isinstance(text_batch, dict) else False)
        
        with torch.no_grad():
            if getattr(self, 'is_llm_baseline', False):
                if getattr(self, 'is_t5_baseline', False):
                    # T5 generation
                    responses = model.generate(
                        input_ids=text_batch.input_ids,
                        attention_mask=text_batch.attention_mask,
                        max_length=512,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    # Causal LM generation
                    eos_ids = self._get_eos_token_ids()
                    gen_kwargs = {
                        'input_ids': text_batch.input_ids,
                        'attention_mask': text_batch.attention_mask,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'max_new_tokens': 512,
                        'do_sample': True,
                        'temperature': 0.7,
                    }
                    if eos_ids is not None:
                        gen_kwargs['eos_token_id'] = eos_ids
                    responses = model.generate(**gen_kwargs)
            else:
                responses = model.generate(
                    graph_batch,
                    text_batch,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=[self.tokenizer.eos_token_id],
                )
            generated_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

            for pred_text, gt_text in zip(generated_texts, other_infos['answer']):
                self.test_step_outputs.append({
                    'prediction': pred_text,
                    'ground_truth': gt_text,
                })
        
        # Compute loss only if we have valid labels (not all -100)
        # Note: In inference mode (do_infer=True), labels are all -100, so loss will be NaN
        loss = None
        if has_labels:
            try:
                with torch.no_grad():
                    # Check if labels contain any valid (non -100) values
                    if hasattr(text_batch, 'labels'):
                        valid_labels = (text_batch.labels != -100).any()
                        if valid_labels:
                            loss = self.compute_loss(model, inputs, return_outputs=False)
                    else:
                        loss = self.compute_loss(model, inputs, return_outputs=False)
            except Exception as e:
                logger.warning(f"Could not compute loss during evaluation: {e}")
                loss = None
        
        return (loss, None, None)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to compute custom metrics.
        Note: eval_loss is automatically computed by HuggingFace Trainer.
        """
        self.test_step_outputs = []
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Compute metrics on all ranks from their local outputs
        metrics, per_sample = self.compute_metrics_regression(self.test_step_outputs)
        
        # Add MAE to output metrics (all ranks need this for best model selection)
        if 'MAE' in metrics:
            output[f"{metric_key_prefix}_mae"] = metrics['MAE']
        else:
            # If no outputs collected, set to a large value
            output[f"{metric_key_prefix}_mae"] = float('inf')
        
        # Save results only on main process
        if self.args.local_rank in [-1, 0]:
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Include global step in filename to avoid overwriting
            step_suffix = f"_step{self.state.global_step}" if self.state.global_step > 0 else ""
            
            with open(os.path.join(output_dir, f"{metric_key_prefix}_regression_results{step_suffix}.json"), "w") as f:
                json.dump(per_sample, f, indent=4)
            with open(os.path.join(output_dir, f"{metric_key_prefix}_regression_metrics{step_suffix}.json"), "w") as f:
                json.dump(metrics, f, indent=4)
        
        return output

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        """Override predict to compute custom metrics."""
        self.test_step_outputs = []
        output = super().predict(test_dataset, ignore_keys, metric_key_prefix)
        
        if self.args.local_rank in [-1, 0]:
            metrics, per_sample = self.compute_metrics_regression(self.test_step_outputs)
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, f"{metric_key_prefix}_regression_results.json"), "w") as f:
                json.dump(per_sample, f, indent=4)
            with open(os.path.join(output_dir, f"{metric_key_prefix}_regression_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
        
        return output
                

    def compute_metrics_regression(self, outputs):
        """Compute metrics for regression/property prediction task."""
        # Handle empty outputs (can happen on some ranks in distributed training)
        if not outputs:
            return {
                'MAE': float('inf'),
                'valid_count': 0,
                'total_count': 0,
            }, []
        
        # Extract last numeric value from prediction text and ground-truth text, compute MAE
        def _extract_last_number(text: Any):
            try:
                # If already numeric
                if isinstance(text, (int, float, np.number)):
                    return float(text)
                s = str(text)
                # Match floats/ints with optional sign and scientific notation
                matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
                if not matches:
                    return None
                return float(matches[-1])
            except Exception:
                return None

        per_sample = []
        abs_errors = []
        total = 0
        valid = 0

        for o in outputs:
            gt_text = o['ground_truth']
            pred_text = o['prediction']
            gt_val = _extract_last_number(gt_text)
            pred_val = _extract_last_number(pred_text)
            total += 1
            record = {
                'ground_truth_text': gt_text,
                'prediction_text': pred_text,
                'ground_truth': gt_val,
                'prediction': pred_val,
            }
            per_sample.append(record)
            if gt_val is not None and pred_val is not None:
                abs_errors.append(abs(pred_val - gt_val))
                valid += 1

        mae = float(np.mean(abs_errors)) if len(abs_errors) > 0 else float('nan')
        metrics = {
            'MAE': mae,
            'valid_count': int(valid),
            'total_count': int(total),
        }

        return metrics, per_sample

    def _get_eos_token_ids(self):
        ids = []
        try:
            if getattr(self.tokenizer, 'eos_token_id', None) is not None:
                ids.append(self.tokenizer.eos_token_id)
        except Exception:
            pass
        for tok in ["<|eot_id|>", "<eos_token>", "<end_of_turn>", "<|endoftext|>", "<eos>", "</s>"]:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    ids.append(tid)
            except Exception:
                continue
        ids = list(dict.fromkeys(ids))
        return ids if len(ids) > 0 else None


class MoleculeReactionTrainer(Trainer):
    """
    Trainer for molecule reaction/generation tasks (forward reaction, retrosynthesis, 
    reagent prediction, description-guided molecule design).
    
    Supports metrics: Exact match, BLEU, Levenshtein, RDK FTS, MACC FTS, Morgan FTS, Validity
    """
    def __init__(self, vocab_size, model_config, train_config, tokenizer, use_dq_encoder=False, torch_dtype=None, **kwargs):
        self.train_config = train_config
        
        if torch_dtype is None:
            if train_config.precision == 'bf16-mixed':
                torch_dtype = "bfloat16"
            elif train_config.precision == '16':
                torch_dtype = "float16"
            elif train_config.precision == '32':
                torch_dtype = "float32"
        
        self.use_dq_encoder = use_dq_encoder
        logger.info(f"use_dq_encoder: {use_dq_encoder}")

        if train_config.get('llm_only', False) and train_config.get('llm_backbone', None) is not None:
            logger.info("Using LLM baseline: ", train_config.llm_model_path)
            
            # Check if using T5 model (encoder-decoder architecture) by inspecting model config
            use_t5 = False
            try:
                from transformers import AutoConfig
                model_cfg = AutoConfig.from_pretrained(train_config.llm_model_path)
                if hasattr(model_cfg, 'architectures') and model_cfg.architectures:
                    use_t5 = any('T5ForConditionalGeneration' in arch for arch in model_cfg.architectures)
                    logger.info(f"Model architectures: {model_cfg.architectures}")
            except Exception as e:
                logger.warning(f"Could not load model config, falling back to path-based detection: {e}")
                use_t5 = getattr(train_config, 'use_t5', False) or 't5' in str(train_config.llm_model_path).lower()
            
            if use_t5:
                # T5 is a seq2seq model
                logger.info("Detected T5 model - using T5ForConditionalGeneration")
                model = T5ForConditionalGeneration.from_pretrained(
                    train_config.llm_model_path,
                    torch_dtype=torch_dtype,
                )
                model.resize_token_embeddings(vocab_size)
                
                # Apply LoRA if not freezing LLM
                if not getattr(train_config, 'freeze_llm', False):
                    peft_config = LoraConfig(
                        task_type=TaskType.SEQ_2_SEQ_LM,
                        inference_mode=False,
                        r=model_config.llm_config.lora_config.r,
                        lora_alpha=model_config.llm_config.lora_config.lora_alpha,
                        lora_dropout=model_config.llm_config.lora_config.lora_dropout,
                        target_modules=["q", "v"]  # T5 uses different attention module names
                    )
                    model = get_peft_model(model, peft_config)
                    logger.info("Applied LoRA to T5 baseline")
                
                self.is_t5_baseline = True
            elif train_config.enable_flash:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        train_config.llm_model_path,
                        torch_dtype=torch_dtype,
                        attn_implementation="flash_attention_2",
                    )
                    logger.info("Using flash attention for LLM baseline")
                except TypeError:
                    model = AutoModelForCausalLM.from_pretrained(
                        train_config.llm_model_path,
                        torch_dtype=torch_dtype,
                    )
                model.resize_token_embeddings(vocab_size)
                if not getattr(train_config, 'freeze_llm', False):
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=model_config.llm_config.lora_config.r,
                        lora_alpha=model_config.llm_config.lora_config.lora_alpha,
                        lora_dropout=model_config.llm_config.lora_config.lora_dropout,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    )
                    model = get_peft_model(model, peft_config)
                    logger.info("Applied LoRA to LLM baseline")
                self.is_t5_baseline = False
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    train_config.llm_model_path,
                    torch_dtype=torch_dtype,
                )
                model.resize_token_embeddings(vocab_size)
                if not getattr(train_config, 'freeze_llm', False):
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=model_config.llm_config.lora_config.r,
                        lora_alpha=model_config.llm_config.lora_config.lora_alpha,
                        lora_dropout=model_config.llm_config.lora_config.lora_dropout,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    )
                    model = get_peft_model(model, peft_config)
                    logger.info("Applied LoRA to LLM baseline")
                self.is_t5_baseline = False
            
            self.is_llm_baseline = True
        else:
            # Use DQ encoder (default molecular encoder)
            self.is_llm_baseline = False
            # Override LLM backbone if explicitly specified (e.g., via llm_backbone parameter)
            # For edt_former, this is typically None, using the default LLM from config
            if hasattr(train_config, 'llm_model_path') and train_config.llm_model_path is not None:
                model_config.llm_config.llm_model = train_config.llm_model_path
            model = EDTFormer(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = getattr(train_config, 'freeze_llm', False),
                brics_gids_enable = train_config.brics_gids_enable,
                entropy_gids_enable = train_config.entropy_gids_enable,
                enable_blending = getattr(train_config, 'enable_blending', False),
                load_ckpt_before_peft = getattr(train_config, 'load_ckpt_before_peft', False),
                ckpt_path = getattr(train_config, 'ckpt_path', None),
                llm_only = getattr(train_config, 'llm_only', False),  # Skip encoder for text-only tasks
            )

        self.test_step_outputs = []
        
        # Initialize parent Trainer
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
    
    @property
    def tokenizer(self):
        """Access tokenizer via processing_class to avoid deprecation warning."""
        return self.processing_class

    def load_from_ckpt(self, ckpt_path, lora_init=False):
        """Load checkpoint from either PyTorch checkpoint or HuggingFace safetensors."""
        if hasattr(self.model, 'load_from_ckpt'):
            self.model.load_from_ckpt(ckpt_path, lora_init=lora_init)
        else:
            path = Path(ckpt_path)
            if path.suffix == '.safetensors':
                logger.info("Detected safetensors format")
                state_dict = load_safetensors(ckpt_path)
            else:
                logger.info("Detected PyTorch checkpoint format")
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"✅ Successfully loaded weights from {ckpt_path}")

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and scheduler for HF Trainer."""
        if self.optimizer is None:
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.train_config.init_lr, 
                weight_decay=self.train_config.weight_decay
            )
            self.optimizer = optimizer
        
        if self.lr_scheduler is None:
            if self.train_config.scheduler == 'linear_warmup_cosine_lr':
                warmup_steps = min(num_training_steps, self.train_config.warmup_steps)
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps
                )
            elif self.train_config.scheduler == 'None':
                self.lr_scheduler = None
        
        return self.optimizer, self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss for training."""
        graph_batch = inputs.get('graph_batch', {})
        text_batch = inputs['text_batch']
        
        if getattr(self, 'is_llm_baseline', False):
            output = model(
                input_ids=text_batch.input_ids,
                attention_mask=text_batch.attention_mask,
                labels=text_batch.input_ids
            )
            loss = output.loss
        else:
            output = model(graph_batch, text_batch)
            loss = output['loss'] if isinstance(output, dict) else output.loss

        return (loss, output) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Perform an evaluation/prediction step for molecule generation."""
        inputs = self._prepare_inputs(inputs)
        
        graph_batch = inputs.get('graph_batch', {})
        text_batch = inputs['text_batch']
        other_infos = inputs.get('other_infos', {})
        
        # Check if text_batch has labels attribute
        has_labels = hasattr(text_batch, 'labels') or ('labels' in text_batch if isinstance(text_batch, dict) else False)
        
        with torch.no_grad():
            if getattr(self, 'is_llm_baseline', False):
                if getattr(self, 'is_t5_baseline', False):
                    # T5 generation
                    responses = model.generate(
                        input_ids=text_batch.input_ids,
                        attention_mask=text_batch.attention_mask,
                        max_length=512,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    # Causal LM generation
                    eos_ids = self._get_eos_token_ids()
                    gen_kwargs = {
                        'input_ids': text_batch.input_ids,
                        'attention_mask': text_batch.attention_mask,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'max_new_tokens': 512,
                        'do_sample': True,
                        'temperature': 0.7,
                    }
                    if eos_ids is not None:
                        gen_kwargs['eos_token_id'] = eos_ids
                    responses = model.generate(**gen_kwargs)
            else:
                responses = model.generate(
                    graph_batch,
                    text_batch,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=[self.tokenizer.eos_token_id],
                )
            generated_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

            for pred_text, gt_text in zip(generated_texts, other_infos['answer']):
                self.test_step_outputs.append({
                    'prediction': pred_text,
                    'ground_truth': gt_text,
                })
        
        # Compute loss only if we have valid labels
        loss = None
        if has_labels:
            try:
                with torch.no_grad():
                    if hasattr(text_batch, 'labels'):
                        valid_labels = (text_batch.labels != -100).any()
                        if valid_labels:
                            loss = self.compute_loss(model, inputs, return_outputs=False)
                    else:
                        loss = self.compute_loss(model, inputs, return_outputs=False)
            except Exception as e:
                logger.warning(f"Could not compute loss during evaluation: {e}")
                loss = None
        
        return (loss, None, None)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to compute custom metrics."""
        self.test_step_outputs = []
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Compute metrics on all ranks from their local outputs
        metrics, per_sample = self.compute_metrics_reaction(self.test_step_outputs)
        
        # Add metrics to output (all ranks need these for best model selection)
        for k, v in metrics.items():
            output[f"{metric_key_prefix}_{k}"] = v
        
        # Save results only on main process
        if self.args.local_rank in [-1, 0]:
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Include global step in filename to avoid overwriting
            step_suffix = f"_step{self.state.global_step}" if self.state.global_step > 0 else ""
            
            with open(os.path.join(output_dir, f"{metric_key_prefix}_reaction_results{step_suffix}.json"), "w") as f:
                json.dump(per_sample, f, indent=4)
            with open(os.path.join(output_dir, f"{metric_key_prefix}_reaction_metrics{step_suffix}.json"), "w") as f:
                json.dump(metrics, f, indent=4)
        
        return output

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        """Override predict to compute custom metrics."""
        self.test_step_outputs = []
        output = super().predict(test_dataset, ignore_keys, metric_key_prefix)
        
        if self.args.local_rank in [-1, 0]:
            metrics, per_sample = self.compute_metrics_reaction(self.test_step_outputs)
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, f"{metric_key_prefix}_reaction_results.json"), "w") as f:
                json.dump(per_sample, f, indent=4)
            with open(os.path.join(output_dir, f"{metric_key_prefix}_reaction_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
        
        return output

    def _selfies_to_smiles(self, selfies_str):
        """Convert SELFIES to SMILES. Returns None if invalid."""
        try:
            import selfies as sf
            smiles = sf.decoder(selfies_str)
            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol)  # Canonicalize
        except Exception:
            return None

    def _compute_fingerprint_similarity(self, smiles1, smiles2, fp_type='morgan'):
        """Compute Tanimoto similarity between two SMILES using specified fingerprint."""
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            
            if mol1 is None or mol2 is None:
                return 0.0
            
            if fp_type == 'morgan':
                fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
            elif fp_type == 'rdk':
                fp1 = Chem.RDKFingerprint(mol1)
                fp2 = Chem.RDKFingerprint(mol2)
            elif fp_type == 'maccs':
                fp1 = MACCSkeys.GenMACCSKeys(mol1)
                fp2 = MACCSkeys.GenMACCSKeys(mol2)
            else:
                return 0.0
            
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except Exception:
            return 0.0

    def compute_metrics_reaction(self, outputs):
        """
        Compute metrics for reaction/molecule generation tasks.
        
        Metrics:
        - Exact: Exact match accuracy
        - BLEU: BLEU score
        - Levenshtein: Levenshtein distance
        - RDK FTS: RDKit fingerprint Tanimoto similarity
        - MACC FTS: MACCS fingerprint Tanimoto similarity
        - Morgan FTS: Morgan fingerprint Tanimoto similarity
        - Validity: Valid SMILES percentage
        """
        # Handle empty outputs
        if not outputs:
            return {
                'exact': 0.0,
                'bleu': 0.0,
                'levenshtein': float('inf'),
                'rdk_fts': 0.0,
                'maccs_fts': 0.0,
                'morgan_fts': 0.0,
                'validity': 0.0,
            }, []
        
        per_sample = []
        exact_matches = []
        bleu_scores = []
        lev_distances = []
        rdk_similarities = []
        maccs_similarities = []
        morgan_similarities = []
        valid_count = 0
        total_count = len(outputs)
        
        for o in outputs:
            gt_selfies = o['ground_truth']
            pred_selfies = o['prediction']
            
            # Convert SELFIES to SMILES
            gt_smiles = self._selfies_to_smiles(gt_selfies)
            pred_smiles = self._selfies_to_smiles(pred_selfies)
            
            # Exact match (on SELFIES)
            exact_match = int(gt_selfies == pred_selfies)
            exact_matches.append(exact_match)
            
            # BLEU score (on SELFIES strings as tokens)
            try:
                ref_tokens = list(gt_selfies)
                pred_tokens = list(pred_selfies)
                bleu = corpus_bleu([[ref_tokens]], [pred_tokens], weights=(0.5, 0.5))
                bleu_scores.append(bleu)
            except Exception:
                bleu_scores.append(0.0)
            
            # Levenshtein distance (on SELFIES)
            try:
                lev_dist = levenshtein_distance(gt_selfies, pred_selfies)
                lev_distances.append(lev_dist)
            except Exception:
                lev_distances.append(len(gt_selfies))  # Worst case
            
            # Fingerprint similarities (only if both SMILES are valid)
            if gt_smiles and pred_smiles:
                valid_count += 1
                rdk_sim = self._compute_fingerprint_similarity(gt_smiles, pred_smiles, 'rdk')
                maccs_sim = self._compute_fingerprint_similarity(gt_smiles, pred_smiles, 'maccs')
                morgan_sim = self._compute_fingerprint_similarity(gt_smiles, pred_smiles, 'morgan')
                rdk_similarities.append(rdk_sim)
                maccs_similarities.append(maccs_sim)
                morgan_similarities.append(morgan_sim)
            else:
                # Invalid SMILES - assign 0 similarity
                rdk_similarities.append(0.0)
                maccs_similarities.append(0.0)
                morgan_similarities.append(0.0)
            
            per_sample.append({
                'ground_truth_selfies': gt_selfies,
                'prediction_selfies': pred_selfies,
                'ground_truth_smiles': gt_smiles,
                'prediction_smiles': pred_smiles,
                'exact_match': exact_match,
                'bleu': bleu_scores[-1],
                'levenshtein': lev_distances[-1],
                'rdk_fts': rdk_similarities[-1],
                'maccs_fts': maccs_similarities[-1],
                'morgan_fts': morgan_similarities[-1],
            })
        
        # Aggregate metrics
        metrics = {
            'exact': float(np.mean(exact_matches)),
            'bleu': float(np.mean(bleu_scores)),
            'levenshtein': float(np.mean(lev_distances)),
            'rdk_fts': float(np.mean(rdk_similarities)),
            'maccs_fts': float(np.mean(maccs_similarities)),
            'morgan_fts': float(np.mean(morgan_similarities)),
            'validity': float(valid_count / total_count) if total_count > 0 else 0.0,
            'valid_count': int(valid_count),
            'total_count': int(total_count),
        }
        
        return metrics, per_sample

    def _get_eos_token_ids(self):
        ids = []
        try:
            if getattr(self.tokenizer, 'eos_token_id', None) is not None:
                ids.append(self.tokenizer.eos_token_id)
        except Exception:
            pass
        for tok in ["<|eot_id|>", "<eos_token>", "<end_of_turn>", "<|endoftext|>", "<eos>", "</s>"]:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    ids.append(tid)
            except Exception:
                continue
        ids = list(dict.fromkeys(ids))
        return ids if len(ids) > 0 else None


class MoleculeOpenQuestionTrainer(Trainer):
    """
    Trainer for molecule open-question tasks.
    
    Supports metrics: BLEU, ROUGE-1, BertScore
    """
    def __init__(self, vocab_size, model_config, train_config, tokenizer, use_dq_encoder=False, torch_dtype=None, **kwargs):
        self.train_config = train_config
        
        if torch_dtype is None:
            if train_config.precision == 'bf16-mixed':
                torch_dtype = "bfloat16"
            elif train_config.precision == '16':
                torch_dtype = "float16"
            elif train_config.precision == '32':
                torch_dtype = "float32"
        
        self.use_dq_encoder = use_dq_encoder
        logger.info(f"use_dq_encoder: {use_dq_encoder}")

        if train_config.get('llm_only', False) and train_config.get('llm_backbone', None) is not None:
            logger.info("Using LLM baseline: ", train_config.llm_model_path)
            
            # Check if using T5 model (encoder-decoder architecture) by inspecting model config
            use_t5 = False
            try:
                from transformers import AutoConfig
                model_cfg = AutoConfig.from_pretrained(train_config.llm_model_path)
                if hasattr(model_cfg, 'architectures') and model_cfg.architectures:
                    use_t5 = any('T5ForConditionalGeneration' in arch for arch in model_cfg.architectures)
                    logger.info(f"Model architectures: {model_cfg.architectures}")
            except Exception as e:
                logger.warning(f"Could not load model config, falling back to path-based detection: {e}")
                use_t5 = getattr(train_config, 'use_t5', False) or 't5' in str(train_config.llm_model_path).lower()
            
            if use_t5:
                # T5 is a seq2seq model
                logger.info("Detected T5 model - using T5ForConditionalGeneration")
                model = T5ForConditionalGeneration.from_pretrained(
                    train_config.llm_model_path,
                    torch_dtype=torch_dtype,
                )
                model.resize_token_embeddings(vocab_size)
                
                # Apply LoRA if not freezing LLM
                if not getattr(train_config, 'freeze_llm', False):
                    peft_config = LoraConfig(
                        task_type=TaskType.SEQ_2_SEQ_LM,
                        inference_mode=False,
                        r=model_config.llm_config.lora_config.r,
                        lora_alpha=model_config.llm_config.lora_config.lora_alpha,
                        lora_dropout=model_config.llm_config.lora_config.lora_dropout,
                        target_modules=["q", "v"]  # T5 uses different attention module names
                    )
                    model = get_peft_model(model, peft_config)
                    logger.info("Applied LoRA to T5 baseline")
                
                self.is_t5_baseline = True
            elif train_config.enable_flash:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        train_config.llm_model_path,
                        torch_dtype=torch_dtype,
                        attn_implementation="flash_attention_2",
                    )
                    logger.info("Using flash attention for LLM baseline")
                except TypeError:
                    model = AutoModelForCausalLM.from_pretrained(
                        train_config.llm_model_path,
                        torch_dtype=torch_dtype,
                    )
                model.resize_token_embeddings(vocab_size)
                if not getattr(train_config, 'freeze_llm', False):
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=model_config.llm_config.lora_config.r,
                        lora_alpha=model_config.llm_config.lora_config.lora_alpha,
                        lora_dropout=model_config.llm_config.lora_config.lora_dropout,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    )
                    model = get_peft_model(model, peft_config)
                    logger.info("Applied LoRA to LLM baseline")
                self.is_t5_baseline = False
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    train_config.llm_model_path,
                    torch_dtype=torch_dtype,
                )
                model.resize_token_embeddings(vocab_size)
                if not getattr(train_config, 'freeze_llm', False):
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=model_config.llm_config.lora_config.r,
                        lora_alpha=model_config.llm_config.lora_config.lora_alpha,
                        lora_dropout=model_config.llm_config.lora_config.lora_dropout,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    )
                    model = get_peft_model(model, peft_config)
                    logger.info("Applied LoRA to LLM baseline")
                self.is_t5_baseline = False
            
            self.is_llm_baseline = True
        else:
            # Use DQ encoder (default molecular encoder)
            self.is_llm_baseline = False
            # Override LLM backbone if explicitly specified (e.g., via llm_backbone parameter)
            # For edt_former, this is typically None, using the default LLM from config
            if hasattr(train_config, 'llm_model_path') and train_config.llm_model_path is not None:
                model_config.llm_config.llm_model = train_config.llm_model_path
            model = EDTFormer(
                config=model_config,
                vocab_size=vocab_size,
                torch_dtype = torch_dtype,
                enable_flash = train_config.enable_flash,
                freeze_llm = getattr(train_config, 'freeze_llm', False),
                brics_gids_enable = train_config.brics_gids_enable,
                entropy_gids_enable = train_config.entropy_gids_enable,
                enable_blending = getattr(train_config, 'enable_blending', False),
                load_ckpt_before_peft = getattr(train_config, 'load_ckpt_before_peft', False),
                ckpt_path = getattr(train_config, 'ckpt_path', None),
                llm_only = getattr(train_config, 'llm_only', False),  # Skip encoder for text-only tasks
            )

        self.test_step_outputs = []
        
        # Initialize parent Trainer
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
    
    @property
    def tokenizer(self):
        """Access tokenizer via processing_class to avoid deprecation warning."""
        return self.processing_class

    def load_from_ckpt(self, ckpt_path, lora_init=False):
        """Load checkpoint from either PyTorch checkpoint or HuggingFace safetensors."""
        if hasattr(self.model, 'load_from_ckpt'):
            self.model.load_from_ckpt(ckpt_path, lora_init=lora_init)
        else:
            path = Path(ckpt_path)
            if path.suffix == '.safetensors':
                logger.info("Detected safetensors format")
                state_dict = load_safetensors(ckpt_path)
            else:
                logger.info("Detected PyTorch checkpoint format")
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"✅ Successfully loaded weights from {ckpt_path}")

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and scheduler for HF Trainer."""
        if self.optimizer is None:
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.train_config.init_lr, 
                weight_decay=self.train_config.weight_decay
            )
            self.optimizer = optimizer
        
        if self.lr_scheduler is None:
            if self.train_config.scheduler == 'linear_warmup_cosine_lr':
                warmup_steps = min(num_training_steps, self.train_config.warmup_steps)
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps
                )
            elif self.train_config.scheduler == 'None':
                self.lr_scheduler = None
        
        return self.optimizer, self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss for training."""
        graph_batch = inputs.get('graph_batch', {})
        text_batch = inputs['text_batch']
        
        if getattr(self, 'is_llm_baseline', False):
            output = model(
                input_ids=text_batch.input_ids,
                attention_mask=text_batch.attention_mask,
                labels=text_batch.input_ids
            )
            loss = output.loss
        else:
            output = model(graph_batch, text_batch)
            loss = output['loss'] if isinstance(output, dict) else output.loss

        return (loss, output) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Perform an evaluation/prediction step for open-question generation."""
        inputs = self._prepare_inputs(inputs)
        
        graph_batch = inputs.get('graph_batch', {})
        text_batch = inputs['text_batch']
        other_infos = inputs.get('other_infos', {})
        
        # Check if text_batch has labels attribute
        has_labels = hasattr(text_batch, 'labels') or ('labels' in text_batch if isinstance(text_batch, dict) else False)
        
        with torch.no_grad():
            if getattr(self, 'is_llm_baseline', False):
                if getattr(self, 'is_t5_baseline', False):
                    # T5 generation
                    responses = model.generate(
                        input_ids=text_batch.input_ids,
                        attention_mask=text_batch.attention_mask,
                        max_length=512,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    # Causal LM generation
                    eos_ids = self._get_eos_token_ids()
                    gen_kwargs = {
                        'input_ids': text_batch.input_ids,
                        'attention_mask': text_batch.attention_mask,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'max_new_tokens': 512,
                        'do_sample': True,
                        'temperature': 0.7,
                    }
                    if eos_ids is not None:
                        gen_kwargs['eos_token_id'] = eos_ids
                    responses = model.generate(**gen_kwargs)
            else:
                responses = model.generate(
                    graph_batch,
                    text_batch,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=[self.tokenizer.eos_token_id],
                )
            generated_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

            for pred_text, gt_text in zip(generated_texts, other_infos['answer']):
                self.test_step_outputs.append({
                    'prediction': pred_text,
                    'ground_truth': gt_text,
                })
        
        # Compute loss only if we have valid labels
        loss = None
        if has_labels:
            try:
                with torch.no_grad():
                    if hasattr(text_batch, 'labels'):
                        valid_labels = (text_batch.labels != -100).any()
                        if valid_labels:
                            loss = self.compute_loss(model, inputs, return_outputs=False)
                    else:
                        loss = self.compute_loss(model, inputs, return_outputs=False)
            except Exception as e:
                logger.warning(f"Could not compute loss during evaluation: {e}")
                loss = None
        
        return (loss, None, None)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to compute custom metrics."""
        self.test_step_outputs = []
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Compute metrics on all ranks from their local outputs
        metrics, per_sample = self.compute_metrics_open_question(self.test_step_outputs)
        
        # Add metrics to output (all ranks need these for best model selection)
        for k, v in metrics.items():
            output[f"{metric_key_prefix}_{k}"] = v
        
        # Save results only on main process
        if self.args.local_rank in [-1, 0]:
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Include global step in filename to avoid overwriting
            step_suffix = f"_step{self.state.global_step}" if self.state.global_step > 0 else ""
            
            with open(os.path.join(output_dir, f"{metric_key_prefix}_open_question_results{step_suffix}.json"), "w") as f:
                json.dump(per_sample, f, indent=4)
            with open(os.path.join(output_dir, f"{metric_key_prefix}_open_question_metrics{step_suffix}.json"), "w") as f:
                json.dump(metrics, f, indent=4)
        
        return output

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        """Override predict to compute custom metrics."""
        self.test_step_outputs = []
        output = super().predict(test_dataset, ignore_keys, metric_key_prefix)
        
        if self.args.local_rank in [-1, 0]:
            metrics, per_sample = self.compute_metrics_open_question(self.test_step_outputs)
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, f"{metric_key_prefix}_open_question_results.json"), "w") as f:
                json.dump(per_sample, f, indent=4)
            with open(os.path.join(output_dir, f"{metric_key_prefix}_open_question_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
        
        return output

    def compute_metrics_open_question(self, outputs):
        """
        Compute metrics for open-question tasks.
        
        Metrics:
        - BLEU: BLEU score
        - ROUGE-1: ROUGE-1 F-measure
        - BertScore: BertScore F1
        """
        # Handle empty outputs
        if not outputs:
            return {
                'bleu': 0.0,
                'rouge1': 0.0,
                'bertscore': 0.0,
            }, []
        
        per_sample = []
        
        # Collect predictions and ground truths
        predictions = [o['prediction'] for o in outputs]
        ground_truths = [o['ground_truth'] for o in outputs]
        
        # BLEU Score - compute on word tokens
        bleu_scores = []
        for pred, gt in zip(predictions, ground_truths):
            try:
                ref_tokens = gt.split()
                pred_tokens = pred.split()
                if len(pred_tokens) > 0:
                    bleu = corpus_bleu([[ref_tokens]], [pred_tokens], weights=(0.5, 0.5))
                else:
                    bleu = 0.0
                bleu_scores.append(bleu)
            except Exception:
                bleu_scores.append(0.0)
        
        # ROUGE Score
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        rouge_scores = []
        for pred, gt in zip(predictions, ground_truths):
            try:
                rs = scorer.score(gt, pred)
                rouge_scores.append(rs['rouge1'].fmeasure)
            except Exception:
                rouge_scores.append(0.0)
        
        # BertScore
        try:
            P, R, F1 = bert_score(predictions, ground_truths, lang='en', verbose=False)
            bert_scores = F1.cpu().numpy().tolist()
        except Exception as e:
            logger.warning(f"Could not compute BertScore: {e}")
            bert_scores = [0.0] * len(predictions)
        
        # Build per-sample results
        for pred, gt, bleu, rouge, bert in zip(predictions, ground_truths, bleu_scores, rouge_scores, bert_scores):
            per_sample.append({
                'prediction': pred,
                'ground_truth': gt,
                'bleu': float(bleu),
                'rouge1': float(rouge),
                'bertscore': float(bert),
            })
        
        # Aggregate metrics
        metrics = {
            'BLEU': float(np.mean(bleu_scores)),
            'ROUGE-1': float(np.mean(rouge_scores)),
            'BertScore': float(np.mean(bert_scores)),
        }
        
        return metrics, per_sample

    def _get_eos_token_ids(self):
        ids = []
        try:
            if getattr(self.tokenizer, 'eos_token_id', None) is not None:
                ids.append(self.tokenizer.eos_token_id)
        except Exception:
            pass
        for tok in ["<|eot_id|>", "<eos_token>", "<end_of_turn>", "<|endoftext|>", "<eos>", "</s>"]:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    ids.append(tid)
            except Exception:
                continue
        ids = list(dict.fromkeys(ids))
        return ids if len(ids) > 0 else None


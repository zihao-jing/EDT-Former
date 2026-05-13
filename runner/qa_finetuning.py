"""
MoleculeQA Finetuning Script using HuggingFace Transformers.
Follows the same pattern as finetuning.py for consistency.

This script supports three types of MoleculeQA tasks:
- qa: Multiple choice question answering
- generation: Molecule captioning/description generation
- property: Property value prediction (regression)
"""

import os
import warnings
import argparse
import yaml
import torch
import random
from easydict import EasyDict as edict
from typing import Dict, Optional
from collections import defaultdict
from torch.utils.data import Subset

from transformers import (
    Trainer,
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging
import wandb

logger = logging.get_logger(__name__)

from peft import get_peft_model, LoraConfig, TaskType

from models.configuration import MolLLaMAConfig
from runner.training_args import (
    ModelArguments,
    DataTrainingArguments,
    parse_args_from_yaml,
)
from data_provider.moleculeqa_dataset import create_moleculeqa_datasets
from data_provider.finetune_dataset import determine_llm_version

# Try importing HF dataset loaders (for preprocessed data)
HF_MOLECULEQA_DATASETS_AVAILABLE = False
try:
    from runner.dataset_creator.hf_moleculeqa_dataset import create_hf_moleculeqa_datasets
    HF_MOLECULEQA_DATASETS_AVAILABLE = True
except ImportError:
    logger.warning("HF MoleculeQA dataset loader not available. Only original data loading supported.")

# Import trainers for metrics computation
from runner.trainers.qa import (
    MoleculeQATrainer,
    MoleculeGENQATrainer,
    MoleculePropertyQATrainer,
    MoleculeReactionTrainer,
    MoleculeOpenQuestionTrainer,
)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

## for pyg bug
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
## for A100 gpus
torch.set_float32_matmul_precision("medium")


def main(model_args, training_args, data_config, test_mode=False, resume_from=None):
    """Main MoleculeQA finetuning function.
    
    Args:
        model_args: ModelArguments parsed by HfArgumentParser
        training_args: HuggingFace TrainingArguments parsed directly from YAML
        data_config: DataTrainingArguments parsed by HfArgumentParser
        test_mode: Whether to use small dataset for testing
        resume_from: Path to checkpoint to resume from
    """
    torch.manual_seed(0)
    
    # Set logging level to INFO for better visibility
    logging.set_verbosity_info()
    
    # Initialize tokenizer
    if model_args.llm_backbone is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.llm_backbone, padding_side="left"
        )
    elif model_args.model_name_or_path is not None and not os.path.exists(model_args.model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, 
            padding_side="left",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            "DongkiKim/Mol-Llama-3.1-8B-Instruct",
            padding_side="left",
        )

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["<mol>"]})
    tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    # Check if using mol-llama baseline
    baseline_type = getattr(model_args, 'baseline_type', None)
    use_mollama_baseline = (baseline_type == 'mollama')
    
    # Create model config from parsed arguments
    if use_mollama_baseline:
        # Mol-LLaMA baseline: use config similar to inference.py
        logger.info("Using Mol-LLaMA baseline model configuration")
        # Priority: llm_backbone > model_name_or_path > default
        logger.info(f"🔧 Base LLM model to load: {model_args.model_name_or_path}")
        model_config = MolLLaMAConfig(
            llm_config={'llm_model': model_args.model_name_or_path},
            qformer_config={
                'use_dq_encoder': model_args.use_dq_encoder,
                'use_flash_attention': True,
                'num_query_tokens': 8,
                'embed_dim': 256,
                'cross_attention_freq': 2,
                'max_local_query': 0,
            },
            graph_encoder_config={
                'encoder_types': ['unimol', 'moleculestm'] if model_args.enable_blending else ['unimol']
            },
            blending_module_config={
                'enable_blending': True,
                'num_layers': 4,
                'num_heads': 8
            },
            torch_dtype="float16"
        )
    else:
        # Default EDT-Former configuration
        model_config = MolLLaMAConfig(
            qformer_config={
                "use_flash_attention": model_args.use_flash_attention,
                "use_dq_encoder": model_args.use_dq_encoder,
                "num_query_tokens": model_args.num_query_tokens,
                "embed_dim": model_args.embed_dim,
                "cross_attention_freq": model_args.cross_attention_freq,
                "enable_lora": model_args.enable_lora_qformer,
                "max_local_query": model_args.max_local_query,
            },
            graph_encoder_config={"local_q_only": model_args.local_q_only},
            blending_module_config={
                "num_layers": model_args.num_layers,
                "num_heads": model_args.num_heads,
                "enable_blending": model_args.enable_blending,
            },
        )
        # Priority: llm_backbone > default (don't use model_name_or_path for LLM backbone in edt_former)
        # model_name_or_path is for the full checkpoint, not the LLM backbone
        if model_args.llm_backbone is not None:
            logger.info(f"🔧 Setting LLM backbone from llm_backbone: {model_args.llm_backbone}")
            model_config.llm_config.llm_model = model_args.llm_backbone
        else:
            # Keep default LLM model from config (unsloth/Llama-3.1-8B-Instruct)
            logger.info(f"🔧 Using default LLM backbone: {model_config.llm_config.llm_model}")

    # Determine torch dtype from training_args
    if training_args.bf16:
        torch_dtype = "bfloat16"
    elif training_args.fp16:
        torch_dtype = "float16"
    else:
        torch_dtype = "float32"

    # Determine LLM version from model config
    llm_version = determine_llm_version(model_config.llm_config.llm_model, default="llama3")

    # Load unimol_dictionary first (needed for both data loading paths)
    logger.info("Loading UniMol dictionary...")
    from huggingface_hub import hf_hub_download
    from utils.unicore import Dictionary
    
    unimol_config = model_config.graph_encoder_config.unimol_config
    unimol_dictionary_path = hf_hub_download(
        repo_id=unimol_config.repo_id,
        filename=unimol_config.dictionary_filename,
    )
    unimol_dictionary = Dictionary.load(unimol_dictionary_path)
    unimol_dictionary.add_symbol("[MASK]", is_special=True)
    logger.info(f"✅ Loaded UniMol dictionary with {len(unimol_dictionary)} symbols")
    
    # Determine whether to use preprocessed datasets
    use_preprocessed = getattr(data_config, 'use_preprocessed', False)
    
    if use_preprocessed and HF_MOLECULEQA_DATASETS_AVAILABLE:
        logger.info("=" * 80)
        logger.info("Using PREPROCESSED MoleculeQA datasets (faster loading!)")
        logger.info("=" * 80)
        
        # Get data path - can be local directory or HuggingFace Hub repo
        data_path = getattr(data_config, 'preprocessed_data', None)
        
        if not data_path:
            raise ValueError(
                "use_preprocessed=True but no 'preprocessed_data' specified in data config"
            )
        
        logger.info(f"Loading preprocessed data from: {data_path}")
        
        # Use HF dataset loader - automatically handles local/Hub sources
        datasets, data_collator = create_hf_moleculeqa_datasets(
            data_path=data_path,
            tokenizer=tokenizer,
            llm_version=llm_version,
            pad_idx=unimol_dictionary.pad(),
            encoder_types=model_config.graph_encoder_config.encoder_types,
            mol_type=getattr(data_config, 'mol_type', 'mol'),
            cache_dir=getattr(data_config, 'cache_dir', None),
            streaming=getattr(data_config, 'use_streaming', False),
            max_input_length=getattr(data_config, 'max_input_length', None),
        )
        
        train_dataset = datasets['train']
        val_dataset = datasets.get('test', None)  # MoleculeQA uses 'test' as validation
        test_dataset = datasets.get('test', None)
        
        logger.info(f"✅ Loaded preprocessed datasets")
        
    elif use_preprocessed and not HF_MOLECULEQA_DATASETS_AVAILABLE:
        raise ImportError(
            "Preprocessed data loading requested but HF MoleculeQA dataset loader not available. "
            "Please ensure runner/dataset_creator/hf_moleculeqa_dataset.py exists."
        )
        
    else:
        logger.info("=" * 80)
        logger.info("Using ORIGINAL MoleculeQA datasets (on-the-fly processing)")
        logger.info("=" * 80)
        
        # Determine limits for test mode
        train_limit = 100 if test_mode else None
        val_limit = 50 if test_mode else None
        test_limit = 50 if test_mode else None
        
        # Clear cache if requested (only on rank 0 to avoid race conditions)
        if training_args.local_rank in [-1, 0]:
            if hasattr(training_args, '_clear_cache') and training_args._clear_cache:
                from utils.cache_utils import clear_cache
                clear_cache(verbose=True)
        
        datasets, data_collator = create_moleculeqa_datasets(
            tokenizer=tokenizer,
            llama_version=llm_version,
            root=data_config.root,
            unimol_dictionary=unimol_dictionary,
            encoder_types=model_config.graph_encoder_config.encoder_types,
            mol_type=getattr(data_config, 'mol_type', 'mol'),
            train_limit=train_limit,
            val_limit=val_limit,
            test_limit=test_limit,
            brics_gids_enable=model_args.brics_gids_enable,
            entropy_gids_enable=model_args.entropy_gids_enable,
            use_cache=True,
            max_input_length=getattr(data_config, 'max_input_length', None),
        )
        
        train_dataset = datasets['train']
        val_dataset = datasets['test']
        test_dataset = datasets['test']
    
    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Val size (full): {len(val_dataset)}")
    logger.info(f"Test size (full): {len(test_dataset)}")
    
    # Apply training_ratio if specified (for data efficiency experiments)
    training_ratio = getattr(data_config, 'training_ratio', 1.0)
    if training_ratio < 1.0 and training_ratio > 0.0:
        random.seed(getattr(data_config, 'random_seed', 42))
        target_size = int(len(train_dataset) * training_ratio)
        logger.info(f"⚠️  Using {training_ratio*100:.1f}% of training data: {target_size} samples (from {len(train_dataset)})")
        indices = random.sample(range(len(train_dataset)), target_size)
        train_dataset = Subset(train_dataset, indices)
        logger.info(f"✅ Train size (limited): {len(train_dataset)}")
    elif training_ratio <= 0.0 or training_ratio > 1.0:
        logger.warning(f"⚠️  Invalid training_ratio: {training_ratio}. Must be between 0.0 and 1.0. Using all data.")
    
    # Apply max_eval_samples limit if specified
    max_eval_samples = getattr(data_config, 'max_eval_samples', None)
    if max_eval_samples is not None and max_eval_samples > 0:
        random.seed(getattr(data_config, 'random_seed', 42))
        
        if len(val_dataset) > max_eval_samples:
            logger.info(f"⚠️  Limiting validation set to {max_eval_samples} samples (from {len(val_dataset)}) for faster evaluation")
            indices = random.sample(range(len(val_dataset)), max_eval_samples)
            val_dataset = Subset(val_dataset, indices)
            logger.info(f"✅ Val size (limited): {len(val_dataset)}")
        
        if len(test_dataset) > max_eval_samples:
            logger.info(f"⚠️  Limiting test set to {max_eval_samples} samples (from {len(test_dataset)})")
            indices = random.sample(range(len(test_dataset)), max_eval_samples)
            test_dataset = Subset(test_dataset, indices)
            logger.info(f"✅ Test size (limited): {len(test_dataset)}")
    
    # Calculate total training steps for distributed training
    num_samples = len(train_dataset)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    samples_per_gpu = num_samples // num_gpus if num_gpus > 1 else num_samples
    steps_per_epoch = samples_per_gpu // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    max_steps = steps_per_epoch * training_args.num_train_epochs
    
    # Update max_steps
    training_args.max_steps = max_steps
    
    # Determine task type and use appropriate trainer
    task_type = getattr(data_config, 'task_type', 'qa')
    logger.info(f"Task type: {task_type}")
    
    # Select appropriate trainer based on task type
    # Note: 'caption' and 'generation' are aliases for the same trainer
    if task_type in ['generation', 'caption']:
        trainer_class = MoleculeGENQATrainer
    elif task_type == 'property':
        trainer_class = MoleculePropertyQATrainer
    elif task_type in ['reaction', 'forward_reaction', 'retrosynthesis', 'reagent_prediction', 'description_guided_molecule_design']:
        # Use MoleculeReactionTrainer for all reaction/molecule generation tasks
        trainer_class = MoleculeReactionTrainer
        logger.info(f"Using MoleculeReactionTrainer for task: {task_type}")
    elif task_type in ['open_question', 'openquestion', 'open-question']:
        # Use MoleculeOpenQuestionTrainer for open-ended question answering tasks
        trainer_class = MoleculeOpenQuestionTrainer
        logger.info(f"Using MoleculeOpenQuestionTrainer for task: {task_type}")
    else:  # default to 'qa'
        trainer_class = MoleculeQATrainer
    
    # Initialize WandB
    if training_args.local_rank in [-1, 0]:
        from dataclasses import asdict
        wandb.init(
            project="QA_Bench",
            name=training_args.run_name,
            config={
                "training": training_args.to_dict(),
                "model": asdict(model_args),
                "data": asdict(data_config),
            },
            mode="offline",
        )
    
    # Initialize Trainer with task-specific config and actual datasets
    from easydict import EasyDict
    # For mol-llama baseline, disable brics_gids and entropy_gids (as in inference.py)
    if use_mollama_baseline:
        brics_gids_enable = False
        entropy_gids_enable = False
    else:
        brics_gids_enable = model_args.brics_gids_enable
        entropy_gids_enable = model_args.entropy_gids_enable
    
    train_config = EasyDict({
        'init_lr': training_args.learning_rate,
        'weight_decay': training_args.weight_decay,
        'warmup_steps': training_args.warmup_steps,
        'max_epochs': training_args.num_train_epochs,
        'min_lr': 1e-7,
        'warmup_lr': 1e-6,
        'scheduler': 'linear_warmup_cosine_lr',
        'precision': 'bf16-mixed' if training_args.bf16 else ('fp16' if training_args.fp16 else 'fp32'),
        'enable_flash': model_args.enable_flash,
        'freeze_llm': model_args.freeze_llm,
        'brics_gids_enable': brics_gids_enable,
        'entropy_gids_enable': entropy_gids_enable,
        'enable_blending': model_args.enable_blending,
        'zero_shot': model_args.zero_shot,
        'load_ckpt_before_peft': model_args.load_ckpt_before_peft,
        'ckpt_path': model_args.model_name_or_path if model_args.load_ckpt_before_peft else None,
        'llm_only': getattr(model_args, 'llm_only', False),  # LLM-only mode for text-only tasks
        # For mol-llama baseline: llm_model_path is the base LLM (from model_name_or_path)
        # For edt_former: llm_model_path should be llm_backbone (or None to use default)
        'llm_backbone': model_args.llm_backbone,
        'llm_model_path': model_args.model_name_or_path if use_mollama_baseline else model_args.llm_backbone,
    })
    
    # Create trainer with actual datasets (no workaround needed!)
    trainer = trainer_class(
        vocab_size=len(tokenizer),
        model_config=model_config,
        train_config=train_config,
        tokenizer=tokenizer,
        use_dq_encoder=model_args.use_dq_encoder,
        torch_dtype=torch_dtype,
        # HF Trainer arguments
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if training_args.eval_strategy != "no" else None,
        data_collator=data_collator,
    )
    
    # Load from pretrained checkpoint (Stage 2 full model)
    # Note: Zero-shot mode REQUIRES a checkpoint for evaluation!
    # Skip if checkpoint was already loaded before PEFT in model initialization
    load_ckpt_before_peft = model_args.load_ckpt_before_peft
    logger.info(f"🔍 load_ckpt_before_peft = {load_ckpt_before_peft}")
    logger.info(f"🔍 model_name_or_path = {model_args.model_name_or_path}")
    logger.info(f"🔍 baseline_type = {baseline_type}")
    
    # Checkpoint loading logic:
    # - model_name_or_path: base LLM to load (e.g., "unsloth/Llama-3.1-8B-Instruct")
    # - lora_path: checkpoint with trained weights/adapters to load on top
    ckpt_path = None
    if use_mollama_baseline:
        ckpt_path = getattr(model_args, 'lora_path', None)
        if ckpt_path:
            logger.info(f"Loading Mol-LLaMA checkpoint from lora_path: {ckpt_path}")
            lora_init = getattr(model_args, 'lora_init', False)
            trainer.load_from_ckpt(ckpt_path, lora_init=lora_init)
        elif model_args.zero_shot:
            raise ValueError(
                "Zero-shot mode requires a pretrained checkpoint! "
                "Please provide lora_path in your model config."
            )
        else:
            logger.info("💡 No lora_path provided - training from base LLM weights")
    elif model_args.model_name_or_path and not load_ckpt_before_peft:
        logger.info(f"Loading from pretrained checkpoint: {model_args.model_name_or_path}")
        trainer.load_from_ckpt(model_args.model_name_or_path)
    elif model_args.model_name_or_path and load_ckpt_before_peft:
        logger.info(f"✅ Checkpoint was loaded before PEFT model creation in model initialization")
    elif model_args.zero_shot:
        raise ValueError(
            "Zero-shot mode requires a pretrained checkpoint! "
            "Please provide model_name_or_path in your model config."
        )
    else:
        logger.warning("⚠️  No checkpoint provided - using randomly initialized weights")
        logger.warning("   This is only appropriate for debugging, not for actual training/evaluation")
    
    # Apply LoRA to Qformer if enabled (after checkpoint loading)
    if model_args.enable_lora_qformer:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=model_config.qformer_config.lora_config.r,
            lora_alpha=model_config.qformer_config.lora_config.lora_alpha,
            lora_dropout=model_config.qformer_config.lora_config.lora_dropout,
            target_modules=[
                'query', 'key', 'value', 'output.dense',
                'intermediate.dense', 'output.dense'
            ],
        )
        trainer.model.encoder.Qformer = get_peft_model(trainer.model.encoder.Qformer, peft_config)
        logger.info("LoRA enabled for Qformer")
    
    # Check for existing checkpoints or resume_from parameter
    ckpt_path = None
    if resume_from is not None:
        if resume_from == "last":
            candidate = os.path.join(training_args.output_dir, "checkpoint-last")
            ckpt_path = candidate if os.path.exists(candidate) else None
            if ckpt_path is None:
                # Try get_last_checkpoint
                ckpt_path = get_last_checkpoint(training_args.output_dir)
        else:
            ckpt_path = resume_from if os.path.exists(resume_from) else None
    else:
        # Auto-detect last checkpoint
        if os.path.isdir(training_args.output_dir):
            ckpt_path = get_last_checkpoint(training_args.output_dir)
    
    # Train (skip if zero-shot mode)
    if model_args.zero_shot:
        logger.info("=" * 80)
        logger.info("🎯 Zero-shot evaluation mode: Skipping training, evaluating pretrained model directly")
        logger.info("=" * 80)
    elif ckpt_path is not None:
        logger.info(f"Resuming training from checkpoint: {ckpt_path}")
        trainer.train(resume_from_checkpoint=ckpt_path)
    else:
        logger.info("No resuming checkpoint found, starting finetuning from model")
        trainer.train()
    
    # Evaluate on validation set if available
    if val_dataset is not None and len(val_dataset) > 0 and training_args.eval_strategy != "no":
        logger.info("\nEvaluating on validation set...")
        eval_results = trainer.evaluate()
        logger.info(f"Validation results: {eval_results}")
    
    # Test on test set
    if test_dataset is not None and len(test_dataset) > 0:
        logger.info("\nTesting on test set...")
        test_results = trainer.predict(test_dataset)
        logger.info(f"Test results saved to {training_args.output_dir}")
    
    # Save final model (skip if zero-shot mode)
    if not model_args.zero_shot:
        trainer.save_model(os.path.join(training_args.output_dir, "final_model"))
    
    # Close WandB
    if training_args.local_rank in [-1, 0]:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoleculeQA Finetuning with HF Transformers")
    parser.add_argument(
        "--model_config_path", type=str, default="configs/moleculeqa/edt_former/model_config.yaml",
        help="Path to model configuration YAML file"
    )
    parser.add_argument(
        "--training_config_path", type=str, default="configs/moleculeqa/edt_former/training_config.yaml",
        help="Path to training configuration YAML file (HF TrainingArguments compatible)"
    )
    parser.add_argument(
        "--data_config_path", type=str, default="configs/moleculeqa/edt_former/data_config.yaml",
        help="Path to data configuration YAML file"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help="Use small dataset for testing",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help='Checkpoint path or "last" to resume from latest',
    )
    parser.add_argument(
        "--deepspeed_stage",
        type=int,
        default=0,
        choices=[0, 2, 3],
        help="DeepSpeed ZeRO stage (2 or 3). Default 3 for evaluation/inference support.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (automatically set by DeepSpeed launcher)",
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        default=False,
        help="Clear dataset cache before training (forces fresh data loading)",
    )

    args = parser.parse_args()
    
    # Get BASE_DIR for paths
    BASE_DIR = os.environ.get('BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Determine output directory from training config
    with open(args.training_config_path, 'r') as f:
        training_config_preview = yaml.load(f, Loader=yaml.FullLoader)
    
    # Setup output directory
    run_name = training_config_preview.get('run_name', 'default_run')
    checkpoint_base_dir = os.environ.get('CHECKPOINT_DIR', 'checkpoints')
    output_dir = os.path.join(checkpoint_base_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we're doing zero-shot evaluation (no training)
    # If num_train_epochs == 0, disable DeepSpeed for pure inference
    num_train_epochs = training_config_preview.get('num_train_epochs', 1)
    
    # Setup DeepSpeed config based on command-line argument
    deepspeed_config = None
    if num_train_epochs == 0 or args.deepspeed_stage == 0:
        logger.info("=" * 80)
        logger.info("🎯 num_train_epochs=0 detected: Disabling DeepSpeed for pure evaluation mode")
        logger.info("=" * 80)
        deepspeed_config = None
    elif args.deepspeed_stage in [2, 3]:
        if args.deepspeed_stage == 3:
            deepspeed_config = os.path.join(BASE_DIR, "configs/deepspeed/ds_config_zero3.json")
        else:
            deepspeed_config = os.path.join(BASE_DIR, "configs/deepspeed/ds_config_zero2.json")
        
        if not os.path.exists(deepspeed_config):
            logger.warning(f"Warning: DeepSpeed config not found at {deepspeed_config}")
            deepspeed_config = None
        else:
            logger.info(f"Using DeepSpeed ZeRO-{args.deepspeed_stage} config: {deepspeed_config}")

    # Parse arguments from YAML files using HfArgumentParser
    model_args, training_args, data_config = parse_args_from_yaml(
        model_config_path=args.model_config_path,
        data_config_path=args.data_config_path,
        training_config_path=args.training_config_path,
        output_dir=output_dir,
        deepspeed_config=deepspeed_config,
    )

    logger.info("-" * 60)
    detected_num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(
        f"batch_size: {training_args.per_device_train_batch_size}\tnum_devices: {detected_num_devices}\taccumulate_grad_batches: {training_args.gradient_accumulation_steps}"
    )
    logger.info(
        f"Total batch size: {training_args.per_device_train_batch_size * detected_num_devices * training_args.gradient_accumulation_steps}"
    )
    logger.info("-" * 60)
    
    task_type = getattr(data_config, 'task_type', 'qa')
    mol_type = getattr(data_config, 'mol_type', 'mol')
    training_ratio = getattr(data_config, 'training_ratio', 1.0)
    logger.info(f"Task Type: {task_type}")
    logger.info(f"Mol Type: {mol_type}")
    logger.info(f"Data Root: {data_config.root}")
    logger.info(f"Training Ratio: {training_ratio*100:.1f}% of training data")
    logger.info(f"Zero-shot Mode: {model_args.zero_shot}")
    logger.info("-" * 60)

    if args.test_mode:
        logger.info("TEST MODE: Using small dataset for quick testing")
    
    if args.clear_cache:
        logger.info("CLEAR CACHE MODE: Will rebuild dataset cache from scratch")
        # Pass the flag via training_args as a custom attribute
        training_args._clear_cache = True
    
    main(
        model_args,
        training_args,
        data_config,
        test_mode=args.test_mode,
        resume_from=args.resume_from,
    )


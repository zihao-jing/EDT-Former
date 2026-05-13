import os
import torch
import warnings
import argparse
import yaml
from transformers import (
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging
import wandb

logger = logging.get_logger(__name__)

from peft import get_peft_model, LoraConfig, TaskType

from models.configuration import MolLLaMAConfig
from models.edt_former import EDTFinetuneModel
from runner.training_args import (
    parse_args_from_yaml,
)
from data_provider.finetune_dataset import create_finetune_dataset, determine_llm_version
from runner.trainers.ft import FinetuningTrainer

# Import new preprocessed dataset loaders for finetuning
try:
    from runner.dataset_creator import create_hf_finetune_dataset
    HF_FT_DATASETS_AVAILABLE = True
    logger.info("HuggingFace dataset_creator module available. Using preprocessed dataset loader.")
except ImportError:
    HF_FT_DATASETS_AVAILABLE = False
    logger.warning("HuggingFace dataset_creator module not available. Using original dataset loader.")

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

## for pyg bug
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


def main(model_args, training_args, data_config, test_mode=False, resume_from=None):
    """Main finetuning function.
    
    Args:
        model_args: ModelArguments parsed by HfArgumentParser
        training_args: HuggingFace TrainingArguments parsed directly from YAML
        data_config: DataTrainingArguments parsed by HfArgumentParser
        test_mode: Whether to use small dataset for testing
        resume_from: Path to checkpoint to resume from
    """
    torch.manual_seed(0)
    
    # Get BASE_DIR for DeepSpeed config paths
    BASE_DIR = os.environ.get('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
    
    # Initialize tokenizer
    if model_args.llm_backbone is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.llm_backbone, padding_side="left"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            "DongkiKim/Mol-Llama-3.1-8B-Instruct",
            padding_side="left",
        )

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["<mol>"]})
    tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    mol_id = tokenizer.convert_tokens_to_ids("<mol>")
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")

    # Create model config from parsed arguments
    model_config = MolLLaMAConfig(
        qformer_config={
            "use_flash_attention": model_args.use_flash_attention,
            "use_dq_encoder": model_args.use_dq_encoder,
            "num_query_tokens": model_args.num_query_tokens,
            "embed_dim": model_args.embed_dim,
            "cross_attention_freq": model_args.cross_attention_freq,
            "enable_lora": model_args.enable_lora_qformer,
        },
        graph_encoder_config={"local_q_only": model_args.local_q_only},
        blending_module_config={
            "num_layers": model_args.num_layers,
            "num_heads": model_args.num_heads,
            "enable_blending": model_args.enable_blending,
        },
    )
    
    if model_args.llm_backbone is not None:
        model_config.llm_config.llm_model = model_args.llm_backbone

    # Determine torch dtype from training_args
    if training_args.bf16:
        torch_dtype = "bfloat16"
    elif training_args.fp16:
        torch_dtype = "float16"
    else:
        torch_dtype = "float32"

    # Initialize model
    model = EDTFinetuneModel(
        vocab_size=len(tokenizer),
        model_config=model_config,
        add_ids=[mol_id, pad_id],
        model_args=model_args,
        torch_dtype=torch_dtype,
    )

    # Load from Stage 1 checkpoint
    if model_args.stage1_path:
        logger.info(f"Loading from Stage 1 checkpoint: {model_args.stage1_path}")
        model.load_from_stage1_ckpt(model_args.stage1_path)

    # Apply LoRA to Qformer if enabled
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
        model.model.encoder.Qformer = get_peft_model(model.model.encoder.Qformer, peft_config)
        logger.info("LoRA enabled for Qformer")

    # Determine LLM version from model config
    llm_version = determine_llm_version(model_config.llm_config.llm_model)

    # Determine whether to use preprocessed datasets
    use_preprocessed = getattr(data_config, 'use_preprocessed', False)
    
    if use_preprocessed and HF_FT_DATASETS_AVAILABLE:
        logger.info("=" * 80)
        logger.info("Using PREPROCESSED finetuning datasets (faster loading!)")
        logger.info("=" * 80)
        
        # Get data path - can be local directory, local file, or HuggingFace Hub repo
        # HF datasets library automatically detects and handles all cases!
        data_path = getattr(data_config, 'preprocessed_data', None)
        
        if not data_path:
            raise ValueError(
                "use_preprocessed=True but no 'preprocessed_data' specified in data config"
            )
        
        logger.info(f"Loading preprocessed data from: {data_path}")
        
        # Use unified loading function - HF datasets handles local/Hub automatically
        train_dataset, val_dataset, data_collator = create_hf_finetune_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            llm_version=llm_version,
            pad_idx=model.model.encoder.unimol_dictionary.pad(),
            encoder_types=model_config.graph_encoder_config.encoder_types,
            streaming=getattr(data_config, 'use_streaming', False),
            cache_dir=getattr(data_config, 'cache_dir', None),
            val_ratio=getattr(data_config, 'val_ratio', 0.01),
            random_seed=getattr(data_config, 'random_seed', 42),
        )
        
        logger.info("✅ Preprocessed finetuning datasets loaded successfully!")
        logger.info("=" * 80)
    else:
        # Use original dataset loader (with on-the-fly processing)
        if use_preprocessed and not HF_FT_DATASETS_AVAILABLE:
            logger.warning("⚠️  use_preprocessed=True but HF ft_datasets not available. Falling back to original loader.")
        
        logger.info("=" * 80)
        logger.info("Using ORIGINAL dataset loader (on-the-fly processing)")
        logger.info("=" * 80)
        
        # Create dataset and collator using pure HuggingFace approach
        train_dataset, data_collator = create_finetune_dataset(
            tokenizer=tokenizer,
            llm_version=llm_version,
            root=data_config.root,
            unimol_dictionary=model.model.encoder.unimol_dictionary,
            encoder_types=model_config.graph_encoder_config.encoder_types,
            data_types=data_config.data_types,
            test_mode=test_mode,
            brics_gids_enable=model_args.brics_gids_enable,
            entropy_gids_enable=model_args.entropy_gids_enable,
        )
        
        logger.info("=" * 80)
    
    # Calculate total training steps for distributed training
    # Note: HF Trainer will create DataLoader internally with DistributedSampler
    num_samples = len(train_dataset)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # With distributed training, each GPU processes num_samples/num_gpus samples
    # Each step processes batch_size samples per GPU
    # After accumulate_grad_batches micro-steps, we do one optimizer step
    samples_per_gpu = num_samples // num_gpus if num_gpus > 1 else num_samples
    steps_per_epoch = samples_per_gpu // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    max_steps = steps_per_epoch * training_args.num_train_epochs
    
    # Training args already parsed from YAML - just need to update max_steps
    training_args.max_steps = max_steps
    
    # Initialize WandB
    if training_args.local_rank in [-1, 0]:
        from dataclasses import asdict
        wandb.init(
            project="EDT_Finetuning",  # Changed from "Stage2"
            name=training_args.run_name,
            config={
                "training": training_args.to_dict(),
                "model": asdict(model_args),
                "data": asdict(data_config),
            },
            # mode="offline",
        )
    
    # Initialize Trainer with pure HuggingFace components
    trainer = FinetuningTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if training_args.eval_strategy != "no" else None,
        data_collator=data_collator,
    )
    
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
    
    # Train
    if ckpt_path is not None:
        logger.info(f"Resuming training from checkpoint: {ckpt_path}")
        trainer.train(resume_from_checkpoint=ckpt_path)
    else:
        logger.info("No resuming checkpoint found, starting finetuning from model")
        trainer.train()
    
    # Save final model
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))
    
    # Close WandB
    if training_args.local_rank in [-1, 0]:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetuning (Stage 2) with HF Transformers")
    parser.add_argument(
        "--model_config_path", type=str, default="configs/stage2/model_config.yaml",
        help="Path to model configuration YAML file"
    )
    parser.add_argument(
        "--training_config_path", type=str, default="configs/stage2/training_config.yaml",
        help="Path to training configuration YAML file (HF TrainingArguments compatible)"
    )
    parser.add_argument(
        "--data_config_path", type=str, default="configs/stage2/data_config.yaml",
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
        default=2,
        choices=[2, 3],
        help="DeepSpeed ZeRO stage (2 or 3)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (automatically set by DeepSpeed launcher)",
    )

    args = parser.parse_args()
    
    # Get BASE_DIR for paths
    BASE_DIR = os.environ.get('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
    
    # Determine output directory from training config
    with open(args.training_config_path, 'r') as f:
        training_config_preview = yaml.load(f, Loader=yaml.FullLoader)
    
    # Setup output directory
    run_name = training_config_preview.get('run_name', 'default_run')
    checkpoint_base_dir = os.environ.get('CHECKPOINT_DIR', 'checkpoints')
    output_dir = os.path.join(checkpoint_base_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup DeepSpeed config based on command-line argument
    deepspeed_config = None
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
    if data_config.data_types:
        logger.info(f"Data Types:")
        for data_type in data_config.data_types:
            logger.info(f"  - {data_type}")
        logger.info("-" * 60)

    if args.test_mode:
        logger.info("TEST MODE: Using small dataset for quick testing")
    
    main(
        model_args,
        training_args,
        data_config,
        test_mode=args.test_mode,
        resume_from=args.resume_from,
    )


import os
from tkinter import FALSE
import torch
import warnings
import argparse
import yaml
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging
import wandb

# Configure transformers logging to show INFO level
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

from models.configuration import MolLLaMAConfig
from models.edt_former import EDTPretrainModel
from runner.training_args import (
    parse_args_from_yaml,
)
from data_provider.pretrain_dataset import create_pretrain_datasets
from runner.trainers.pretrain import PretrainTrainer, LossLoggingCallback

# Import new preprocessed dataset loaders
try:
    from runner.dataset_creator import create_hf_pretrain_datasets
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    logger.warning("HuggingFace datasets module not available. Using original dataset loader.")

## for pyg bug
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


def main(model_args, training_args, data_config, test_mode=False):
    """Main training function.
    
    Args:
        model_args: ModelArguments parsed by HfArgumentParser
        training_args: HuggingFace TrainingArguments parsed directly from YAML
        data_config: DataTrainingArguments parsed by HfArgumentParser
        test_mode: Whether to use small dataset for testing
    """
    torch.manual_seed(0)
    
    # Get BASE_DIR for DeepSpeed config paths
    BASE_DIR = os.environ.get('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
    
    # Create model config from parsed arguments
    model_config = MolLLaMAConfig(
        qformer_config={
            "use_flash_attention": model_args.use_flash_attention,
            "use_dq_encoder": model_args.use_dq_encoder,
            "num_query_tokens": model_args.num_query_tokens,
            "embed_dim": model_args.embed_dim,
            "cross_attention_freq": model_args.cross_attention_freq,
        },
        graph_encoder_config={"local_q_only": model_args.local_q_only},
        blending_module_config={
            "num_layers": model_args.num_layers,
            "num_heads": model_args.num_heads,
            "enable_blending": model_args.enable_blending,
        },
    )
    
    if model_args.enable_blending:
        model_config.graph_encoder_config.encoder_types = ["unimol", "moleculestm"]
        logger.warning(f"Caution: Using blending module" + "-" * 10)
    
    # Initialize model
    model = EDTPretrainModel(model_config, model_args)
    
    # Determine whether to use preprocessed datasets
    use_preprocessed = getattr(data_config, 'use_preprocessed', False)
    
    if use_preprocessed and HF_DATASETS_AVAILABLE:
        logger.info("=" * 80)
        logger.info("Using PREPROCESSED datasets (faster loading!)")
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
        train_dataset, val_dataset, data_collator = create_hf_pretrain_datasets(
            data_path=data_path,
            tokenizer=model.encoder.scibert_tokenizer,
            text_max_len=data_config.text_max_len,
            pad_idx=(
                model.encoder.unimol_dictionary.pad()
                if "unimol" in model_config.graph_encoder_config.encoder_types
                else 0
            ),
            encoder_types=model_config.graph_encoder_config.encoder_types,
            streaming=getattr(data_config, 'use_streaming', False),
            cache_dir=getattr(data_config, 'cache_dir', None),
            val_ratio=getattr(data_config, 'val_ratio', 0.01),
            random_seed=getattr(data_config, 'random_seed', 42),
        )
        
        logger.info("✅ Preprocessed datasets loaded successfully!")
        logger.info("=" * 80)
    else:
        # Use original dataset loader (with on-the-fly processing)
        if use_preprocessed and not HF_DATASETS_AVAILABLE:
            logger.warning("⚠️  use_preprocessed=True but HF datasets not available. Falling back to original loader.")
        
        logger.info("=" * 80)
        logger.info("Using ORIGINAL dataset loader (on-the-fly processing)")
        logger.info("=" * 80)
        
        train_dataset, val_dataset, data_collator = create_pretrain_datasets(
            unimol_dictionary=(
                model.encoder.unimol_dictionary
                if "unimol" in model_config.graph_encoder_config.encoder_types
                else None
            ),
            scibert_tokenizer=model.encoder.scibert_tokenizer,
            encoder_types=model_config.graph_encoder_config.encoder_types,
            text_max_len=data_config.text_max_len,
            root=data_config.root,
            test_mode=test_mode,
            brics_gids_enable=model_args.brics_gids_enable,
            entropy_gids_enable=model_args.entropy_gids_enable,
            use_cache=True,
        )
        
        logger.info("=" * 80)
    
    # Calculate total training steps for distributed training
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    samples_per_gpu = len(train_dataset) // num_gpus if num_gpus > 1 else len(train_dataset)
    steps_per_epoch = samples_per_gpu // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    max_steps = steps_per_epoch * training_args.num_train_epochs
    
    # Training args already parsed from YAML - just need to update max_steps
    training_args.max_steps = max_steps
    
    # Initialize WandB
    if training_args.local_rank in [-1, 0]:
        from dataclasses import asdict
        wandb.init(
            project="edt_former_pretrain",
            name=training_args.run_name,
            config={
                "training": training_args.to_dict(),
                "model": asdict(model_args),
                "data": asdict(data_config),
            },
            # mode="offline",  # Change to "online" if you want online logging
        )
    
    # Create callbacks
    callbacks = [LossLoggingCallback()]
    
    # Initialize Trainer
    trainer = PretrainTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    # Check for existing checkpoints
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    if last_checkpoint is not None:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info("No checkpoint found, starting training from scratch")
        trainer.train() 
    
    # Save final model
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))
    
    # Close WandB
    if training_args.local_rank in [-1, 0]:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 training with HF Transformers")
    parser.add_argument(
        "--model_config_path", type=str, default="configs/stage1/model_config.yaml",
        help="Path to model configuration YAML file"
    )
    parser.add_argument(
        "--training_config_path", type=str, default="configs/stage1/training_config.yaml",
        help="Path to training configuration YAML file (HF TrainingArguments compatible)"
    )
    parser.add_argument(
        "--data_config_path", type=str, default="configs/stage1/data_config.yaml",
        help="Path to data configuration YAML file"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help="Use small dataset for testing",
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
    if args.test_mode:
        logger.info("TEST MODE: Using small dataset for quick testing")
    logger.info("-" * 60)

    main(model_args, training_args, data_config, test_mode=args.test_mode)


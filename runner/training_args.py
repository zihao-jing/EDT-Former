"""
Shared training argument classes for HuggingFace Trainer-based training scripts.
Used by stage1_hf.py and finetuning_hf.py (stage2).

These dataclasses work with HfArgumentParser for automatic parsing from:
- Command-line arguments
- YAML/JSON configuration files
- Python dictionaries
"""
import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments, HfArgumentParser
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Can be used for both Stage1 and Stage2/Finetuning.
    
    Use with HfArgumentParser to parse from YAML/JSON or command-line.
    """
    
    batch_size: int = field(
        default=48,
        metadata={"help": "Batch size per device during training and evaluation (deprecated, use per_device_train_batch_size in TrainingArguments)."}
    )
    num_workers: int = field(
        default=8,
        metadata={"help": "Number of subprocesses to use for data loading (deprecated, use dataloader_num_workers in TrainingArguments)."}
    )
    root: str = field(
        default="data/Mol-LLaMA-Instruct/",
        metadata={"help": "Root directory containing the dataset files."}
    )
    
    # Stage1-specific
    text_max_len: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum length of text sequences (IUPAC names). Used in Stage1."}
    )
    
    # Stage2-specific  
    data_types: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of data types to use for training (e.g., ['moleculenet', 'llm_qa']). Used in Stage2."}
    )
    
    # MoleculeQA-specific
    task_type: Optional[str] = field(
        default="qa",
        metadata={"help": "Task type for MoleculeQA: 'qa' (multiple choice), 'generation' (captioning), or 'property' (regression)."}
    )
    mol_type: Optional[str] = field(
        default="mol",
        metadata={"help": "Molecular representation type: 'mol', 'SMILES', 'SMILES,mol', 'SELFIES', 'SMILES,<graph>'."}
    )
    
    # Preprocessed dataset options
    use_preprocessed: bool = field(
        default=False,
        metadata={"help": "Whether to use preprocessed datasets (faster loading)."}
    )
    preprocessed_data: Optional[str] = field(
        default=None,
        metadata={"help": "Path to preprocessed data (local directory or HuggingFace Hub repo). HF automatically detects the type."}
    )
    use_streaming: bool = field(
        default=False,
        metadata={"help": "Whether to use streaming mode for very large datasets (requires preprocessed data from HF Hub)."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for HuggingFace datasets (optional)."}
    )
    val_ratio: float = field(
        default=0.01,
        metadata={"help": "Validation set ratio if val.jsonl doesn't exist (only for finetuning, default: 0.1)."}
    )
    random_seed: int = field(
        default=42,
        metadata={"help": "Random seed for data splitting (default: 42)."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of evaluation samples to use (for faster evaluation during training). None means use all samples."}
    )
    training_ratio: float = field(
        default=1.0,
        metadata={"help": "Fraction of training data to use (0.0-1.0). Useful for data efficiency experiments. Default: 1.0 (use all data)."}
    )
    max_input_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum input token length for truncation. Sequences exceeding this will be truncated to prevent GPU OOM. None means no truncation. Recommended: 1024-4096."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to model configuration.
    Can be used for both Stage1 and Stage2/Finetuning.
    
    Use with HfArgumentParser to parse from YAML/JSON or command-line.
    """
    
    # Model architecture
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Whether to use Flash Attention in Qformer."}
    )
    use_dq_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to use DQ encoder."}
    )
    num_query_tokens: int = field(
        default=8,
        metadata={"help": "Number of query tokens in Qformer."}
    )
    embed_dim: int = field(
        default=256,
        metadata={"help": "Embedding dimension."}
    )
    cross_attention_freq: int = field(
        default=2,
        metadata={"help": "Frequency of cross-attention layers."}
    )
    local_q_only: bool = field(
        default=False,
        metadata={"help": "Whether to use local queries only."}
    )
    enable_blending: bool = field(
        default=False,
        metadata={"help": "Whether to enable blending module."}
    )
    num_layers: int = field(
        default=4,
        metadata={"help": "Number of layers in blending module."}
    )
    num_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads in blending module."}
    )
    max_local_query: int = field(
        default=64,
        metadata={"help": "Maximum number of local queries."}
    )
    # Pretraining stage-specific model settings
    tune_gnn: bool = field(
        default=True,
        metadata={"help": "Whether to fine-tune GNN encoder."}
    )
    temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature parameter for contrastive learning."}
    )
    brics_gids_enable: bool = field(
        default=False,
        metadata={"help": "Whether to enable BRICS graph ID features."}
    )
    entropy_gids_enable: bool = field(
        default=False,
        metadata={"help": "Whether to enable entropy graph ID features."}
    )
    enable_flash: bool = field(
        default=True,
        metadata={"help": "Whether to enable Flash Attention in LLM. Used in Stage2."}
    )
    
    # Finetuning stage-specific
    enable_lora_qformer: bool = field(
        default=False,
        metadata={"help": "Whether to enable LoRA for Qformer. Used in Stage2."}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Whether to freeze LLM parameters. Used in Stage2."}
    )
    llm_backbone: Optional[str] = field(
        default=None,
        metadata={"help": "LLM backbone model path. Used in Stage2."}
    )
    stage1_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to Stage1 checkpoint for loading encoder. Used in Stage2 finetuning."}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained checkpoint (Stage2 or later) to load from. Auto-detects checkpoint type. Used in downstream tasks."}
    )
    zero_shot: bool = field(
        default=False,
        metadata={"help": "Whether to perform zero-shot evaluation without loading checkpoint weights."}
    )
    load_ckpt_before_peft: bool = field(
        default=False,
        metadata={"help": "Whether to load checkpoint before creating PEFT model (in mol_llama.py) instead of after trainer creation."}
    )
    llm_only: bool = field(
        default=False,
        metadata={"help": "LLM-only mode: Skip encoder initialization entirely for text-only tasks (saves ~20-25 GB GPU memory)."}
    )
    
    # Baseline model support
    baseline_type: Optional[str] = field(
        default=None,
        metadata={"help": "Baseline model type: 'mollama' (Mol-LLaMA baseline), 'llm_lora' (LLM with LoRA), 'llm_only' (LLM only), or None (default EDT-Former)."}
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to Mol-LLaMA checkpoint (used when baseline_type='mollama')."}
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "Whether to initialize LoRA when loading Mol-LLaMA checkpoint (used when baseline_type='mollama')."}
    )
    # LLM baseline (generic) support
    llm_baseline: bool = field(
        default=False,
        metadata={"help": "Enable LLM-only baseline in trainers (skip molecular encoders)."}
    )
    llm_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face model id or local path for LLM baseline (fallback to llm_backbone if unset)."}
    )


def parse_args_from_yaml(
    model_config_path: str,
    data_config_path: str,
    training_config_path: str,
    output_dir: str,
    deepspeed_config: Optional[str] = None,
) -> tuple:
    """
    Parse training arguments from YAML config files using HfArgumentParser.
    Directly parses HuggingFace TrainingArguments from YAML without custom mapping.
    
    Args:
        model_config_path: Path to model config YAML
        data_config_path: Path to data config YAML  
        training_config_path: Path to training config YAML (HF TrainingArguments compatible)
        output_dir: Output directory for checkpoints (required by TrainingArguments)
        deepspeed_config: Optional path to DeepSpeed config file
        
    Returns:
        Tuple of (ModelArguments, TrainingArguments, DataTrainingArguments)
    """
    import yaml
    
    # Load YAML configs
    with open(model_config_path, 'r') as f:
        model_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(data_config_path, 'r') as f:
        data_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(training_config_path, 'r') as f:
        training_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    # Add required fields to training config
    training_config_dict['output_dir'] = output_dir
    if deepspeed_config:
        training_config_dict['deepspeed'] = deepspeed_config
    
    # Set local_rank from environment
    training_config_dict['local_rank'] = int(os.environ.get("LOCAL_RANK", -1))
    
    # Create parser for all argument types
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataTrainingArguments))
    
    # Merge model config into training config for parsing
    merged_config = {**model_config_dict, **training_config_dict, **data_config_dict}
    
    # Parse from dictionaries - YAML keys now match HF TrainingArguments
    model_args, training_args, data_args = parser.parse_dict(merged_config)
    
    # Adjust num_workers for multi-GPU to prevent CPU contention
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            # Get original workers from config or default
            original_workers = training_args.dataloader_num_workers
            if original_workers is None:
                original_workers = data_args.num_workers
            
            # Scale down workers per GPU to keep total workers reasonable
            adjusted_workers = max(2, min(original_workers, 12 // num_gpus))
            if adjusted_workers != original_workers:
                logger.warning(f"⚠️  Multi-GPU detected: Adjusting dataloader_num_workers from {original_workers} to {adjusted_workers} per GPU")
                logger.warning(f"   Total workers: {num_gpus} GPUs × {adjusted_workers} workers = {num_gpus * adjusted_workers} workers")
                logger.warning(f"   (This prevents CPU contention in data loading)")
                # Update the training_args
                training_args.dataloader_num_workers = adjusted_workers
    
    return model_args, training_args, data_args

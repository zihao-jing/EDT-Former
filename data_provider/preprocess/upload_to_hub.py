#!/usr/bin/env python3
"""
Utility script to upload preprocessed datasets to HuggingFace Hub.

This makes it easy to share preprocessed datasets with collaborators
or use them across different machines.

Usage:
    python runner/datasets/upload_to_hub.py \
        --jsonl_path data/preprocessed/molecules.jsonl \
        --repo_id username/dataset-name \
        --private
"""
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def upload_preprocessed_dataset(
    jsonl_path: str,
    repo_id: str,
    private: bool = True,
    token: str = None
):
    """
    Upload preprocessed dataset to HuggingFace Hub.
    
    Args:
        jsonl_path: Path to preprocessed JSONL file
        repo_id: Repository ID on HuggingFace Hub (e.g., 'username/dataset-name')
        private: Whether to make the repository private
        token: HuggingFace token (optional, will use cached token if not provided)
    """
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("Please install required packages:")
        logger.error("  pip install datasets huggingface-hub")
        return False
    
    logger.info("=" * 80)
    logger.info("Uploading Preprocessed Dataset to HuggingFace Hub")
    logger.info("=" * 80)
    logger.info(f"JSONL file: {jsonl_path}")
    logger.info(f"Repository: {repo_id}")
    logger.info(f"Private: {private}")
    logger.info("=" * 80)
    
    # Check if file exists
    if not Path(jsonl_path).exists():
        logger.error(f"File not found: {jsonl_path}")
        return False
    
    try:
        # Load dataset from JSONL
        logger.info("Loading dataset from JSONL...")
        dataset = load_dataset('json', data_files=jsonl_path)
        
        # Log dataset info
        logger.info(f"Dataset loaded: {len(dataset['train'])} samples")
        logger.info(f"Features: {dataset['train'].features.keys()}")
        
        # Push to Hub
        logger.info(f"Uploading to HuggingFace Hub: {repo_id}")
        logger.info("This may take a while depending on dataset size...")
        
        dataset.push_to_hub(
            repo_id,
            private=private,
            token=token
        )
        
        logger.info("=" * 80)
        logger.info("✅ Upload complete!")
        logger.info("=" * 80)
        logger.info(f"Dataset URL: https://huggingface.co/datasets/{repo_id}")
        logger.info(f"\nTo use in training:")
        logger.info(f"  from runner.datasets.hf_pretrain_dataset import load_hf_pretrain_dataset_from_hub")
        logger.info(f"  train_dataset, val_dataset, collator = load_hf_pretrain_dataset_from_hub(")
        logger.info(f"      repo_id='{repo_id}',")
        logger.info(f"      tokenizer=tokenizer,")
        logger.info(f"      text_max_len=128,")
        logger.info(f"      pad_idx=0,")
        logger.info(f"      encoder_types=['unimol', 'moleculestm']")
        logger.info(f"  )")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        logger.error("\nPossible solutions:")
        logger.error("1. Make sure you're logged in:")
        logger.error("     huggingface-cli login")
        logger.error("2. Check your token has write permissions")
        logger.error("3. Verify the repository name is valid")
        return False


def create_dataset_card(
    repo_id: str,
    jsonl_path: str,
    num_samples: int,
    encoder_types: list,
    source_data: str = None
):
    """
    Create a dataset card (README) for the uploaded dataset.
    """
    card_content = f"""---
license: mit
task_categories:
- text-to-structure
- structure-to-text
tags:
- chemistry
- molecules
- molecular-graphs
- pretraining
size_categories:
- {_get_size_category(num_samples)}
---

# {repo_id.split('/')[-1]}

This is a preprocessed molecular dataset for EDT-Former pretraining.

## Dataset Description

- **Samples**: {num_samples:,}
- **Encoder Types**: {', '.join(encoder_types)}
- **Format**: JSONL with preprocessed graph representations
{f'- **Source**: {source_data}' if source_data else ''}

## Dataset Structure

Each sample contains:

- `cid`: Compound ID
- `split`: Train/validation split ('pretrain' or 'valid')
- `iupac_name`: IUPAC chemical name
- `smiles`: SMILES string
- `graph_data`: Preprocessed graph representations
  - `unimol`: UniMol 3D molecular representation (if enabled)
    - `src_tokens`: Tokenized atoms
    - `src_edge_type`: Edge type matrix
    - `src_distance`: Distance matrix
  - `moleculestm`: MoleculeSTM 2D graph (if enabled)
    - `node_feat`: Node features
    - `edge_index`: Edge connectivity
    - `edge_attr`: Edge attributes
- `brics_gids`: BRICS group IDs (optional)
- `entropy_gids`: Entropy group IDs (optional)

## Usage

```python
from runner.datasets.hf_pretrain_dataset import load_hf_pretrain_dataset_from_hub
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

train_dataset, val_dataset, collator = load_hf_pretrain_dataset_from_hub(
    repo_id='{repo_id}',
    tokenizer=tokenizer,
    text_max_len=128,
    pad_idx=0,
    encoder_types={encoder_types}
)
```

## Preprocessing

This dataset was preprocessed using the EDT-Former preprocessing pipeline:

```bash
python runner/datasets/preprocess_pretrain_data.py \\
    --input_json <source_file> \\
    --output_jsonl {jsonl_path} \\
    --encoder_types {' '.join(encoder_types)}
```

## Citation

If you use this dataset, please cite the EDT-Former paper:

```bibtex
@inproceedings{{jing2026edtformer,
  title={{Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding}},
  author={{Jing, Zihao and Zeng, Qiuhao and Fang, Ruiyi and Sun, Yan and Wang, Boyu and Hu, Pingzhao}},
  booktitle={{International Conference on Learning Representations (ICLR)}},
  year={{2026}}
}}
```

## License

Same as EDT-Former project.
"""
    return card_content


def _get_size_category(num_samples):
    """Get HuggingFace size category."""
    if num_samples < 1000:
        return "n<1K"
    elif num_samples < 10000:
        return "1K<n<10K"
    elif num_samples < 100000:
        return "10K<n<100K"
    elif num_samples < 1000000:
        return "100K<n<1M"
    else:
        return "n>1M"


def main():
    parser = argparse.ArgumentParser(
        description="Upload preprocessed dataset to HuggingFace Hub"
    )
    parser.add_argument(
        '--jsonl_path',
        type=str,
        required=True,
        help='Path to preprocessed JSONL file'
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        required=True,
        help='Repository ID on HuggingFace Hub (e.g., username/dataset-name)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make repository private (default: public)'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='HuggingFace token (optional, will use cached token if not provided)'
    )
    
    args = parser.parse_args()
    
    success = upload_preprocessed_dataset(
        jsonl_path=args.jsonl_path,
        repo_id=args.repo_id,
        private=args.private,
        token=args.token
    )
    
    if not success:
        exit(1)


if __name__ == '__main__':
    main()


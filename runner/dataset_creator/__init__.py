"""
Preprocessed datasets module for EDT-Former.

This module provides tools for:
1. Preprocessing molecular data into JSONL format
2. Loading preprocessed data from local files or HuggingFace Hub
3. Uploading preprocessed data to HuggingFace Hub

Main components:
- preprocess_pretrain_data: Script for preprocessing molecular data
- hf_pretrain_dataset: HuggingFace-compatible dataset loader
- upload_to_hub: Utility for uploading datasets to HF Hub
"""

from .hf_pretrain_dataset import (
    HFPretrainDataset,
    HFPretrainCollator,
    create_hf_pretrain_datasets,
)

from .hf_finetune_dataset import (
    HFFinetuneDataset,
    HFFinetuneCollator,
    create_hf_finetune_dataset,
)

from .hf_moleculeqa_dataset import (
    HFMoleculeQADataset,
    HFMoleculeQACollator,
    create_hf_moleculeqa_datasets,
)

__all__ = [
    'HFPretrainDataset',
    'HFPretrainCollator',
    'create_hf_pretrain_datasets',
    'HFFinetuneDataset',
    'HFFinetuneCollator',
    'create_hf_finetune_dataset',
    'HFMoleculeQADataset',
    'HFMoleculeQACollator',
    'create_hf_moleculeqa_datasets',
]

__version__ = '1.0.0'


"""
Preprocessing scripts module for EDT-Former.

This module provides tools for:
1. Preprocessing molecular and instruction data into JSONL format
2. Supporting pretraining, finetuning, and MoleculeQA datasets
3. Uploading preprocessed data to HuggingFace Hub

Main components:
- preprocess_pretrain_data: Script for preprocessing pretraining molecular data
- preprocess_finetune_data: Script for preprocessing finetuning instruction data
- preprocess_moleculeqa_data: Script for preprocessing MoleculeQA data
- upload_to_hub: Script for uploading to HuggingFace Hub
"""

__all__ = [
    'preprocess_pretrain_data',
    'preprocess_finetune_data',
    'preprocess_moleculeqa_data',
    'upload_to_hub',
]

__version__ = '1.0.0'


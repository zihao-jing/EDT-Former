import torch
import torch.nn as nn
from typing import Dict, Optional

from models.mol_llama_encoder import MolLLaMAEncoder
from models.edt_former_encoder import EDTFormerEncoder
from models.mol_llama import EDTFormer


class EDTPretrainModel(nn.Module):
    """
    Stage1 Model for Hugging Face Trainer.
    Converted from PyTorch Lightning LightningModule.
    """
    def __init__(self, model_config, model_args):
        super().__init__()
        self.model_args = model_args
        
        # Read enable_blending from the config that was already set
        enable_blending = getattr(model_config.blending_module_config, 'enable_blending', False)
        
        # Choose encoder based on configuration
        if model_config.qformer_config.use_dq_encoder:
            print("Using EDTFormerEncoder")
            self.encoder = EDTFormerEncoder(
                graph_encoder_config=model_config.graph_encoder_config,
                blending_module_config=model_config.blending_module_config,
                qformer_config=model_config.qformer_config,
                temperature=model_args.temperature,
                tune_gnn=model_args.tune_gnn,
                enable_blending=enable_blending,
                brics_gids_enable=model_args.brics_gids_enable,
                entropy_gids_enable=model_args.entropy_gids_enable,
            )
        else:
            print("Using MolLLaMAEncoder")
            self.encoder = MolLLaMAEncoder(
                graph_encoder_config=model_config.graph_encoder_config,
                blending_module_config=model_config.blending_module_config,
                qformer_config=model_config.qformer_config,
                temperature=model_args.temperature,
                tune_gnn=model_args.tune_gnn,
                enable_blending=enable_blending,
            )
    
    def forward(
        self,
        graph_batch: Dict,
        text_batch: Dict,
        return_dict: bool = True,
    ):
        """
        Forward pass that computes the loss.
        
        Args:
            graph_batch: Dictionary containing graph data (includes brics_gids and entropy_gids)
            text_batch: Dictionary containing text data
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary with 'loss' and individual loss components
        """
        # compute_loss will extract brics_gids and entropy_gids from graph_batch
        loss_dict = self.encoder.compute_loss(graph_batch, text_batch)
        
        if return_dict:
            return loss_dict
        else:
            return loss_dict['loss']
    
    def get_encoder(self):
        """Get the encoder for inference or checkpoint loading."""
        return self.encoder

class EDTFinetuneModel(nn.Module):
    """
    Finetuning Model (Stage 2) for Hugging Face Trainer.
    Converted from PyTorch Lightning LightningModule.
    """
    def __init__(self, vocab_size, model_config, add_ids, model_args, torch_dtype="bfloat16"):
        super().__init__()
        self.model_args = model_args

        # Choose model based on configuration
        # Read enable_blending from the config that was already set
        enable_blending = getattr(model_config.blending_module_config, 'enable_blending', False)
        
        print(f"Using EDTFormer, enable_blending: {enable_blending}")
        self.model = EDTFormer(
            config=model_config,
            vocab_size=vocab_size,
            torch_dtype=torch_dtype,
            enable_flash=model_args.enable_flash,
            add_ids=add_ids,
            freeze_llm=model_args.freeze_llm,
            brics_gids_enable=model_args.brics_gids_enable,
            entropy_gids_enable=model_args.entropy_gids_enable,
            enable_blending=enable_blending,
        )

    def load_from_stage1_ckpt(self, ckpt_path):
        """Load encoder weights from Stage 1 checkpoint."""
        self.model.load_from_stage1_ckpt(ckpt_path)

    def forward(
        self,
        graph_batch: Dict,
        text_batch: Dict,
        return_dict: bool = True,
    ):
        """
        Forward pass that computes the loss.
        
        Args:
            graph_batch: Dictionary containing graph data (includes brics_gids, entropy_gids)
            text_batch: Dictionary containing text data  
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary with 'loss' or just loss tensor
        """
        output = self.model(graph_batch, text_batch)
        
        if return_dict:
            return output
        else:
            return output['loss']
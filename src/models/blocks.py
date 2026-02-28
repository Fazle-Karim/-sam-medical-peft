"""
SAM-Compatible Additive Deep Prompt Encoder with Adapters
Handles dynamic input resolution via positional embedding interpolation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from src.models.adapters import create_adapters_for_sam

logger = logging.getLogger(__name__)


class PromptedSAMEncoder(nn.Module):
    """
    SAM Image Encoder with:
    - Layer-wise additive prompts
    - Optional bottleneck adapters
    Fully compatible with SAM ViT architecture.
    """

    def __init__(self, sam_encoder: nn.Module, use_adapters: bool = True, adapter_layers: list = None):
        super().__init__()

        self.patch_embed = sam_encoder.patch_embed
        self.blocks = sam_encoder.blocks
        self.pos_embed = sam_encoder.pos_embed
        self.neck = sam_encoder.neck
        self.use_adapters = use_adapters

        self.embed_dim = sam_encoder.pos_embed.shape[-1]
        self.num_layers = len(self.blocks)

        # ===============================
        # 1. Deep Additive Prompts
        # ===============================
        self.prompt_embeddings = nn.Parameter(
            torch.randn(self.num_layers, 1, 1, 1, self.embed_dim) * 0.02
        )
        logger.info(f"Initialized {self.num_layers} layers of additive prompts")
        logger.info(f"  Prompt parameters: {self.prompt_embeddings.numel():,}")

        # ===============================
        # 2. Bottleneck Adapters (Optional)
        # ===============================
        if use_adapters:
            if adapter_layers is None:
                # Default: adapters in last 4 layers (efficient)
                adapter_layers = [8, 9, 10, 11]
            
            # Insert adapters into SAM encoder
            self.adapters = create_adapters_for_sam(
                sam_encoder,
                adapter_layers=adapter_layers,
                adapter_positions=['attn', 'mlp']
            )
            
            # Count adapter parameters
            adapter_params = sum(p.numel() for p in self.adapters.parameters())
            logger.info(f"Initialized adapters in layers {adapter_layers}")
            logger.info(f"  Adapter parameters: {adapter_params:,}")
            
            # Freeze everything except prompts and adapters
            for name, param in self.named_parameters():
                if "prompt_embeddings" not in name and "adapter" not in name:
                    param.requires_grad = False
        else:
            self.adapters = None
            # Freeze everything except prompts
            for name, param in self.named_parameters():
                if "prompt_embeddings" not in name:
                    param.requires_grad = False

        # Log trainable parameter count
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters in encoder: {trainable:,}")

    def interpolate_pos_encoding(self, x, pos_embed):
        """
        Interpolates positional embeddings if input resolution differs.
        """
        B, H, W, C = x.shape
        pos_H, pos_W = pos_embed.shape[1], pos_embed.shape[2]

        if H == pos_H and W == pos_W:
            return pos_embed

        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)

        return pos_embed

    def forward(self, x):
        """
        Forward pass with:
        - Additive prompts before each block
        - Adapters inside blocks (if enabled)
        """
        # Patch embedding
        x = self.patch_embed(x)

        # Convert to (B, H, W, C) if needed
        if x.shape[1] == self.embed_dim:
            x = x.permute(0, 2, 3, 1)

        # Interpolate and add positional embeddings
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed

        # Transformer blocks with prompts and adapters
        for i, block in enumerate(self.blocks):
            # Add prompt bias before block
            x = x + self.prompt_embeddings[i]
            
            # Apply original block (adapters are inside via patching)
            x = block(x)

        # Back to (B, C, H, W) for decoder
        x = x.permute(0, 3, 1, 2)

        # Neck
        x = self.neck(x)

        return x
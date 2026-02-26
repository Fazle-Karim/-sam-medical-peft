"""
Deep Prompt Tuning for SAM Encoder
Stable Production Version
Prompts are concatenated with spatial tokens and participate in self-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Prompted Transformer Block
# ============================================================

class PromptedAttentionBlock(nn.Module):
    """
    Transformer block where prompts are concatenated with spatial tokens
    and participate in self-attention.
    
    This enables real deep prompt tuning: prompts influence feature representations
    across all layers.
    """
    
    def __init__(self, original_block: nn.Module, num_prompts: int):
        super().__init__()
        self.original_block = original_block
        self.num_prompts = num_prompts
        
        logger.debug(f"Created PromptedAttentionBlock with {num_prompts} prompts")

    def forward(self, x: torch.Tensor, prompts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Spatial features (B, H, W, C)
            prompts: Prompt tokens (B, num_prompts, C)
        
        Returns:
            x_out: Updated spatial features (B, H, W, C)
            prompts_out: Updated prompts (B, num_prompts, C)
        """
        B, H, W, C = x.shape
        P = self.num_prompts
        
        # Flatten spatial tokens to sequence
        x_flat = x.reshape(B, H * W, C)  # (B, H*W, C)
        
        # Concatenate prompts with spatial tokens
        combined = torch.cat([prompts, x_flat], dim=1)  # (B, P + H*W, C)
        
        # Apply original transformer block to combined sequence
        # This allows prompts to attend to spatial tokens and vice versa
        combined_out = self.original_block(combined)
        
        # Split back into prompts and spatial tokens
        prompts_out = combined_out[:, :P, :]          # (B, P, C)
        x_out_flat = combined_out[:, P:, :]           # (B, H*W, C)
        
        # Reshape spatial tokens back to grid
        x_out = x_out_flat.reshape(B, H, W, C)        # (B, H, W, C)
        
        return x_out, prompts_out


# ============================================================
# Prompted SAM Encoder
# ============================================================

class PromptedSAMEncoder(nn.Module):
    """
    SAM encoder modified to propagate prompts through all layers.
    Each transformer block is wrapped to include prompts in attention.
    """
    
    def __init__(self, sam_encoder: nn.Module, num_prompts: int = 16):
        super().__init__()

        self.num_prompts = num_prompts
        self.original_encoder = sam_encoder

        # Replace each block with prompted version
        self.blocks = nn.ModuleList([
            PromptedAttentionBlock(block, num_prompts)
            for block in sam_encoder.blocks
        ])

        # Keep other encoder components
        self.patch_embed = sam_encoder.patch_embed
        self.pos_embed = getattr(sam_encoder, "pos_embed", None)
        self.neck = getattr(sam_encoder, "neck", None)

        # Get embedding dimension safely
        if hasattr(sam_encoder, "embed_dim"):
            self.embed_dim = sam_encoder.embed_dim
        else:
            # Infer from first block
            self.embed_dim = sam_encoder.blocks[0].norm1.normalized_shape[0]
        
        logger.info(f"Created PromptedSAMEncoder with {len(self.blocks)} layers")
        logger.info(f"  num_prompts={num_prompts}, embed_dim={self.embed_dim}")

    def forward(self, x: torch.Tensor, prompts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with prompts through all layers
        
        Args:
            x: Input images (B, 3, H, W) or patch embeddings
            prompts: Initial prompts (B, num_prompts, C)
        
        Returns:
            x: Final spatial features (B, C, H, W) for decoder
            prompts: Final prompts (B, num_prompts, C)
        """
        # ----------------------------------------------------
        # Patch Embedding
        # ----------------------------------------------------
        x = self.patch_embed(x)

        # If Conv2D style output: (B, C, H, W) -> (B, H, W, C)
        if x.shape[1] == self.embed_dim:
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        B, H, W, C = x.shape

        # ----------------------------------------------------
        # Positional Embedding
        # ----------------------------------------------------
        if self.pos_embed is not None:
            pos_embed = self.pos_embed

            # Case 1: Flattened positional embedding (1, N, C)
            if len(pos_embed.shape) == 3:
                x_flat = x.reshape(B, H * W, C)
                
                # Handle size mismatch by truncating or interpolating
                if pos_embed.shape[1] != H * W:
                    if pos_embed.shape[1] > H * W:
                        pos_embed = pos_embed[:, :H * W, :]
                    else:
                        # Simple repeat to match size
                        repeat_factor = (H * W) // pos_embed.shape[1]
                        pos_embed = pos_embed.repeat(1, repeat_factor, 1)
                        if pos_embed.shape[1] > H * W:
                            pos_embed = pos_embed[:, :H * W, :]
                
                x_flat = x_flat + pos_embed
                x = x_flat.reshape(B, H, W, C)

            # Case 2: Grid positional embedding (1, H, W, C)
            elif len(pos_embed.shape) == 4:
                # Handle size mismatch by interpolating
                if pos_embed.shape[1] != H or pos_embed.shape[2] != W:
                    pos_2d = pos_embed.permute(0, 3, 1, 2)  # (1, C, H, W)
                    pos_2d = F.interpolate(
                        pos_2d,
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False
                    )
                    pos_embed = pos_2d.permute(0, 2, 3, 1)  # (1, H, W, C)

                x = x + pos_embed

        # ----------------------------------------------------
        # Transformer Blocks with Prompts
        # ----------------------------------------------------
        for i, block in enumerate(self.blocks):
            x, prompts = block(x, prompts)
            
            # Optional: log shapes for debugging
            if i == 0 or i == len(self.blocks) - 1:
                logger.debug(f"After block {i}: x {x.shape}, prompts {prompts.shape}")

        # ----------------------------------------------------
        # Convert back to (B, C, H, W) for decoder
        # ----------------------------------------------------
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Apply neck if it exists
        if self.neck is not None:
            x = self.neck(x)

        return x, prompts


# ============================================================
# Test and Verification
# ============================================================

def test_prompted_blocks():
    """Minimal test to verify deep prompt implementation works"""
    
    print("\n" + "=" * 60)
    print("TESTING: Final Stable Deep Prompt Implementation")
    print("=" * 60)

    # Test parameters
    B, H, W, C = 2, 16, 16, 768
    P = 16  # num_prompts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Batch size: {B}, Grid: {H}x{W}, Channels: {C}, Prompts: {P}")

    # ----------------------------------------------------
    # Create dummy transformer block for testing
    # ----------------------------------------------------
    class DummyBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )

        def forward(self, x):
            # Pre-norm transformer block
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            x = x + self.mlp(self.norm2(x))
            return x

    # ----------------------------------------------------
    # Create dummy encoder
    # ----------------------------------------------------
    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([DummyBlock(C) for _ in range(4)])
            self.patch_embed = nn.Conv2d(3, C, kernel_size=16, stride=16)
            self.pos_embed = nn.Parameter(torch.randn(1, 16 * 16, C))

    # ----------------------------------------------------
    # Build prompted encoder
    # ----------------------------------------------------
    dummy_encoder = DummyEncoder().to(device)
    model = PromptedSAMEncoder(dummy_encoder, num_prompts=P).to(device)

    # ----------------------------------------------------
    # Test forward pass
    # ----------------------------------------------------
    images = torch.randn(B, 3, 256, 256, device=device)
    prompts = torch.randn(B, P, C, device=device)

    print("\n📥 Input shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Prompts: {prompts.shape}")

    # Forward pass
    x_out, prompts_out = model(images, prompts)

    print("\n📤 Output shapes:")
    print(f"  Spatial features: {x_out.shape}")
    print(f"  Prompts: {prompts_out.shape}")

    # ----------------------------------------------------
    # Verify shapes
    # ----------------------------------------------------
    expected_spatial = (B, C, 16, 16)
    expected_prompts = (B, P, C)

    spatial_correct = x_out.shape == expected_spatial
    prompts_correct = prompts_out.shape == expected_prompts

    print("\n🔍 Shape verification:")
    print(f"  Spatial: {'✅' if spatial_correct else '❌'} {x_out.shape} vs expected {expected_spatial}")
    print(f"  Prompts: {'✅' if prompts_correct else '❌'} {prompts_out.shape} vs expected {expected_prompts}")

    # ----------------------------------------------------
    # Verify prompt influence (optional)
    # ----------------------------------------------------
    with torch.no_grad():
        # Run with different prompts and compare outputs
        prompts2 = torch.randn(B, P, C, device=device)
        x_out2, _ = model(images, prompts2)
        diff = (x_out - x_out2).abs().mean().item()
        print(f"\n📊 Prompt influence: mean diff = {diff:.6f}")

    assert spatial_correct and prompts_correct, "Shape test failed!"
    
    print("\n" + "=" * 60)
    print("✅ FINAL STABLE DEEP PROMPT ENCODER WORKING!")
    print("=" * 60 + "\n")
    
    return model


# ============================================================
# Main Guard
# ============================================================

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    model = test_prompted_blocks()
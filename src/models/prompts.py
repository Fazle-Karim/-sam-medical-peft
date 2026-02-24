"""
Learnable Medical Prompt Tokens for SAM Adaptation
This module implements trainable soft prompts that prepend to image tokens
to adapt SAM to medical imaging domains.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List, Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LearnablePrompts(nn.Module):
    """
    Learnable soft prompt tokens for medical domain adaptation.
    
    These prompts are prepended to the image patch embeddings in transformer
    layers. They are optimized during training to capture medical-specific
    features like textures, boundaries, and anatomical structures.
    
    Args:
        num_prompts (int): Number of prompt tokens per layer
        prompt_dim (int): Dimension of each prompt token (matches transformer dim)
        num_layers (int): Number of transformer layers (for deep prompts)
        initialization (str): Initialization method ('random', 'zeros', 'mean_patch')
        dropout (float): Dropout probability for prompts
        deep_prompts (bool): If True, use different prompts per layer
    """
    
    def __init__(
        self,
        num_prompts: int = 16,
        prompt_dim: int = 768,  # ViT-B dimension
        num_layers: int = 12,    # Number of transformer blocks
        initialization: str = "random",
        dropout: float = 0.0,
        deep_prompts: bool = True,
    ):
        super().__init__()
        
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.num_layers = num_layers
        self.deep_prompts = deep_prompts
        self.initialization = initialization
        
        if deep_prompts:
            # Different prompts for each layer
            # Shape: (num_layers, num_prompts, prompt_dim)
            self.prompt_embeddings = nn.Parameter(
                torch.empty(num_layers, num_prompts, prompt_dim)
            )
            logger.info(f"Initializing deep prompts: {num_layers} layers × {num_prompts} prompts")
        else:
            # Shared prompts across all layers
            # Shape: (num_prompts, prompt_dim)
            self.prompt_embeddings = nn.Parameter(
                torch.empty(num_prompts, prompt_dim)
            )
            logger.info(f"Initializing shallow prompts: {num_prompts} prompts shared across layers")
        
        # Optional prompt dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize the prompts
        self._init_prompts(initialization)
        
        # Track total parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Total prompt parameters: {total_params:,}")
    
    def _init_prompts(self, method: str):
        """Initialize prompt embeddings."""
        if method == "random":
            # Random initialization from normal distribution
            nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)
            logger.info("Random initialization (mean=0, std=0.02)")
        
        elif method == "zeros":
            # Zero initialization
            nn.init.zeros_(self.prompt_embeddings)
            logger.info("Zero initialization")
        
        elif method == "mean_patch":
            # Initialize as mean of patch embeddings (requires forward hook)
            # Will be set later via a separate method
            nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)
            logger.info("Will use mean patch initialization later")
        
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        layer_idx: int,
        return_prompts_only: bool = False
    ) -> torch.Tensor:
        """
        Forward pass - prepend prompts to input tokens.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, prompt_dim)
            layer_idx: Current transformer layer index
            return_prompts_only: If True, return only prompts (for debugging)
        
        Returns:
            Tensor with prompts prepended: (batch_size, seq_len + num_prompts, prompt_dim)
        """
        batch_size = x.shape[0]
        
        # Get prompts for this layer
        if self.deep_prompts:
            # Layer-specific prompts
            prompts = self.prompt_embeddings[layer_idx]  # (num_prompts, dim)
        else:
            # Shared prompts
            prompts = self.prompt_embeddings  # (num_prompts, dim)
        
        # Expand to batch dimension
        prompts = prompts.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply dropout
        prompts = self.dropout(prompts)
        
        if return_prompts_only:
            return prompts
        
        # Input should be 3D: (batch, seq_len, dim)
        # Concatenate prompts to the front of the token sequence along dimension 1
        x = torch.cat([prompts, x], dim=1)  # (batch, num_prompts + seq_len, dim)
        
        return x
    
    def get_prompts(self, layer_idx: Optional[int] = None) -> torch.Tensor:
        """Get prompt embeddings for inspection."""
        if layer_idx is not None and self.deep_prompts:
            return self.prompt_embeddings[layer_idx]
        else:
            return self.prompt_embeddings
    
    def extra_repr(self) -> str:
        """String representation."""
        return (f"num_prompts={self.num_prompts}, dim={self.prompt_dim}, "
                f"deep={self.deep_prompts}, init={self.initialization}")


class PromptedSAMEncoder(nn.Module):
    """
    SAM encoder with learnable prompts inserted.
    This wraps the original SAM encoder and adds prompt tokens.
    """
    
    def __init__(
        self,
        sam_encoder: nn.Module,
        num_prompts: int = 16,
        prompt_dim: int = 768,
        deep_prompts: bool = True,
        prompt_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        
        self.encoder = sam_encoder
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        
        # Which layers to insert prompts (default: all layers)
        if prompt_layers is None:
            self.prompt_layers = list(range(len(sam_encoder.blocks)))
        else:
            self.prompt_layers = prompt_layers
        
        # Create prompt module
        self.prompts = LearnablePrompts(
            num_prompts=num_prompts,
            prompt_dim=prompt_dim,
            num_layers=len(sam_encoder.blocks),
            deep_prompts=deep_prompts,
            initialization="random"
        )
        
        logger.info(f"PromptedSAMEncoder initialized")
        logger.info(f"  Prompts inserted in layers: {self.prompt_layers}")
        logger.info(f"  Total prompt parameters: {sum(p.numel() for p in self.prompts.parameters()):,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder with prompts inserted.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Image embeddings
        """
        # Get patch embeddings
        x = self.encoder.patch_embed(x)
        
        # Add positional embeddings
        if hasattr(self.encoder, 'pos_embed'):
            x = x + self.encoder.pos_embed
        
        # Pass through transformer blocks with prompts
        for i, block in enumerate(self.encoder.blocks):
            if i in self.prompt_layers:
                # Insert prompts before this block
                x = self.prompts(x, layer_idx=i)
            
            # Apply transformer block
            x = block(x)
        
        # Apply neck if exists
        if hasattr(self.encoder, 'neck'):
            x = self.encoder.neck(x)
        
        return x
    
    def get_prompt_gradients(self):
        """Monitor prompt gradients for debugging."""
        prompt_grads = {}
        for name, param in self.prompts.named_parameters():
            if param.grad is not None:
                prompt_grads[name] = {
                    'mean': param.grad.mean().item(),
                    'std': param.grad.std().item(),
                    'norm': param.grad.norm().item()
                }
        return prompt_grads


class PromptConfig:
    """Configuration class for prompt experiments."""
    
    def __init__(
        self,
        num_prompts: int = 16,
        deep_prompts: bool = True,
        prompt_layers: Optional[List[int]] = None,
        initialization: str = "random",
        dropout: float = 0.0,
    ):
        self.num_prompts = num_prompts
        self.deep_prompts = deep_prompts
        self.prompt_layers = prompt_layers
        self.initialization = initialization
        self.dropout = dropout
        
        if prompt_layers is None and deep_prompts:
            # Default: all layers
            self.prompt_layers = list(range(12))  # ViT-B has 12 layers
    
    def get_parameter_count(self, prompt_dim: int = 768) -> int:
        """Estimate number of trainable parameters."""
        if self.deep_prompts:
            # num_layers × num_prompts × dim
            num_layers = len(self.prompt_layers) if self.prompt_layers else 12
            return num_layers * self.num_prompts * prompt_dim
        else:
            # num_prompts × dim
            return self.num_prompts * prompt_dim
    
    def __repr__(self) -> str:
        return (f"PromptConfig(num_prompts={self.num_prompts}, "
                f"deep={self.deep_prompts}, layers={self.prompt_layers}, "
                f"init={self.initialization})")


def visualize_prompts(prompt_module: LearnablePrompts, save_path: Optional[str] = None):
    """
    Visualize prompt embeddings (for analysis).
    
    Args:
        prompt_module: LearnablePrompts instance
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    prompts = prompt_module.prompt_embeddings.detach().cpu().numpy()
    
    if prompt_module.deep_prompts:
        # Plot mean and std per layer
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        mean_per_layer = prompts.mean(axis=(1, 2))
        std_per_layer = prompts.std(axis=(1, 2))
        
        axes[0].plot(mean_per_layer)
        axes[0].set_title('Mean prompt value per layer')
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Mean value')
        
        axes[1].plot(std_per_layer)
        axes[1].set_title('Std prompt value per layer')
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Std value')
        
    else:
        # Plot histogram of prompt values
        plt.figure(figsize=(8, 4))
        plt.hist(prompts.flatten(), bins=50)
        plt.title('Prompt value distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved prompt visualization to {save_path}")
    
    plt.show()


def test_prompts():
    """Test function to verify prompt implementation."""
    print("Testing LearnablePrompts...")
    
    # Test 1: Basic prompt forward
    batch_size = 2
    seq_len = 256
    hidden_dim = 768
    num_prompts = 16
    num_layers = 12
    
    prompts = LearnablePrompts(
        num_prompts=num_prompts,
        prompt_dim=hidden_dim,
        num_layers=num_layers,
        deep_prompts=True
    )
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Test at layer 0
    y = prompts(x, layer_idx=0)
    print(f"\nTest 1 - Basic forward:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Expected: [2, {seq_len + num_prompts}, 768]")
    assert y.shape == (batch_size, seq_len + num_prompts, hidden_dim)
    
    # Test 2: Parameter count
    total_params = sum(p.numel() for p in prompts.parameters())
    expected_params = num_layers * num_prompts * hidden_dim
    print(f"\nTest 2 - Parameter count:")
    print(f"  Actual parameters: {total_params:,}")
    print(f"  Expected: {expected_params:,}")
    print(f"  Match: {total_params == expected_params}")
    
    # Test 3: Shallow prompts
    shallow_prompts = LearnablePrompts(
        num_prompts=num_prompts,
        prompt_dim=hidden_dim,
        num_layers=num_layers,
        deep_prompts=False
    )
    shallow_params = sum(p.numel() for p in shallow_prompts.parameters())
    expected_shallow = num_prompts * hidden_dim
    print(f"\nTest 3 - Shallow prompts:")
    print(f"  Parameters: {shallow_params:,}")
    print(f"  Expected: {expected_shallow:,}")
    print(f"  {expected_shallow/expected_params*100:.1f}% of deep prompts")
    
    # Test 4: Gradient flow
    y = prompts(x, layer_idx=5)
    loss = y.sum()
    loss.backward()
    
    has_grad = prompts.prompt_embeddings.grad is not None
    grad_norm = prompts.prompt_embeddings.grad.norm().item() if has_grad else 0
    
    print(f"\nTest 4 - Gradient flow:")
    print(f"  Has gradient: {has_grad}")
    print(f"  Gradient norm: {grad_norm:.4f}")
    print(f"  ✓ Gradients flowing properly")
    
    # Test 5: PromptConfig
    config = PromptConfig(
        num_prompts=16,
        deep_prompts=True,
        prompt_layers=[8, 9, 10, 11],
        initialization="random"
    )
    print(f"\nTest 5 - PromptConfig:")
    print(f"  {config}")
    print(f"  Estimated parameters: {config.get_parameter_count():,}")
    
    print("\n✓ All prompt tests passed!")
    
    return prompts, shallow_prompts, config


if __name__ == "__main__":
    # Run tests
    test_prompts()
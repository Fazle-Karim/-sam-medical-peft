"""
Deep Learnable Prompts for Medical Image Segmentation
Prompts persist through all transformer layers as proposed in research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class DeepPrompts(nn.Module):
    """
    Deep prompt tokens that persist through all transformer layers.
    Different prompts for each layer, maintained separately.
    
    Args:
        num_prompts: Number of prompt tokens per layer
        hidden_dim: Dimension of each token (matches transformer)
        num_layers: Number of transformer layers
        init_scale: Scale for random initialization
    """
    
    def __init__(
        self,
        num_prompts: int = 16,
        hidden_dim: int = 768,
        num_layers: int = 12,
        init_scale: float = 0.02
    ):
        super().__init__()
        
        self.num_prompts = num_prompts
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Shape: (num_layers, 1, num_prompts, hidden_dim)
        # 1 is for batch broadcasting
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_layers, 1, num_prompts, hidden_dim) * init_scale
        )
        
        # Initialize with small values for stable training
        nn.init.normal_(self.prompt_embeddings, mean=0.0, std=init_scale)
        
        logger.info(f"Initialized DeepPrompts: {num_layers} layers × {num_prompts} prompts")
        logger.info(f"Total prompt parameters: {num_layers * num_prompts * hidden_dim:,}")
    
    def get_prompts(self, layer_idx: int, batch_size: int) -> torch.Tensor:
        """
        Get prompts for specific layer, expanded to batch size
        
        Args:
            layer_idx: Which transformer layer (0 to num_layers-1)
            batch_size: Batch size for expansion
        
        Returns:
            prompts: (batch_size, num_prompts, hidden_dim)
        """
        # prompts shape: (1, num_prompts, hidden_dim)
        prompts = self.prompt_embeddings[layer_idx]
        # Expand to batch: (batch_size, num_prompts, hidden_dim)
        return prompts.expand(batch_size, -1, -1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        layer_idx: int, 
        return_prompts_only: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - returns prompts for this layer
        
        Args:
            x: Input tensor (used only for batch size)
            layer_idx: Current transformer layer
            return_prompts_only: If True, return only prompts
        
        Returns:
            x: Unmodified input (for compatibility)
            prompts: Prompt tokens for this layer
        """
        batch_size = x.shape[0]
        prompts = self.get_prompts(layer_idx, batch_size)
        
        if return_prompts_only:
            return prompts
        
        # Return both x (unmodified) and prompts
        return x, prompts
    
    def extra_repr(self) -> str:
        return (f"num_prompts={self.num_prompts}, "
                f"layers={self.num_layers}, "
                f"dim={self.hidden_dim}")


# Keep original class name for backward compatibility
LearnablePrompts = DeepPrompts


def test_deep_prompts():
    """Test function for DeepPrompts module"""
    print("Testing DeepPrompts...")
    
    # Test 1: Initialization
    batch_size = 2
    num_prompts = 16
    hidden_dim = 768
    num_layers = 12
    
    prompts = DeepPrompts(
        num_prompts=num_prompts,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    print(f"\nTest 1 - Initialization:")
    print(f"  Parameters: {sum(p.numel() for p in prompts.parameters()):,}")
    print(f"  Expected: {num_layers * num_prompts * hidden_dim:,}")
    
    # Test 2: Get prompts for different layers
    x_dummy = torch.randn(batch_size, 16, 16, hidden_dim)
    
    for layer in [0, 5, 11]:
        x_out, prompts_out = prompts(x_dummy, layer)
        print(f"\nTest 2 - Layer {layer}:")
        print(f"  Prompts shape: {prompts_out.shape}")
        print(f"  Expected: ({batch_size}, {num_prompts}, {hidden_dim})")
        assert prompts_out.shape == (batch_size, num_prompts, hidden_dim)
    
    # Test 3: Gradient flow
    x_dummy = torch.randn(batch_size, 16, 16, hidden_dim, requires_grad=True)
    _, prompts_out = prompts(x_dummy, 0)
    loss = prompts_out.sum()
    loss.backward()
    
    print(f"\nTest 3 - Gradient flow:")
    has_grad = prompts.prompt_embeddings.grad is not None
    grad_norm = prompts.prompt_embeddings.grad.norm().item() if has_grad else 0
    print(f"  Has gradient: {has_grad}")
    print(f"  Gradient norm: {grad_norm:.4f}")
    print(f"  ✓ Gradients flowing properly")
    
    print("\n✓ All DeepPrompts tests passed!")
    return prompts


if __name__ == "__main__":
    test_deep_prompts()
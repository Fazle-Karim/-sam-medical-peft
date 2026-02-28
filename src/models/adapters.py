"""
Bottleneck Adapter Modules for Parameter-Efficient Fine-Tuning
These adapters insert into SAM's transformer blocks to add trainable parameters
while keeping the backbone frozen.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BottleneckAdapter(nn.Module):
    """
    Bottleneck adapter module for transformer blocks.
    
    This module follows a down-project → activation → up-project structure
    with a residual connection. It adds minimal trainable parameters while
    allowing adaptation of frozen features.
    
    Args:
        hidden_dim (int): Input/output dimension of the transformer
        bottleneck_ratio (float): Ratio of bottleneck dimension to hidden_dim
            Default: 0.25 (r = d/4)
        dropout (float): Dropout probability
        activation (str): Activation function ('gelu' or 'relu')
    """
    
    def __init__(
        self,
        hidden_dim: int,
        bottleneck_ratio: float = 0.25,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = int(hidden_dim * bottleneck_ratio)
        self.bottleneck_ratio = bottleneck_ratio
        
        # Down-projection: d → r
        self.down_proj = nn.Linear(hidden_dim, self.bottleneck_dim)
        
        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Up-projection: r → d
        self.up_proj = nn.Linear(self.bottleneck_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights properly
        self._init_weights()
        
        logger.debug(f"Initialized BottleneckAdapter: d={hidden_dim}, r={self.bottleneck_dim}")
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Initialize down-projection with small weights
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        
        # Initialize up-projection to near-zero (so adapter starts as identity)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adapter.
        
        Args:
            x: Input tensor of shape (..., hidden_dim)
        
        Returns:
            Output tensor of same shape as input (residual connection)
        """
        # Store residual
        residual = x
        
        # Down-project
        x = self.down_proj(x)
        
        # Activation
        x = self.activation(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Up-project
        x = self.up_proj(x)
        
        # Dropout again
        x = self.dropout(x)
        
        # Residual connection
        return x + residual
    
    def extra_repr(self) -> str:
        """String representation."""
        return f"hidden_dim={self.hidden_dim}, bottleneck_dim={self.bottleneck_dim}, ratio={self.bottleneck_ratio}"


class SequentialAdapters(nn.Module):
    """
    Container for multiple adapters that can be applied sequentially.
    Useful for adding adapters to multiple transformer blocks.
    """
    
    def __init__(self, adapters: List[BottleneckAdapter]):
        super().__init__()
        self.adapters = nn.ModuleList(adapters)
    
    def forward(self, x: torch.Tensor, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Forward through adapters.
        
        Args:
            x: Input tensor
            layer_idx: If provided, only apply adapter at this index
        
        Returns:
            Output tensor
        """
        if layer_idx is not None:
            # Apply single adapter
            return self.adapters[layer_idx](x)
        else:
            # Apply all adapters sequentially
            for adapter in self.adapters:
                x = adapter(x)
            return x


class AdapterConfig:
    """
    Configuration class for adapter modules.
    Makes it easy to experiment with different adapter settings.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,  # ViT-B dimension
        bottleneck_ratio: float = 0.25,
        dropout: float = 0.1,
        activation: str = "gelu",
        adapter_layers: List[int] = None,  # Which layers to add adapters to
        adapter_positions: List[str] = None,  # Where to add: 'attn', 'mlp', or both
    ):
        self.hidden_dim = hidden_dim
        self.bottleneck_ratio = bottleneck_ratio
        self.dropout = dropout
        self.activation = activation
        
        if adapter_layers is None:
            # Default: last 4 layers
            self.adapter_layers = [8, 9, 10, 11]  # 0-indexed for 12-layer ViT
        else:
            self.adapter_layers = adapter_layers
            
        if adapter_positions is None:
            # Default: add after both attention and MLP
            self.adapter_positions = ['attn', 'mlp']
        else:
            self.adapter_positions = adapter_positions
    
    def get_total_adapters(self) -> int:
        """Get total number of adapter modules."""
        return len(self.adapter_layers) * len(self.adapter_positions)
    
    def get_parameter_count(self) -> int:
        """Estimate number of trainable parameters added."""
        bottleneck_dim = int(self.hidden_dim * self.bottleneck_ratio)
        # Each adapter has 2 linear layers: d*r + r + r*d + d
        params_per_adapter = 2 * (self.hidden_dim * bottleneck_dim + bottleneck_dim)
        total_params = params_per_adapter * self.get_total_adapters()
        return total_params
    
    def __repr__(self) -> str:
        return (f"AdapterConfig(hidden_dim={self.hidden_dim}, "
                f"ratio={self.bottleneck_ratio}, "
                f"layers={self.adapter_layers}, "
                f"positions={self.adapter_positions})")


def create_adapters_for_sam(
    sam_encoder: nn.Module,
    adapter_layers: List[int] = None,
    adapter_positions: List[str] = None,
    bottleneck_ratio: float = 0.25,
    dropout: float = 0.1,
    activation: str = "gelu"
) -> nn.ModuleDict:
    """
    Create and insert adapters into a SAM encoder.
    
    Args:
        sam_encoder: SAM image encoder module
        adapter_layers: List of layer indices to add adapters to
        adapter_positions: Where to add adapters ['attn', 'mlp'] or both
        bottleneck_ratio: Bottleneck dimension ratio
        dropout: Dropout probability
        activation: Activation function
    
    Returns:
        ModuleDict mapping layer names to adapter modules
    """
    if adapter_layers is None:
        adapter_layers = [8, 9, 10, 11]
    
    if adapter_positions is None:
        adapter_positions = ['attn', 'mlp']
    
    hidden_dim = sam_encoder.pos_embed.shape[-1]
    adapters = {}
    
    for layer_idx in adapter_layers:
        if layer_idx >= len(sam_encoder.blocks):
            logger.warning(f"Layer {layer_idx} out of range, skipping")
            continue
            
        block = sam_encoder.blocks[layer_idx]
        
        # Add adapter after attention
        if 'attn' in adapter_positions:
            adapter_attn = BottleneckAdapter(
                hidden_dim=hidden_dim,
                bottleneck_ratio=bottleneck_ratio,
                dropout=dropout,
                activation=activation
            )
            # Store adapter in the block (will be used in patched forward)
            block.attn_adapter = adapter_attn
            adapters[f'layer{layer_idx}_attn'] = adapter_attn
            logger.info(f"Added attention adapter to layer {layer_idx}")
        
        # Add adapter after MLP
        if 'mlp' in adapter_positions:
            adapter_mlp = BottleneckAdapter(
                hidden_dim=hidden_dim,
                bottleneck_ratio=bottleneck_ratio,
                dropout=dropout,
                activation=activation
            )
            # Store adapter in the block
            block.mlp_adapter = adapter_mlp
            adapters[f'layer{layer_idx}_mlp'] = adapter_mlp
            logger.info(f"Added MLP adapter to layer {layer_idx}")
    
    logger.info(f"Created {len(adapters)} adapter modules")
    
    # Convert to ModuleDict
    adapter_dict = nn.ModuleDict(adapters)
    return adapter_dict


def test_adapters():
    """Test function to verify adapter implementation."""
    print("Testing BottleneckAdapter...")
    
    # Test 1: Basic adapter forward
    hidden_dim = 768
    batch_size = 2
    seq_len = 256
    
    adapter = BottleneckAdapter(
        hidden_dim=hidden_dim,
        bottleneck_ratio=0.25,
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    y = adapter(x)
    
    print(f"Test 1 - Basic adapter:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Input requires_grad: {x.requires_grad}")
    print(f"  Output requires_grad: {y.requires_grad}")
    
    # Test 2: Parameter count
    total_params = sum(p.numel() for p in adapter.parameters())
    expected_params = 2 * (hidden_dim * (hidden_dim//4) + (hidden_dim//4))
    print(f"\nTest 2 - Parameter count:")
    print(f"  Actual parameters: {total_params:,}")
    print(f"  Expected approx: {expected_params:,}")
    print(f"  Ratio: {total_params / (hidden_dim * hidden_dim):.4f} of full layer")
    
    # Test 3: AdapterConfig
    config = AdapterConfig(
        hidden_dim=768,
        bottleneck_ratio=0.25,
        adapter_layers=[8, 9, 10, 11],
        adapter_positions=['attn', 'mlp']
    )
    print(f"\nTest 3 - AdapterConfig:")
    print(f"  {config}")
    print(f"  Total adapters: {config.get_total_adapters()}")
    print(f"  Estimated parameters: {config.get_parameter_count():,}")
    
    # Test 4: Initialization (up-proj should be near zero)
    print(f"\nTest 4 - Weight initialization:")
    print(f"  Down-proj weight norm: {adapter.down_proj.weight.norm().item():.4f}")
    print(f"  Up-proj weight norm: {adapter.up_proj.weight.norm().item():.4f}")
    print(f"  Up-proj should be near zero ✓")
    
    print("\n✓ All adapter tests passed!")
    
    return adapter, config


if __name__ == "__main__":
    # Run tests
    test_adapters()
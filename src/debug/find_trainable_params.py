"""
Debug script to find trainable parameters for any config
"""

import torch
import sys
import argparse
from pathlib import Path
import yaml
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.train import PEFT_SAM, set_seed

def find_trainable_params(config_path=None):
    """Find all trainable parameters for a given config"""
    
    print("\n" + "="*70)
    print("FINDING TRAINABLE PARAMETERS")
    print("="*70)
    
    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"Using config: {config_path}")
    else:
        # Default zero-shot test config
        config = {
            'sam_checkpoint': 'checkpoints/sam_vit_b_01ec64.pth',
            'model_type': 'vit_b',
            'freeze_encoder': True,
            'freeze_decoder': True,
            'freeze_prompt_encoder': True,
            'use_prompts': False,
            'use_adapters': False,
            'batch_size': 2,
            'num_epochs': 0,
            'target_size': [256, 256],
            'seed': 42
        }
        print("Using default zero-shot test config")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config['seed'])
    
    # Create model
    model = PEFT_SAM(config, device)
    
    # Find all trainable parameters
    trainable_params = []
    total_params = 0
    trainable_count = 0
    
    print("\n📋 Trainable Parameters:")
    print("-" * 70)
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
            trainable_count += param.numel()
            print(f"  🔴 {name}: {param.numel():,} params")
    
    if not trainable_params:
        print("  (none)")
    
    print("-" * 70)
    print(f"\n📊 Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_count:,}")
    print(f"  Number of trainable parameter groups: {len(trainable_params)}")
    
    if total_params > 0:
        percentage = 100 * trainable_count / total_params
        print(f"  Trainable percentage: {percentage:.4f}%")
    
    print("\n" + "="*70)
    
    return trainable_count, percentage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    
    find_trainable_params(args.config)
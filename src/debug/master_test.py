"""
Master Test Script for PEFT-SAM
Tests all components together with ablation controls
"""

import torch
from pathlib import Path
import sys
import os
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.adapters import test_adapters
from src.models.prompts import test_deep_prompts
from src.utils.data_loader import create_sample_data
from src.training.train import PEFT_SAM, Trainer, set_seed
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ablation_configs():
    """Test that all ablation configs work correctly."""
    print("\n" + "="*70)
    print("TESTING ABLATION CONFIGURATIONS")
    print("="*70)
    
    configs = [
        ('zero_shot', {'use_prompts': False, 'use_adapters': False, 'freeze_decoder': True, 'freeze_prompt_encoder': True, 'num_epochs': 0}),
        ('full_finetune', {'freeze_encoder': False, 'use_prompts': False, 'use_adapters': False}),
        ('adapter_only', {'use_prompts': False, 'use_adapters': True}),
        ('prompt_only', {'use_prompts': True, 'use_adapters': False}),
        ('proposed', {'use_prompts': True, 'use_adapters': True}),
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for name, overrides in configs:
        print(f"\n📦 Testing {name} configuration...")
        
        config = {
            'sam_checkpoint': 'checkpoints/sam_vit_b_01ec64.pth',
            'model_type': 'vit_b',
            'freeze_encoder': True,
            'freeze_decoder': False,
            'freeze_prompt_encoder': False,
            'use_prompts': False,
            'use_adapters': False,
            'adapter_layers': [8, 9, 10, 11],
            'batch_size': 2,
            'num_epochs': 1,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'target_size': [256, 256],
            'seed': 42,
            **overrides
        }
        
        set_seed(config['seed'])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create BOTH datasets to satisfy data loader
            create_sample_data(Path(tmpdir) / "ISIC", num_samples=10)
            create_sample_data(Path(tmpdir) / "BUSI", num_samples=10)
            
            config['data_root'] = tmpdir
            config['dataset'] = 'isic'
            
            # Import here to avoid circular imports
            from src.utils.data_loader import create_dataloaders
            
            dataloaders = create_dataloaders(
                root_dir=tmpdir,
                batch_size=2,
                target_size=(256, 256),
                num_workers=0
            )
            
            train_loader = dataloaders['isic']['train']
            
            # Create model
            model = PEFT_SAM(config, device)
            print(f"  ✅ {name} model created")
            
            # Test forward pass
            batch = next(iter(train_loader))
            images = batch['image'].to(device)
            outputs = model(images)
            print(f"  ✅ Forward pass: {images.shape} -> {outputs.shape}")
            
            # Quick parameter check
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"  📊 Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    print("\n" + "="*70)
    print("✅ ALL ABLATION CONFIGS TESTED")
    print("="*70)


def test_data_loader_quick():
    """Quick test for data loader."""
    print("\n📦 Quick Data Loader Test")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample data
        create_sample_data(Path(tmpdir) / "ISIC", num_samples=5)
        create_sample_data(Path(tmpdir) / "BUSI", num_samples=5)
        
        from src.utils.data_loader import create_dataloaders
        
        dataloaders = create_dataloaders(
            root_dir=tmpdir,
            batch_size=2,
            target_size=(256, 256),
            num_workers=0
        )
        
        # Test ISIC loader
        isic_train = dataloaders['isic']['train']
        batch = next(iter(isic_train))
        print(f"ISIC batch - Images: {batch['image'].shape}, Masks: {batch['mask'].shape}")
        
        # Test BUSI loader
        busi_train = dataloaders['busi']['train']
        batch = next(iter(busi_train))
        print(f"BUSI batch - Images: {batch['image'].shape}, Masks: {batch['mask'].shape}")
        
        print("✅ Data loader quick test passed")


def test_all():
    """Run all tests."""
    print("\n" + "="*70)
    print("MASTER TEST: PEFT-SAM COMPLETE PIPELINE")
    print("="*70)

    # Test 1: Adapters
    print("\n📦 Test 1: Adapters Module")
    print("-" * 40)
    test_adapters()
    print("✅ Adapters test passed")

    # Test 2: Prompts
    print("\n📦 Test 2: Prompts Module")
    print("-" * 40)
    test_deep_prompts()
    print("✅ Prompts test passed")

    # Test 3: Data Loader Quick Test
    test_data_loader_quick()

    # Test 4: Ablation Configs
    test_ablation_configs()

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED! PEFT-SAM IS READY FOR EXPERIMENTS!")
    print("="*70)


if __name__ == "__main__":
    test_all()
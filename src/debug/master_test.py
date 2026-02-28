"""
Master Test Script for PEFT-SAM
Tests all components together
"""

import torch
from pathlib import Path
import sys
import os
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.adapters import test_adapters
from src.models.blocks import PromptedSAMEncoder
from src.training.train import PEFT_SAM, Trainer
from src.utils.data_loader import create_sample_data, create_dataloaders
import tempfile

def test_all():
    print("\n" + "="*70)
    print("MASTER TEST: PEFT-SAM COMPLETE PIPELINE")
    print("="*70)

    # ===============================
    # TEST 1: Adapters
    # ===============================
    print("\n📦 Test 1: Adapters Module")
    print("-" * 40)
    try:
        from src.models.adapters import test_adapters
        test_adapters()
        print("✅ Adapters test passed")
    except Exception as e:
        print(f"❌ Adapters test failed: {e}")
        return

    # ===============================
    # TEST 2: Prompts
    # ===============================
    print("\n📦 Test 2: Prompts Module")
    print("-" * 40)
    try:
        from src.models.prompts import test_deep_prompts
        test_deep_prompts()
        print("✅ Prompts test passed")
    except Exception as e:
        print(f"❌ Prompts test failed: {e}")
        return

    # ===============================
    # TEST 3: Data Loader
    # ===============================
    print("\n📦 Test 3: Data Loader")
    print("-" * 40)
    try:
        from src.utils.data_loader import test_data_loader
        test_data_loader()
        print("✅ Data loader test passed")
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return

    # ===============================
    # TEST 4: Full Model Integration
    # ===============================
    print("\n📦 Test 4: Full Model Integration")
    print("-" * 40)
    
    config = {
        'sam_checkpoint': 'checkpoints/sam_vit_b_01ec64.pth',
        'model_type': 'vit_b',
        'freeze_encoder': True,
        'use_adapters': True,
        'adapter_layers': [8, 9, 10, 11],
        'batch_size': 2,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'target_size': [256, 256],
        'num_epochs': 2
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample data
        create_sample_data(Path(tmpdir) / "ISIC", num_samples=10)
        create_sample_data(Path(tmpdir) / "BUSI", num_samples=10)
        
        config['data_root'] = tmpdir
        config['dataset'] = 'isic'
        
        # Create dataloaders
        dataloaders = create_dataloaders(
            root_dir=tmpdir,
            batch_size=2,
            target_size=(256, 256),
            num_workers=0
        )
        
        train_loader = dataloaders['isic']['train']
        val_loader = dataloaders['isic']['val']
        
        # Create model
        model = PEFT_SAM(config, device)
        
        # Test forward pass
        batch = next(iter(train_loader))
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        outputs = model(images)
        print(f"Forward pass: {images.shape} -> {outputs.shape}")
        assert outputs.shape == masks.shape, f"Shape mismatch: {outputs.shape} vs {masks.shape}"
        print("✅ Forward pass OK")
        
        # Test training step
        trainer = Trainer(
            model, train_loader, val_loader, config,
            Path(tmpdir) / "test_exp"
        )
        loss = trainer.train_epoch()
        print(f"Training loss: {loss:.4f}")
        print("✅ Training step OK")
        
        # Test validation
        dice, iou = trainer.validate()
        print(f"Validation Dice: {dice:.2f}%, IoU: {iou:.2f}%")
        print("✅ Validation OK")

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED! PEFT-SAM IS WORKING!")
    print("="*70)

if __name__ == "__main__":
    test_all()
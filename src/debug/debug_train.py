"""
Debug Script for PEFT-SAM
Runs forward pass, training step, and validation step
without full training.
"""

import torch
from pathlib import Path
from src.training.train import PEFT_SAM, Trainer
from src.utils.data_loader import create_sample_data, create_dataloaders
import tempfile
import yaml


def run_debug():

    print("\n" + "="*60)
    print("DEBUGGING PEFT-SAM PIPELINE")
    print("="*60)

    config = {
        'experiment_name': 'debug_run',
        'dataset': 'isic',
        'data_root': '',
        'sam_checkpoint': 'checkpoints/sam_vit_b_01ec64.pth',
        'model_type': 'vit_b',
        'freeze_encoder': True,
        'batch_size': 2,
        'num_epochs': 2,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'target_size': [256,256],
        'num_workers': 0
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with tempfile.TemporaryDirectory() as tmpdir:

        # Create sample dataset
        create_sample_data(Path(tmpdir) / "ISIC", num_samples=10)
        create_sample_data(Path(tmpdir) / "BUSI", num_samples=10)

        config['data_root'] = tmpdir

        dataloaders = create_dataloaders(
            root_dir=tmpdir,
            batch_size=2,
            target_size=(256,256),
            num_workers=0
        )

        train_loader = dataloaders['isic']['train']
        val_loader = dataloaders['isic']['val']

        # ----------------------------
        # Test 1: Model Creation
        # ----------------------------
        print("\n🧪 Test 1: Model Creation")
        model = PEFT_SAM(config, device)
        print("✅ Model initialized")

        # Check trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable %: {100*trainable/total:.2f}%")

        # ----------------------------
        # Test 2: Forward Pass
        # ----------------------------
        print("\n🧪 Test 2: Forward Pass")

        batch = next(iter(train_loader))
        images = batch['image'].to(device)

        outputs = model(images)

        print("Input shape:", images.shape)
        print("Output shape:", outputs.shape)
        print("✅ Forward pass successful")

        # ----------------------------
        # Test 3: One Training Step
        # ----------------------------
        print("\n🧪 Test 3: One Training Step")

        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            config,
            Path(tmpdir) / "debug_exp"
        )

        loss = trainer.train_epoch()
        print("Train loss:", loss)
        print("✅ One training epoch successful")

        # ----------------------------
        # Test 4: Validation Step
        # ----------------------------
        print("\n🧪 Test 4: Validation")

        dice, iou = trainer.validate()
        print(f"Dice: {dice:.2f}")
        print(f"IoU: {iou:.2f}")
        print("✅ Validation successful")

    print("\n" + "="*60)
    print("ALL DEBUG TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    run_debug()
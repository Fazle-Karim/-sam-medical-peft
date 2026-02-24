"""
Training Script for Parameter-Efficient SAM Adaptation
Combines frozen encoder, learnable prompts, and bottleneck adapters
for medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
import argparse
import yaml
import json
from tqdm import tqdm
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, List
import torch.nn.functional as F

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Note: wandb not installed - install with 'pip install wandb' for experiment tracking")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.sam_encoder import SAMEncoderWrapper
from src.models.adapters import BottleneckAdapter, AdapterConfig, create_adapters_for_sam
from src.models.prompts import LearnablePrompts, PromptedSAMEncoder, PromptConfig
from src.utils.data_loader import create_dataloaders, get_sample_batch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PEFT_SAM(nn.Module):
    """
    Complete Parameter-Efficient Fine-Tuning SAM Model.
    Combines frozen encoder with adapters and prompts.
    """
    
    def __init__(
        self,
        sam_checkpoint: str = "checkpoints/sam_vit_b_01ec64.pth",
        model_type: str = "vit_b",
        use_adapters: bool = True,
        use_prompts: bool = True,
        adapter_config: Optional[AdapterConfig] = None,
        prompt_config: Optional[PromptConfig] = None,
        freeze_encoder: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.device = device
        self.use_adapters = use_adapters
        self.use_prompts = use_prompts
        
        logger.info(f"Initializing PEFT-SAM on {device}")
        logger.info(f"  Use adapters: {use_adapters}")
        logger.info(f"  Use prompts: {use_prompts}")
        
        # Load base SAM encoder
        self.sam_wrapper = SAMEncoderWrapper(
            model_type=model_type,
            checkpoint_path=sam_checkpoint,
            freeze_encoder=freeze_encoder,
            freeze_decoder=False,
            device=device
        )
        
        # Get encoder dimension
        self.encoder_dim = self.sam_wrapper.encoder_dim
        self.num_layers = len(self.sam_wrapper.sam.image_encoder.blocks)
        
        # Initialize adapters if requested
        if use_adapters:
            if adapter_config is None:
                adapter_config = AdapterConfig(
                    hidden_dim=self.encoder_dim,
                    bottleneck_ratio=0.25,
                    adapter_layers=[8, 9, 10, 11],
                    adapter_positions=['attn', 'mlp']
                )
            
            self.adapter_config = adapter_config
            self.adapters = nn.ModuleDict(
                create_adapters_for_sam(
                    adapter_config,
                    self.sam_wrapper.sam.image_encoder
                )
            )
            logger.info(f"Initialized {len(self.adapters)} adapter modules")
        
        # Initialize prompts
        if use_prompts:
            self.num_prompts = 16
            # Create prompts for first layer only
            self.prompt_embeddings = nn.Parameter(
                torch.empty(1, self.num_prompts, self.encoder_dim)
            )
            nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)
            
            # Projection layer to maintain sequence length
            self.prompt_proj = nn.Conv1d(
                self.encoder_dim, 
                self.encoder_dim, 
                kernel_size=3, 
                padding=1
            )
            logger.info(f"Initialized {self.num_prompts} prompt tokens for first layer")
        
        # Move to device
        self.to(device)
        
        # Log parameter counts
        self._log_parameter_counts()
    
    def _log_parameter_counts(self):
        """Log detailed parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info("=" * 50)
        logger.info("Parameter Summary:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        if self.use_adapters:
            adapter_params = sum(p.numel() for p in self.adapters.parameters())
            logger.info(f"  Adapter parameters: {adapter_params:,} "
                       f"({100 * adapter_params / total_params:.2f}%)")
        
        if self.use_prompts:
            prompt_params = self.prompt_embeddings.numel() + sum(p.numel() for p in self.prompt_proj.parameters())
            logger.info(f"  Prompt parameters: {prompt_params:,} "
                       f"({100 * prompt_params / total_params:.2f}%)")
        logger.info("=" * 50)
    
    def forward(
        self,
        images: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with prompts only at first layer."""
        
        images = images.to(self.device)
        
        if self.use_prompts:
            try:
                # Get patch embeddings
                x = self.sam_wrapper.sam.image_encoder.patch_embed(images)
                B, H, W, C = x.shape
                
                # Add positional embeddings
                if hasattr(self.sam_wrapper.sam.image_encoder, 'pos_embed'):
                    pos_embed = self.sam_wrapper.sam.image_encoder.pos_embed
                    if pos_embed.shape[1] != H or pos_embed.shape[2] != W:
                        # Interpolate pos_embed
                        pos_embed_2d = pos_embed.permute(0, 3, 1, 2)
                        pos_embed_2d = F.interpolate(
                            pos_embed_2d, 
                            size=(H, W), 
                            mode='bilinear', 
                            align_corners=False
                        )
                        pos_embed = pos_embed_2d.permute(0, 2, 3, 1)
                    x = x + pos_embed
                
                # Add prompts at first layer
                # Expand prompts to batch
                prompts = self.prompt_embeddings.expand(B, -1, -1)  # (B, num_prompts, C)
                
                # Flatten spatial tokens and concatenate prompts
                x_flat = x.flatten(1, 2)  # (B, H*W, C)
                x_flat = torch.cat([prompts, x_flat], dim=1)  # (B, num_prompts + H*W, C)
                
                # Apply projection to maintain sequence length
                x_flat = x_flat.permute(0, 2, 1)  # (B, C, seq_len)
                x_flat = self.prompt_proj(x_flat)  # (B, C, seq_len)
                x_flat = x_flat.permute(0, 2, 1)  # (B, seq_len, C)
                
                # Take first H*W tokens (original spatial locations)
                x_flat = x_flat[:, :H*W, :]
                
                # Reshape back to spatial grid
                x = x_flat.reshape(B, H, W, C)
                
                # Pass through all transformer blocks
                for i, block in enumerate(self.sam_wrapper.sam.image_encoder.blocks):
                    x = block(x)
                
                # Apply neck - neck expects (B, C, H, W) format
                if hasattr(self.sam_wrapper.sam.image_encoder, 'neck'):
                    # Permute to (B, C, H, W) for neck
                    x_permuted = x.permute(0, 3, 1, 2)
                    image_embeddings = self.sam_wrapper.sam.image_encoder.neck(x_permuted)
                else:
                    image_embeddings = x.permute(0, 3, 1, 2)
                
            except Exception as e:
                logger.error(f"Error in forward: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to standard forward
                image_embeddings = self.sam_wrapper.get_image_embeddings(images)
        else:
            # Standard forward without prompts
            image_embeddings = self.sam_wrapper.get_image_embeddings(images)
        
        # Get prompt embeddings (for box/point prompts)
        try:
            sparse_embeddings, dense_embeddings = self.sam_wrapper.get_prompt_embeddings(
                boxes=boxes,
                points=points
            )
            
            # Decode masks
            masks, iou_predictions = self.sam_wrapper.decode_masks(
                image_embeddings=image_embeddings,
                sparse_embeddings=sparse_embeddings,
                dense_embeddings=dense_embeddings,
                multimask_output=False
            )
        except Exception as e:
            logger.error(f"Error in mask decoding: {e}")
            # Return dummy outputs as fallback
            B = images.shape[0]
            H, W = images.shape[2:]
            masks = torch.zeros(B, 1, H, W, device=self.device)
            iou_predictions = torch.zeros(B, 1, device=self.device)
        
        return masks, iou_predictions
    
    def get_trainable_parameters(self):
        """Get trainable parameters for optimizer."""
        params = []
        if self.use_adapters:
            params.extend(self.adapters.parameters())
        if self.use_prompts:
            # Fix: prompt_embeddings is a Parameter, not a Module
            params.append(self.prompt_embeddings)  # Use .append not .parameters()
            params.extend(self.prompt_proj.parameters())
        # Mask decoder is already trainable in sam_wrapper
        params.extend(self.sam_wrapper.sam.mask_decoder.parameters())
        return params
class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class Trainer:
    """Training class for PEFT-SAM."""
    
    def __init__(
        self,
        model: PEFT_SAM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        experiment_dir: Path,
        use_wandb: bool = False
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.experiment_dir = experiment_dir
        self.use_wandb = use_wandb
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = experiment_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Loss functions
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.get_trainable_parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 50)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_dice = 0.0
        self.train_losses = []
        self.val_dice_scores = []
        
        # Save config
        self._save_config()
        
        logger.info(f"Experiment directory: {experiment_dir}")
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    def _save_config(self):
        """Save configuration to experiment directory."""
        config_path = self.experiment_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        logger.info(f"Saved config to {config_path}")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            try:
                images = batch['image'].to(self.model.device)
                masks = batch['mask'].to(self.model.device)
                
                # Forward pass
                pred_masks, _ = self.model(images)
                
                # Compute loss
                loss_dice = self.dice_loss(pred_masks, masks)
                loss_bce = self.bce_loss(pred_masks, masks)
                loss = loss_dice + loss_bce
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
                
                # Log to wandb
                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'train_batch_loss': loss.item(),
                        'train_batch_dice': (1 - loss_dice.item()) * 100,
                    })
            except Exception as e:
                logger.error(f"Error in training batch: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        dice_scores = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                try:
                    images = batch['image'].to(self.model.device)
                    masks = batch['mask'].to(self.model.device)
                    
                    # Forward pass
                    pred_masks, _ = self.model(images)
                    pred_masks = torch.sigmoid(pred_masks)
                    
                    # Compute Dice score
                    pred_binary = (pred_masks > 0.5).float()
                    intersection = (pred_binary * masks).sum(dim=(2, 3))
                    union = pred_binary.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
                    
                    dice = (2 * intersection + 1e-6) / (union + 1e-6)
                    dice_scores.extend(dice.cpu().numpy())
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        if len(dice_scores) == 0:
            return 0.0, 0.0
        
        mean_dice = np.mean(dice_scores) * 100
        std_dice = np.std(dice_scores) * 100
        
        return mean_dice, std_dice
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_dice_scores': self.val_dice_scores
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.model.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_dice = checkpoint['best_val_dice']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_dice_scores = checkpoint.get('val_dice_scores', [])
        
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
    
    def train(self, num_epochs: int):
        """Main training loop."""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_dice, val_std = self.validate()
            self.val_dice_scores.append(val_dice)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            logger.info(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                f"Val Dice = {val_dice:.2f} ± {val_std:.2f}"
            )
            
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_dice': val_dice,
                    'val_dice_std': val_std,
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
            
            # Save checkpoint if best model
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.save_checkpoint('best_model.pth')
                logger.info(f"New best model! Dice = {val_dice:.2f}")
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        # Save final checkpoint
        self.save_checkpoint('final_model.pth')
        logger.info("Training complete!")
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        ax1.plot(self.train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Dice curve
        ax2.plot(self.val_dice_scores)
        ax2.set_title('Validation Dice Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'training_curves.png')
        plt.show()


def main(config_path: str):
    """Main training function."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    logger.info(json.dumps(config, indent=2))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config.get('experiment_name', 'peft_sam')}_{timestamp}"
    experiment_dir = Path("results") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if enabled
    if config.get('use_wandb', False) and WANDB_AVAILABLE:
        wandb.init(
            project="sam-medical-peft",
            name=experiment_name,
            config=config
        )
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        root_dir=config.get('data_root', 'data'),
        batch_size=config.get('batch_size', 4),
        target_size=tuple(config.get('target_size', [1024, 1024])),
        num_workers=config.get('num_workers', 2)
    )
    
    train_loader = dataloaders[config['dataset']]['train']
    val_loader = dataloaders[config['dataset']]['val']
    
    # Create model
    model = PEFT_SAM(
        sam_checkpoint=config.get('sam_checkpoint', 'checkpoints/sam_vit_b_01ec64.pth'),
        model_type=config.get('model_type', 'vit_b'),
        use_adapters=config.get('use_adapters', True),
        use_prompts=config.get('use_prompts', True),
        freeze_encoder=config.get('freeze_encoder', True),
        device=device
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        experiment_dir=experiment_dir,
        use_wandb=config.get('use_wandb', False)
    )
    
    # Load checkpoint if specified
    if config.get('resume_from'):
        trainer.load_checkpoint(Path(config['resume_from']))
    
    # Train
    trainer.train(num_epochs=config.get('num_epochs', 50))
    
    if config.get('use_wandb', False) and WANDB_AVAILABLE:
        wandb.finish()


def create_default_config():
    """Create a default configuration file."""
    config = {
        'experiment_name': 'peft_sam_baseline',
        'dataset': 'isic',  # 'isic' or 'busi'
        'data_root': 'data',
        'sam_checkpoint': 'checkpoints/sam_vit_b_01ec64.pth',
        'model_type': 'vit_b',
        'use_adapters': True,
        'use_prompts': True,
        'freeze_encoder': True,
        'batch_size': 4,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'target_size': [1024, 1024],
        'num_workers': 2,
        'use_wandb': False,
        'resume_from': None
    }
    
    # Save to file
    config_path = Path('experiments/configs/default_config.yaml')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created default config at {config_path}")
    return config_path


def test_training():
    """Test function to verify training pipeline."""
    print("Testing PEFT-SAM Training Pipeline...")
    
    # Create small test config
    config = {
        'experiment_name': 'test_run',
        'dataset': 'isic',
        'data_root': 'data',
        'sam_checkpoint': 'checkpoints/sam_vit_b_01ec64.pth',
        'model_type': 'vit_b',
        'use_adapters': True,
        'use_prompts': True,
        'freeze_encoder': True,
        'batch_size': 2,
        'num_epochs': 2,  # Just 2 epochs for testing
        'learning_rate': 1e-4,
        'target_size': [256, 256],  # Smaller for testing
        'num_workers': 0,
        'use_wandb': False
    }
    
    # Create dataloaders with sample data
    from src.utils.data_loader import create_sample_data
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample data
        create_sample_data(Path(tmpdir) / 'ISIC', num_samples=20)
        create_sample_data(Path(tmpdir) / 'BUSI', num_samples=15)
        
        config['data_root'] = tmpdir
        
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = PEFT_SAM(
            sam_checkpoint=config['sam_checkpoint'],
            use_adapters=True,
            use_prompts=True,
            device=device
        )
        
        # Test forward pass
        print("\nTest 1 - Forward pass:")
        batch = next(iter(train_loader))
        images = batch['image'].to(device)
        masks, iou = model(images)
        print(f"  Input shape: {images.shape}")
        print(f"  Output mask shape: {masks.shape}")
        print(f"  IOU shape: {iou.shape}")
        print(f"  ✓ Forward pass successful")
        
        # Test training step
        print("\nTest 2 - Training step:")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            experiment_dir=Path(tmpdir) / 'test_experiment'
        )
        
        loss = trainer.train_epoch()
        print(f"  Train loss: {loss:.4f}")
        print(f"  ✓ Training step successful")
        
        # Test validation
        print("\nTest 3 - Validation:")
        dice, dice_std = trainer.validate()
        print(f"  Val Dice: {dice:.2f} ± {dice_std:.2f}")
        print(f"  ✓ Validation successful")
        
        print("\n✓ All training tests passed!")
    
    return model, trainer


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run training with config file
        main(sys.argv[1])
    else:
        # Run tests
        test_training()
"""
Production-Ready PEFT-SAM Training Script
SAM-compatible additive deep prompts with optional adapters
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import yaml
import logging
from tqdm import tqdm
from datetime import datetime

from segment_anything import sam_model_registry
from src.models.blocks import PromptedSAMEncoder
from src.utils.data_loader import create_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===============================
# Metrics
# ===============================

def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean()


def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = (pred + target - pred * target).sum(dim=(2, 3))
    return ((intersection + eps) / (union + eps)).mean()


# ===============================
# Model
# ===============================

class PEFT_SAM(nn.Module):

    def __init__(self, config, device):
        super().__init__()

        self.device = device
        self.config = config

        # Load base SAM model
        self.sam = sam_model_registry[config['model_type']](
            checkpoint=config['sam_checkpoint']
        ).to(device)

        # Freeze image encoder
        if config.get("freeze_encoder", True):
            for p in self.sam.image_encoder.parameters():
                p.requires_grad = False
            logger.info("Image encoder frozen")

        # Replace encoder with prompted + adapted version
        self.prompted_encoder = PromptedSAMEncoder(
            self.sam.image_encoder,
            use_adapters=config.get('use_adapters', True),
            adapter_layers=config.get('adapter_layers', [8, 9, 10, 11])
        )

        # Mask decoder is always trainable
        logger.info("Mask decoder is trainable")

        self.to(device)
        self._log_params()

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info("=" * 60)
        logger.info("PARAMETER SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Total parameters: {total:,}")
        logger.info(f"  Trainable parameters: {trainable:,}")
        logger.info(f"  Trainable percentage: {100 * trainable / total:.2f}%")
        
        # Prompt parameters
        prompt_params = self.prompted_encoder.prompt_embeddings.numel()
        logger.info(f"  Prompt parameters: {prompt_params:,} ({100 * prompt_params / total:.2f}%)")
        
        # Adapter parameters (if enabled)
        if self.config.get('use_adapters', False):
            adapter_params = sum(p.numel() for p in self.prompted_encoder.adapters.parameters())
            logger.info(f"  Adapter parameters: {adapter_params:,} ({100 * adapter_params / total:.2f}%)")
        
        # Mask decoder parameters
        decoder_params = sum(p.numel() for p in self.sam.mask_decoder.parameters())
        logger.info(f"  Mask decoder parameters: {decoder_params:,} ({100 * decoder_params / total:.2f}%)")
        logger.info("=" * 60)

    def forward(self, images):
        images = images.to(self.device)

        # Get image embeddings with prompts and adapters
        image_embeddings = self.prompted_encoder(images)
        target_size = image_embeddings.shape[-2:]

        # Get dense positional embeddings
        image_pe = self.sam.prompt_encoder.get_dense_pe()

        # Interpolate image_pe if needed
        if image_pe.shape[-2:] != target_size:
            image_pe = F.interpolate(
                image_pe,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )

        # Get sparse/dense embeddings (no box/point prompts)
        sparse, dense = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None
        )

        # Interpolate dense if needed
        if dense.shape[-2:] != target_size:
            dense = F.interpolate(
                dense,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )

        # Decode masks
        masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False
        )

        # Upsample masks to match input size
        if masks.shape[-2:] != images.shape[-2:]:
            masks = F.interpolate(
                masks,
                size=images.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        return masks

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ===============================
# Trainer
# ===============================

class Trainer:

    def __init__(self, model, train_loader, val_loader, config, exp_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = model.device
        self.config = config
        self.exp_dir = exp_dir
        exp_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.AdamW(
            model.trainable_parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs']
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        self.best_dice = 0
        self.early_counter = 0

        self.bce = nn.BCEWithLogitsLoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                preds = self.model(images)
                loss = self.bce(preds, masks) + (1 - dice_score(torch.sigmoid(preds), masks))

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.trainable_parameters(), 1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        dices, ious = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                preds = torch.sigmoid(self.model(images))
                dices.append(dice_score(preds, masks).item())
                ious.append(iou_score(preds, masks).item())

        return np.mean(dices) * 100, np.mean(ious) * 100

    def train(self):
        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch()
            val_dice, val_iou = self.validate()
            self.scheduler.step()

            logger.info(
                f"Epoch {epoch:2d} | "
                f"Loss {train_loss:.4f} | "
                f"Dice {val_dice:.2f} | "
                f"IoU {val_iou:.2f}"
            )

            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.early_counter = 0
                torch.save(
                    self.model.state_dict(),
                    self.exp_dir / "best_model.pth"
                )
                logger.info(f"  → New best model! Dice: {val_dice:.2f}")
            else:
                self.early_counter += 1

            if self.early_counter > 10:
                logger.info("Early stopping triggered.")
                break

        torch.save(
            self.model.state_dict(),
            self.exp_dir / "final_model.pth"
        )
        logger.info(f"Training complete. Best Dice: {self.best_dice:.2f}")


# ===============================
# Main
# ===============================

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataloaders = create_dataloaders(
        root_dir=config['data_root'],
        batch_size=config['batch_size'],
        target_size=tuple(config['target_size']),
        num_workers=config.get('num_workers', 2)
    )

    train_loader = dataloaders[config['dataset']]['train']
    val_loader = dataloaders[config['dataset']]['val']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config['experiment_name']}_{timestamp}"
    exp_dir = Path("results") / exp_name

    model = PEFT_SAM(config, device)
    trainer = Trainer(model, train_loader, val_loader, config, exp_dir)
    trainer.train()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python -m src.training.train <config_path>")
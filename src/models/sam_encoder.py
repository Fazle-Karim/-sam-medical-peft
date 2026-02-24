"""
SAM Encoder Wrapper for Medical Image Segmentation
Frozen image encoder with trainable mask decoder
"""

import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from typing import Optional, Tuple, List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAMEncoderWrapper(nn.Module):
    """
    Wrapper for SAM with frozen image encoder.
    
    This module loads a pretrained SAM model and freezes the image encoder
    while keeping the mask decoder trainable. This forms the foundation
    for parameter-efficient fine-tuning with prompts and adapters.
    
    Args:
        model_type (str): SAM model type ('vit_b', 'vit_l', 'vit_h')
        checkpoint_path (str): Path to SAM checkpoint
        freeze_encoder (bool): Whether to freeze image encoder
        freeze_decoder (bool): Whether to freeze mask decoder (usually False)
        device (str): Device to load model on ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_path: str = "checkpoints/sam_vit_b_01ec64.pth",
        freeze_encoder: bool = True,
        freeze_decoder: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        logger.info(f"Loading SAM {model_type} from {checkpoint_path}")
        logger.info(f"Using device: {device}")
        
        # Load SAM model
        try:
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam.to(device)
            logger.info("SAM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            raise
        
        # Get model dimensions
        self.encoder_dim = self._get_encoder_dim()
        logger.info(f"Encoder dimension: {self.encoder_dim}")
        
        # Freeze components as specified
        self._freeze_components(freeze_encoder, freeze_decoder)
        
        # Count and log trainable parameters
        self._log_parameter_counts()
    
    def _get_encoder_dim(self) -> int:
        """Get the dimension of the image encoder output."""
        # For ViT-B, dimension is 768
        # For ViT-L, dimension is 1024
        # For ViT-H, dimension is 1280
        if self.model_type == "vit_b":
            return 768
        elif self.model_type == "vit_l":
            return 1024
        elif self.model_type == "vit_h":
            return 1280
        else:
            # Try to infer from model
            dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
            with torch.no_grad():
                features = self.sam.image_encoder(dummy_input)
            return features.shape[1]
    
    def _freeze_components(self, freeze_encoder: bool, freeze_decoder: bool):
        """Freeze specified components of the model."""
        
        # Freeze image encoder if requested
        if freeze_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
            logger.info("Image encoder frozen")
        else:
            logger.info("Image encoder will be fine-tuned")
        
        # Freeze mask decoder if requested (usually we want it trainable)
        if freeze_decoder:
            for param in self.sam.mask_decoder.parameters():
                param.requires_grad = False
            logger.info("Mask decoder frozen")
        else:
            logger.info("Mask decoder is trainable")
        
        # Prompt encoder is always trainable (it's lightweight)
        logger.info("Prompt encoder is trainable")
    
    def _log_parameter_counts(self):
        """Log the number of trainable and total parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    def get_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get image embeddings from the frozen encoder.
        
        Args:
            images: Input images tensor of shape (B, 3, H, W)
                   H and W should be multiples of 16 (SAM requirement)
        
        Returns:
            Image embeddings of shape (B, encoder_dim, H//16, W//16)
        """
        return self.sam.image_encoder(images)
    
    def get_prompt_embeddings(
        self,
        boxes: Optional[torch.Tensor] = None,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prompt embeddings from the prompt encoder.
        
        Args:
            boxes: Bounding boxes of shape (B, N, 4) in xyxy format
            points: Tuple of (point_coords, point_labels)
            masks: Input masks for mask prompts
        
        Returns:
            Tuple of (sparse_embeddings, dense_embeddings)
        """
        return self.sam.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks
        )
    
    def decode_masks(
        self,
        image_embeddings: torch.Tensor,
        sparse_embeddings: torch.Tensor,
        dense_embeddings: torch.Tensor,
        multimask_output: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode masks from image and prompt embeddings.
        
        Args:
            image_embeddings: Image embeddings from get_image_embeddings()
            sparse_embeddings: Sparse prompt embeddings from get_prompt_embeddings()
            dense_embeddings: Dense prompt embeddings from get_prompt_embeddings()
            multimask_output: Whether to output multiple masks
        
        Returns:
            Tuple of (masks, iou_predictions)
        """
        return self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
    
    def forward(
        self,
        images: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        multimask_output: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the entire SAM model.
        
        Args:
            images: Input images tensor of shape (B, 3, H, W)
            boxes: Optional bounding boxes of shape (B, N, 4)
            points: Optional (point_coords, point_labels)
            multimask_output: Whether to output multiple masks
        
        Returns:
            Tuple of (masks, iou_predictions)
        """
        # Get image embeddings
        image_embeddings = self.get_image_embeddings(images)
        
        # Get prompt embeddings
        sparse_embeddings, dense_embeddings = self.get_prompt_embeddings(
            boxes=boxes,
            points=points,
            masks=None
        )
        
        # Decode masks
        masks, iou_predictions = self.decode_masks(
            image_embeddings=image_embeddings,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        
        return masks, iou_predictions
    
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for SAM input.
        
        Args:
            image: Input image tensor of shape (C, H, W) or (B, C, H, W)
                  Values should be in [0, 255] range
        
        Returns:
            Preprocessed image ready for SAM
        """
        # SAM expects images in [0, 255] range, normalized by its own transform
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # SAM's built-in preprocessing will handle normalization
        return image
    
    def get_encoder(self):
        """Get the image encoder module."""
        return self.sam.image_encoder
    
    def get_decoder(self):
        """Get the mask decoder module."""
        return self.sam.mask_decoder
    
    def get_prompt_encoder(self):
        """Get the prompt encoder module."""
        return self.sam.prompt_encoder


# Quick test function
def test_sam_encoder():
    """Simple test to verify SAM loads and encoder works."""
    print("Testing SAMEncoderWrapper...")
    
    # Initialize model
    model = SAMEncoderWrapper(
        model_type="vit_b",
        checkpoint_path="checkpoints/sam_vit_b_01ec64.pth"
    )
    
    # Create dummy input of correct size for SAM (1024x1024)
    dummy_image = torch.randn(1, 3, 1024, 1024)
    
    print("Testing image encoder only...")
    with torch.no_grad():
        embeddings = model.get_image_embeddings(dummy_image)
    
    print(f"Image embeddings shape: {embeddings.shape}")
    print("✓ Encoder test passed!")
    
    return model
if __name__ == "__main__":
    # Run test if script is executed directly
    test_sam_encoder()
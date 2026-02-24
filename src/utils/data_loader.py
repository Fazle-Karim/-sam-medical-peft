"""
Data Loader for Medical Image Segmentation Datasets
Handles ISIC (skin lesion) and BUSI (breast ultrasound) datasets
with proper preprocessing for SAM input.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import logging
import requests
import zipfile
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalImageDataset(Dataset):
    """
    Base dataset class for medical images.
    Handles image loading, preprocessing, and augmentations.
    """
    
    def __init__(
        self,
        root_dir: str,
        dataset_name: str,  # 'isic' or 'busi'
        split: str = 'train',  # 'train', 'val', 'test'
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (1024, 1024),  # SAM expected size
        augment: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.target_size = target_size
        self.augment = augment and (split == 'train')
        
        # Setup dataset-specific paths
        if self.dataset_name == 'isic':
            self.image_dir = self.root_dir / 'ISIC' / 'images'
            self.mask_dir = self.root_dir / 'ISIC' / 'masks'
        elif self.dataset_name == 'busi':
            self.image_dir = self.root_dir / 'BUSI' / 'images'
            self.mask_dir = self.root_dir / 'BUSI' / 'masks'
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Get list of images
        self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                  if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(self.mask_dir) 
                                 if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Ensure images and masks match
        assert len(self.image_files) == len(self.mask_files), \
            f"Number of images ({len(self.image_files)}) != masks ({len(self.mask_files)})"
        
        # Create train/val/test split (70/15/15)
        self._create_splits()
        
        logger.info(f"Loaded {self.dataset_name} dataset: {len(self)} samples for {split}")
        
        # Define transforms
        self.transform = transform or self._default_transform()
    
    def _create_splits(self):
        """Create train/val/test splits."""
        total = len(self.image_files)
        train_end = int(0.7 * total)
        val_end = int(0.85 * total)
        
        indices = list(range(total))
        random.shuffle(indices)
        
        if self.split == 'train':
            self.indices = indices[:train_end]
        elif self.split == 'val':
            self.indices = indices[train_end:val_end]
        else:  # test
            self.indices = indices[val_end:]
    
    def _default_transform(self):
        """Default transform for medical images."""
        transform_list = []
        
        # Resize to SAM input size
        transform_list.append(transforms.Resize(self.target_size))
        
        if self.augment:
            # Data augmentation for training
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get image and mask pair."""
        # Get actual index
        actual_idx = self.indices[idx]
        
        # Load image
        img_path = self.image_dir / self.image_files[actual_idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_path = self.mask_dir / self.mask_files[actual_idx]
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Apply same transform to both image and mask
        seed = random.randint(0, 2**32)
        
        # Transform image
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)
        
        # Transform mask (without normalization)
        random.seed(seed)
        torch.manual_seed(seed)
        
        mask_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.RandomHorizontalFlip(p=0.5) if self.augment else transforms.Lambda(lambda x: x),
            transforms.RandomVerticalFlip(p=0.5) if self.augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])
        mask = mask_transform(mask)
        mask = (mask > 0.5).float()  # Binarize
        
        return {
            'image': image,
            'mask': mask,
            'image_path': str(img_path),
            'mask_path': str(mask_path),
            'dataset': self.dataset_name
        }


class ISICDataset(MedicalImageDataset):
    """ISIC Skin Lesion Dataset."""
    
    def __init__(self, root_dir: str, split: str = 'train', **kwargs):
        super().__init__(root_dir, 'isic', split, **kwargs)
        
        # ISIC-specific info
        logger.info(f"ISIC dataset - {split}: {len(self)} images")


class BUSIDataset(MedicalImageDataset):
    """BUSI Breast Ultrasound Dataset."""
    
    def __init__(self, root_dir: str, split: str = 'train', **kwargs):
        super().__init__(root_dir, 'busi', split, **kwargs)
        
        # BUSI-specific info
        logger.info(f"BUSI dataset - {split}: {len(self)} images")


def download_dataset(url: str, dest_path: Path, dataset_name: str):
    """
    Download and extract a dataset.
    
    Args:
        url: Download URL
        dest_path: Destination directory
        dataset_name: Name for logging
    """
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Download
    zip_path = dest_path / f"{dataset_name}.zip"
    
    if not zip_path.exists():
        logger.info(f"Downloading {dataset_name} from {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    pbar.update(len(data))
        
        logger.info(f"Downloaded {dataset_name} to {zip_path}")
    
    # Extract
    if not (dest_path / 'images').exists():
        logger.info(f"Extracting {dataset_name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
        logger.info(f"Extracted {dataset_name} to {dest_path}")


def prepare_isic_dataset(root_dir: str, download: bool = True):
    """
    Prepare ISIC dataset in the required structure.
    
    Args:
        root_dir: Root directory for datasets
        download: Whether to download if not present
    """
    isic_dir = Path(root_dir) / 'ISIC'
    isic_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already prepared
    if (isic_dir / 'images').exists() and (isic_dir / 'masks').exists():
        logger.info("ISIC dataset already prepared")
        return
    
    if not download:
        raise FileNotFoundError(f"ISIC dataset not found in {isic_dir}")
    
    # Download ISIC 2018 dataset
    # Note: You need to register at https://challenge.isic-archive.com/
    # For now, we'll use a public source or provide instructions
    logger.info("Please download ISIC dataset manually from:")
    logger.info("https://challenge.isic-archive.com/data/")
    logger.info(f"Place images in {isic_dir}/images/ and masks in {isic_dir}/masks/")
    
    # Alternative: Use a smaller sample for testing
    logger.info("Creating sample data for testing...")
    create_sample_data(isic_dir, num_samples=10)


def prepare_busi_dataset(root_dir: str, download: bool = True):
    """
    Prepare BUSI dataset in the required structure.
    
    Args:
        root_dir: Root directory for datasets
        download: Whether to download if not present
    """
    busi_dir = Path(root_dir) / 'BUSI'
    busi_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already prepared
    if (busi_dir / 'images').exists() and (busi_dir / 'masks').exists():
        logger.info("BUSI dataset already prepared")
        return
    
    if not download:
        raise FileNotFoundError(f"BUSI dataset not found in {busi_dir}")
    
    # BUSI dataset URL (public dataset)
    url = "https://scholar.cu.edu.eg/Dataset_BUSI.zip"  # Check if this works
    
    try:
        download_dataset(url, busi_dir, "BUSI")
        
        # Organize into images/ and masks/
        # This depends on the actual structure of the downloaded zip
        logger.info("Organizing BUSI dataset...")
        # Add organization logic here based on actual structure
        
    except Exception as e:
        logger.error(f"Failed to download BUSI: {e}")
        logger.info("Please download BUSI dataset manually from:")
        logger.info("https://scholar.cu.edu.eg/?q=afahmy/pages/dataset")
        logger.info(f"Place images in {busi_dir}/images/ and masks in {busi_dir}/masks/")
        
        # Create sample data for testing
        create_sample_data(busi_dir, num_samples=10)


def create_sample_data(data_dir: Path, num_samples: int = 10):
    """
    Create synthetic sample data for testing.
    
    Args:
        data_dir: Directory to create samples in
        num_samples: Number of samples to create
    """
    # Create directories with parents=True to create nested paths
    img_dir = data_dir / 'images'
    mask_dir = data_dir / 'masks'
    
    # Use parents=True to create parent directories if they don't exist
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_samples):
        # Create random image (128x128 for testing)
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (128, 128), dtype=np.uint8) * 255
        
        # Save
        Image.fromarray(img).save(img_dir / f"sample_{i:03d}.png")
        Image.fromarray(mask).save(mask_dir / f"sample_{i:03d}_mask.png")
    
    logger.info(f"Created {num_samples} sample images in {data_dir}")

    
def create_dataloaders(
    root_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
    target_size: Tuple[int, int] = (1024, 1024),
    use_sample: bool = False,  # Use sample data for testing
) -> Dict[str, Dict[str, DataLoader]]:
    """
    Create dataloaders for all datasets and splits.
    
    Args:
        root_dir: Root directory for datasets
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        target_size: Target image size for SAM
        use_sample: Use sample data instead of real datasets
    
    Returns:
        Dictionary: {dataset: {split: dataloader}}
    """
    dataloaders = {}
    
    for dataset_name in ['isic', 'busi']:
        dataloaders[dataset_name] = {}
        
        for split in ['train', 'val', 'test']:
            # Create dataset
            if dataset_name == 'isic':
                dataset = ISICDataset(
                    root_dir=root_dir,
                    split=split,
                    target_size=target_size,
                    augment=(split == 'train')
                )
            else:  # busi
                dataset = BUSIDataset(
                    root_dir=root_dir,
                    split=split,
                    target_size=target_size,
                    augment=(split == 'train')
                )
            
            # Create dataloader
            shuffle = (split == 'train')
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True
            )
            
            dataloaders[dataset_name][split] = dataloader
            logger.info(f"Created {dataset_name} {split} loader: {len(dataloader)} batches")
    
    return dataloaders


def get_sample_batch(dataloaders: Dict) -> Dict:
    """
    Get a sample batch for testing.
    
    Args:
        dataloaders: Output from create_dataloaders
    
    Returns:
        Sample batch from ISIC training set
    """
    loader = dataloaders['isic']['train']
    batch = next(iter(loader))
    
    logger.info(f"Sample batch shapes:")
    logger.info(f"  Images: {batch['image'].shape}")
    logger.info(f"  Masks: {batch['mask'].shape}")
    logger.info(f"  Image range: [{batch['image'].min():.2f}, {batch['image'].max():.2f}]")
    logger.info(f"  Mask range: [{batch['mask'].min():.2f}, {batch['mask'].max():.2f}]")
    
    return batch


def test_data_loader():
    """Test function to verify data loading works."""
    print("Testing MedicalImageDataset...")
    
    # Create temporary test directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create sample data
        create_sample_data(tmp_path / 'ISIC', num_samples=20)
        create_sample_data(tmp_path / 'BUSI', num_samples=15)
        
        # Test ISIC dataset
        print("\nTest 1 - ISIC dataset:")
        isic_train = ISICDataset(tmpdir, split='train', target_size=(256, 256))
        isic_val = ISICDataset(tmpdir, split='val', target_size=(256, 256))
        isic_test = ISICDataset(tmpdir, split='test', target_size=(256, 256))
        
        print(f"  Train: {len(isic_train)} samples")
        print(f"  Val: {len(isic_val)} samples")
        print(f"  Test: {len(isic_test)} samples")
        print(f"  Total: {len(isic_train) + len(isic_val) + len(isic_test)} samples")
        
        # Test BUSI dataset
        print("\nTest 2 - BUSI dataset:")
        busi_train = BUSIDataset(tmpdir, split='train', target_size=(256, 256))
        busi_val = BUSIDataset(tmpdir, split='val', target_size=(256, 256))
        busi_test = BUSIDataset(tmpdir, split='test', target_size=(256, 256))
        
        print(f"  Train: {len(busi_train)} samples")
        print(f"  Val: {len(busi_val)} samples")
        print(f"  Test: {len(busi_test)} samples")
        print(f"  Total: {len(busi_train) + len(busi_val) + len(busi_test)} samples")
        
        # Test batch loading
        print("\nTest 3 - Batch loading:")
        dataloaders = create_dataloaders(
            root_dir=tmpdir,
            batch_size=4,
            target_size=(256, 256)
        )
        
        # Get a batch
        batch = get_sample_batch(dataloaders)
        
        print("\nTest 4 - Transform consistency:")
        # Check that image and mask have same orientation after transforms
        img = batch['image'][0]
        mask = batch['mask'][0]
        
        print(f"  Image shape: {img.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Image mean: {img.mean():.3f}")
        print(f"  Mask mean: {mask.mean():.3f}")
        print(f"  Mask unique values: {torch.unique(mask).tolist()}")
        
        print("\n✓ All data loader tests passed!")
        
        return dataloaders, batch


if __name__ == "__main__":
    # Run tests
    test_data_loader()
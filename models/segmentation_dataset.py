"""
PyTorch Dataset for Lesion Segmentation using ISLES2015 data.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationStrokeDataset(Dataset):
    """Dataset for lesion segmentation using preprocessed MRI slices and masks."""

    def __init__(self, 
                 data_dir: str, 
                 modalities: Optional[List[str]] = None, 
                 transform: Optional[A.Compose] = None,
                 use_all_slices: bool = True):
        """
        Initialize segmentation dataset.
        
        Args:
            data_dir: Directory with preprocessed patient data
            modalities: List of modalities to use (default: ['DWI'])
            transform: Albumentations transforms for augmentation
            use_all_slices: Whether to use all slices or just 3 center slices
        """
        self.data_dir = Path(data_dir)
        self.modalities = modalities or ["DWI"]
        self.transform = transform
        self.use_all_slices = use_all_slices
        
        # Get patient IDs
        self.patient_ids = sorted([p.name for p in self.data_dir.iterdir() if p.is_dir()])
        
        # Build slice-level index for training
        self.slice_indices = []
        for patient_id in self.patient_ids:
            patient_dir = self.data_dir / patient_id
            modality_dir = patient_dir / self.modalities[0]
            if modality_dir.exists():
                slice_files = sorted(modality_dir.glob("*.png"))
                if use_all_slices:
                    for i, slice_file in enumerate(slice_files):
                        self.slice_indices.append((patient_id, i))
                else:
                    # Use only 3 center slices
                    if len(slice_files) >= 3:
                        mid = len(slice_files) // 2
                        for i in range(mid - 1, mid + 2):
                            self.slice_indices.append((patient_id, i))
        
        print(f"Loaded {len(self.slice_indices)} slices from {len(self.patient_ids)} patients")

    def _load_slice(self, patient_dir: Path, slice_idx: int) -> np.ndarray:
        """Load a single slice from specified modality."""
        modality_dir = patient_dir / self.modalities[0]
        slice_files = sorted(modality_dir.glob("*.png"))
        
        if slice_idx >= len(slice_files):
            raise ValueError(f"Slice index {slice_idx} out of range for {patient_dir.name}")
        
        img = cv2.imread(str(slice_files[slice_idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load slice {slice_idx} for {patient_dir.name}")
        
        return img.astype(np.float32) / 255.0

    def _load_mask_slice(self, patient_dir: Path, slice_idx: int) -> np.ndarray:
        """Load corresponding mask slice."""
        mask_dir = patient_dir / "mask"
        mask_files = sorted(mask_dir.glob("*.png"))
        
        if slice_idx >= len(mask_files):
            # If mask has fewer slices, use the last one
            slice_idx = len(mask_files) - 1
        
        img = cv2.imread(str(mask_files[slice_idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load mask slice {slice_idx} for {patient_dir.name}")
        
        return img.astype(np.float32) / 255.0

    def __len__(self) -> int:
        return len(self.slice_indices)

    def __getitem__(self, idx: int):
        patient_id, slice_idx = self.slice_indices[idx]
        patient_dir = self.data_dir / patient_id
        
        # Load image and mask
        image = self._load_slice(patient_dir, slice_idx)
        mask = self._load_mask_slice(patient_dir, slice_idx)
        
        # Apply transforms if provided
        if self.transform:
            # Albumentations expects (H, W) for grayscale
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            # Ensure mask has channel dimension
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            # Convert to tensor manually
            image = torch.from_numpy(image).unsqueeze(0).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return {
            "image": image,
            "mask": mask,
            "patient_id": patient_id,
            "slice_idx": slice_idx
        }


def get_train_transforms() -> A.Compose:
    """Get training transforms for segmentation."""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ElasticTransform(p=0.3),
        A.GaussNoise(p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.RandomGamma(p=1.0),
        ], p=0.3),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])

def get_val_transforms() -> A.Compose:
    """Get validation transforms for segmentation."""
    return A.Compose([
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])


def create_segmentation_dataloaders(data_dir: str, 
                                   batch_size: int = 4,
                                   num_workers: int = 0,
                                   use_all_slices: bool = True) -> Tuple:
    """Create train and validation dataloaders for segmentation."""
    
    # Load full dataset
    full_dataset = SegmentationStrokeDataset(
        data_dir=data_dir,
        modalities=['DWI'],
        transform=None,
        use_all_slices=use_all_slices
    )
    
    # Stratified split based on patient (not slices)
    patient_ids = full_dataset.patient_ids
    from sklearn.model_selection import train_test_split
    train_patients, val_patients = train_test_split(
        patient_ids, test_size=0.2, random_state=42
    )
    
    # Create filtered datasets
    train_indices = [i for i, (pid, _) in enumerate(full_dataset.slice_indices) if pid in train_patients]
    val_indices = [i for i, (pid, _) in enumerate(full_dataset.slice_indices) if pid in val_patients]
    
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Apply transforms
    train_dataset.dataset.transform = get_train_transforms()
    val_dataset.dataset.transform = get_val_transforms()
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train slices: {len(train_dataset)}, Val slices: {len(val_dataset)}")
    print(f"Train patients: {len(train_patients)}, Val patients: {len(val_patients)}")
    
    return train_loader, val_loader

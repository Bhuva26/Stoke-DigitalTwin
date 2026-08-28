"""
PyTorch Dataset for ISLES2015 stroke data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import json
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.append(str(Path(__file__).parent.parent))


class StrokeDataset(Dataset):
    """PyTorch Dataset for stroke classification."""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'training',
                 modalities: List[str] = ['DWI'],
                 transform: Optional[A.Compose] = None,
                 use_mask: bool = True,
                 patient_ids: Optional[List[str]] = None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing preprocessed data
            split: 'training' or 'testing'
            modalities: List of modalities to use
            transform: Albumentations transform
            use_mask: Whether to use mask information
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.modalities = modalities
        self.transform = transform
        self.use_mask = use_mask
        
        # Load patient list
        self.patients = self._get_patient_list() if patient_ids is None else sorted(patient_ids)
        
        print(f"Loaded {len(self.patients)} patients for {split} split")
    
    def _get_patient_list(self) -> List[str]:
        """Get list of patient IDs."""
        patient_dir = self.data_dir
        if not patient_dir.exists():
            raise ValueError(f"Data directory {patient_dir} does not exist")
        
        patients = [p.name for p in patient_dir.iterdir() if p.is_dir()]
        return sorted(patients)
    
    def _load_slices(self, patient_dir: Path, modality: str) -> np.ndarray:
        """
        Load all slices for a modality.
        
        Args:
            patient_dir: Patient directory
            modality: Modality name
            
        Returns:
            Stack of slices as numpy array (3, H, W)
        """
        modality_dir = patient_dir / modality
        if not modality_dir.exists():
            raise ValueError(f"Modality directory {modality_dir} does not exist")
        
        # Load all slice files
        slice_files = sorted(modality_dir.glob("*.png"))
        if len(slice_files) != 3:
            print(f"Warning: Expected 3 slices, found {len(slice_files)} for {modality}")
        
        slices = []
        for slice_file in slice_files[:3]:  # Take first 3 slices
            img = cv2.imread(str(slice_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                slices.append(img)
        
        if len(slices) == 0:
            raise ValueError(f"No slices found for {modality}")
        
        # Stack as channels (3, H, W)
        stacked = np.stack(slices, axis=0)
        
        return stacked
    
    def _load_mask(self, patient_dir: Path) -> Optional[np.ndarray]:
        """
        Load mask slices.
        
        Args:
            patient_dir: Patient directory
            
        Returns:
            Stack of mask slices as numpy array (3, H, W)
        """
        mask_dir = patient_dir / 'mask'
        if not mask_dir.exists():
            return None
        
        # Load all mask files
        mask_files = sorted(mask_dir.glob("*.png"))
        if len(mask_files) != 3:
            print(f"Warning: Expected 3 mask slices, found {len(mask_files)}")
        
        masks = []
        for mask_file in mask_files[:3]:  # Take first 3 slices
            img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                masks.append(img)
        
        if len(masks) == 0:
            return None
        
        # Stack as channels (3, H, W)
        stacked = np.stack(masks, axis=0)
        
        return stacked
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.patients)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary containing:
                - image: Tensor of shape (C, H, W)
                - label: Tensor (0 or 1)
                - mask: Optional tensor
                - patient_id: str
        """
        patient_id = self.patients[idx]
        patient_dir = self.data_dir / patient_id
        
        # Determine label based on mask presence
        # If mask exists and has non-zero values, label = 1 (stroke)
        # Otherwise label = 0 (no stroke)
        mask_dir = patient_dir / 'mask'
        if mask_dir.exists():
            mask_files = list(mask_dir.glob("*.png"))
            if mask_files:
                total_mask_sum = 0
                for mask_file in mask_files:
                    mask_img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    if mask_img is not None:
                        total_mask_sum += np.sum(mask_img)
                label = 1 if total_mask_sum > 0 else 0
            else:
                label = 0
        else:
            label = 0
        
        # Load modality data
        if len(self.modalities) == 1:
            # Single modality
            modality = self.modalities[0]
            try:
                image = self._load_slices(patient_dir, modality)
            except ValueError:
                return {
                    'image': torch.zeros((3, 224, 224), dtype=torch.float32),
                    'label': torch.tensor(0, dtype=torch.long),
                    'mask': None,
                    'patient_id': patient_id
                }
            
            # Apply transforms
            if self.transform:
                # Convert to HWC for albumentations
                image_hwc = np.transpose(image, (1, 2, 0))
                transformed = self.transform(image=image_hwc)
                image = transformed['image']
        else:
            # Multi-modal - stack modalities as channels
            modality_slices = []
            for modality in self.modalities:
                slices = self._load_slices(patient_dir, modality)
                modality_slices.append(slices)
            
            # Stack all modalities (num_modalities * 3, H, W)
            image = np.concatenate(modality_slices, axis=0)
            
            # Apply transforms if needed
            if self.transform:
                # For multi-channel, we might need custom handling
                # For now, just normalize
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image)
        
        # Load mask if needed
        mask = None
        if self.use_mask:
            mask_data = self._load_mask(patient_dir)
            if mask_data is not None:
                mask = torch.from_numpy(mask_data).float() / 255.0
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'mask': mask,
            'patient_id': patient_id
        }


class MultiModalStrokeDataset(Dataset):
    """Dataset for multi-modal stroke classification."""
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'training',
                 modalities: List[str] = ['DWI', 'Flair', 'T1', 'T2'],
                 transform: Optional[A.Compose] = None,
                 patient_ids: Optional[List[str]] = None):
        """
        Initialize the multi-modal dataset.
        
        Args:
            data_dir: Directory containing preprocessed data
            split: 'training' or 'testing'
            modalities: List of modalities to use
            transform: Albumentations transform
            patient_ids: Optional list of specific patient IDs to use
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.modalities = modalities
        self.transform = transform
        
        # Load patient list
        if patient_ids is not None:
            self.patients = patient_ids
        else:
            self.patients = self._get_patient_list()
        
        print(f"Loaded {len(self.patients)} patients for {split} split (multi-modal)")
    
    def _get_patient_list(self) -> List[str]:
        """Get list of patient IDs."""
        patient_dir = self.data_dir
        if not patient_dir.exists():
            raise ValueError(f"Data directory {patient_dir} does not exist")
        
        patients = [p.name for p in patient_dir.iterdir() if p.is_dir()]
        return sorted(patients)
    
    def _load_modality(self, patient_dir: Path, modality: str) -> np.ndarray:
        """Load slices for a specific modality."""
        modality_dir = patient_dir / modality
        if not modality_dir.exists():
            # Return zeros if modality doesn't exist
            return np.zeros((3, 224, 224), dtype=np.uint8)
        
        slice_files = sorted(modality_dir.glob("*.png"))
        slices = []
        for slice_file in slice_files[:3]:
            img = cv2.imread(str(slice_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                slices.append(img)
        
        if len(slices) == 0:
            return np.zeros((3, 224, 224), dtype=np.uint8)
        
        stacked = np.stack(slices, axis=0)
        return stacked
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.patients)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary containing modality tensors and label
        """
        patient_id = self.patients[idx]
        patient_dir = self.data_dir / patient_id
        
        # Determine label
        mask_dir = patient_dir / 'mask'
        if mask_dir.exists():
            mask_files = list(mask_dir.glob("*.png"))
            if mask_files:
                mask_img = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
                label = 1 if np.sum(mask_img) > 0 else 0
            else:
                label = 0
        else:
            label = 0
        
        # Load each modality
        modality_data = {}
        for modality in self.modalities:
            slices = self._load_modality(patient_dir, modality)
            
            # Apply transforms
            if self.transform:
                slices_hwc = np.transpose(slices, (1, 2, 0))
                transformed = self.transform(image=slices_hwc)
                slices = transformed['image']
            
            modality_data[modality] = slices
        
        return {
            'modalities': modality_data,
            'label': torch.tensor(label, dtype=torch.long),
            'patient_id': patient_id
        }


def get_train_transforms() -> A.Compose:
    """Get training data augmentation transforms."""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms() -> A.Compose:
    """Get validation/test transforms."""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_label_distribution(dataset: Dataset) -> np.ndarray:
    """Return the class counts for a dataset."""
    labels = [dataset[i]['label'].item() for i in range(len(dataset))]
    return np.bincount(labels, minlength=2)


def validate_label_distribution(dataset: Dataset, name: str = 'dataset') -> np.ndarray:
    """Ensure the dataset has both positive and negative classes for binary training."""
    counts = get_label_distribution(dataset)
    if len(counts) < 2 or np.count_nonzero(counts) < 2:
        raise ValueError(
            f"{name} contains only one class (counts={counts}). "
            "Binary stroke-vs-no-stroke training is not meaningful with this data. "
            "Provide a dataset with both class 0 and class 1 examples."
        )
    return counts


def create_dataloaders(data_dir: str,
                       batch_size: int = 8,
                       num_workers: int = 4,
                       modalities: List[str] = ['DWI'],
                       multi_modal: bool = False,
                       train_val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        data_dir: Directory containing preprocessed data
        batch_size: Batch size
        num_workers: Number of worker processes
        modalities: List of modalities to use
        multi_modal: Whether to use multi-modal dataset
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if multi_modal:
        base_dataset = MultiModalStrokeDataset(
            data_dir=data_dir,
            split='training',
            modalities=modalities,
            transform=None
        )
        patient_ids = base_dataset.patients
        train_size = int((1 - train_val_split) * len(base_dataset))
        val_size = len(base_dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(base_dataset), generator=generator).tolist()
        train_patient_ids = [patient_ids[i] for i in indices[:train_size]]
        val_patient_ids = [patient_ids[i] for i in indices[train_size:]]

        train_dataset = MultiModalStrokeDataset(
            data_dir=data_dir,
            split='training',
            modalities=modalities,
            transform=get_train_transforms(),
            patient_ids=train_patient_ids
        )
        val_dataset = MultiModalStrokeDataset(
            data_dir=data_dir,
            split='training',
            modalities=modalities,
            transform=get_val_transforms(),
            patient_ids=val_patient_ids
        )
    else:
        base_dataset = StrokeDataset(
            data_dir=data_dir,
            split='training',
            modalities=modalities,
            transform=None
        )
        patient_ids = base_dataset.patients
        train_size = int((1 - train_val_split) * len(base_dataset))
        val_size = len(base_dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(base_dataset), generator=generator).tolist()
        train_patient_ids = [patient_ids[i] for i in indices[:train_size]]
        val_patient_ids = [patient_ids[i] for i in indices[train_size:]]

        train_dataset = StrokeDataset(
            data_dir=data_dir,
            split='training',
            modalities=modalities,
            transform=get_train_transforms(),
            patient_ids=train_patient_ids
        )
        val_dataset = StrokeDataset(
            data_dir=data_dir,
            split='training',
            modalities=modalities,
            transform=get_val_transforms(),
            patient_ids=val_patient_ids
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def test_dataset():
    """Test the dataset."""
    print("Testing StrokeDataset...")
    
    # Use preprocessed data directory
    data_dir = r"d:\Stroke-DigitalTwin\outputs\preprocessed"
    
    if not Path(data_dir).exists():
        print(f"Data directory {data_dir} does not exist. Skipping test.")
        return
    
    # Create dataset
    dataset = StrokeDataset(
        data_dir=data_dir,
        split='training',
        modalities=['DWI'],
        transform=None
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Label: {sample['label']}")
        print(f"Patient ID: {sample['patient_id']}")
        if sample['mask'] is not None:
            print(f"Mask shape: {sample['mask'].shape}")
    
    print("Dataset test passed!")


if __name__ == "__main__":
    test_dataset()

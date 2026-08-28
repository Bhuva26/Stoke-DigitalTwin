"""
Module 3: Preprocessing
This module normalizes MRI, resizes images, extracts useful slices, and saves processed images.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import json
from skimage.transform import resize
from scipy.ndimage import zoom
import nibabel as nib
from .loader import ISLES2015Loader, resolve_dataset_path


class MRIPreprocessor:
    """Preprocessor for MRI data."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target size for resizing (height, width)
        """
        self.target_size = target_size
    
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize a 3D MRI volume using z-score normalization.
        
        Args:
            volume: 3D MRI volume
            
        Returns:
            Normalized volume
        """
        # Calculate mean and std
        mean = np.mean(volume)
        std = np.std(volume)
        
        # Avoid division by zero
        if std > 0:
            normalized = (volume - mean) / std
        else:
            normalized = volume - mean
        
        return normalized
    
    def normalize_slice(self, slice_data: np.ndarray) -> np.ndarray:
        """
        Normalize a 2D slice using min-max normalization.
        
        Args:
            slice_data: 2D slice data
            
        Returns:
            Normalized slice
        """
        # Clip to reasonable range
        p1, p99 = np.percentile(slice_data, (1, 99))
        slice_data = np.clip(slice_data, p1, p99)
        
        # Min-max normalization to [0, 1]
        if slice_data.max() > slice_data.min():
            normalized = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
        else:
            normalized = slice_data - slice_data.min()
        
        return normalized
    
    def resize_slice(self, slice_data: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize a 2D slice to target size.
        
        Args:
            slice_data: 2D slice data
            target_size: Target size (height, width). If None, uses self.target_size
            
        Returns:
            Resized slice
        """
        if target_size is None:
            target_size = self.target_size
        
        # Use OpenCV for resizing
        resized = cv2.resize(slice_data, target_size, interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def resize_volume(self, volume: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize a 3D volume to target size in the first two dimensions.
        
        Args:
            volume: 3D volume (H, W, D)
            target_size: Target size (height, width). If None, uses self.target_size
            
        Returns:
            Resized volume
        """
        if target_size is None:
            target_size = self.target_size
        
        # Calculate zoom factors
        zoom_factors = (target_size[0] / volume.shape[0], 
                       target_size[1] / volume.shape[1], 
                       1.0)  # Keep depth dimension unchanged
        
        # Use scipy for 3D resizing
        resized = zoom(volume, zoom_factors, order=3, mode='constant', cval=0)
        
        return resized
    
    def extract_center_slices(self, volume: np.ndarray, num_slices: int = 3) -> np.ndarray:
        """
        Extract center slices from a volume.
        
        Args:
            volume: 3D volume (H, W, D)
            num_slices: Number of slices to extract
            
        Returns:
            Extracted slices (num_slices, H, W)
        """
        depth = volume.shape[2]
        
        # Calculate indices
        if num_slices == 1:
            indices = [depth // 2]
        else:
            start = depth // 2 - num_slices // 2
            indices = list(range(start, start + num_slices))
        
        # Extract slices
        slices = np.stack([volume[:, :, i] for i in indices], axis=0)
        
        return slices
    
    def extract_slices_with_lesion(self, volume: np.ndarray, mask: np.ndarray, 
                                   num_slices: int = 3) -> np.ndarray:
        """
        Extract slices that contain the most lesion tissue.
        
        Args:
            volume: 3D MRI volume (H, W, D)
            mask: 3D lesion mask (H, W, D)
            num_slices: Number of slices to extract
            
        Returns:
            Extracted slices (num_slices, H, W)
        """
        # Calculate lesion area per slice
        lesion_areas = np.sum(mask, axis=(0, 1))
        
        # Get indices of slices with most lesion
        if np.sum(lesion_areas) == 0:
            # No lesion found, return center slices
            return self.extract_center_slices(volume, num_slices)
        
        # Get top slices with most lesion
        top_indices = np.argsort(lesion_areas)[-num_slices:]
        top_indices = np.sort(top_indices)
        
        # Extract slices
        slices = np.stack([volume[:, :, i] for i in top_indices], axis=0)
        
        return slices
    
    def preprocess_slice(self, slice_data: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline for a single slice.
        
        Args:
            slice_data: 2D slice data
            
        Returns:
            Preprocessed slice
        """
        # Normalize
        normalized = self.normalize_slice(slice_data)
        
        # Resize
        resized = self.resize_slice(normalized)
        
        # Convert to uint8 for saving
        processed = (resized * 255).astype(np.uint8)
        
        return processed
    
    def preprocess_volume_for_vit(self, volume: np.ndarray, 
                                  mask: Optional[np.ndarray] = None,
                                  method: str = 'center') -> np.ndarray:
        """
        Preprocess a volume for Vision Transformer input.
        
        Args:
            volume: 3D MRI volume
            mask: Optional lesion mask
            method: Method to extract slices ('center' or 'lesion')
            
        Returns:
            Preprocessed data ready for ViT (C, H, W) where C is number of channels
        """
        # Normalize volume
        normalized = self.normalize_volume(volume)
        
        # Extract slices
        if mask is not None and method == 'lesion':
            slices = self.extract_slices_with_lesion(normalized, mask, num_slices=3)
        else:
            slices = self.extract_center_slices(normalized, num_slices=3)
        
        # Resize each slice
        resized_slices = np.stack([self.resize_slice(s) for s in slices], axis=0)
        
        # Stack as channels (3, H, W) for RGB-like input
        # Or keep as (3, H, W) for multi-channel input
        processed = resized_slices
        
        return processed
    
    def save_processed_slice(self, slice_data: np.ndarray, save_path: Path):
        """
        Save a processed slice as an image.
        
        Args:
            slice_data: 2D processed slice data
            save_path: Path to save the image
        """
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save using OpenCV
        cv2.imwrite(str(save_path), slice_data)
    
    def save_processed_volume(self, volume: np.ndarray, save_dir: Path, prefix: str):
        """
        Save a processed volume as individual slices.
        
        Args:
            volume: 3D processed volume (num_slices, H, W)
            save_dir: Directory to save slices
            prefix: Prefix for filename
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, slice_data in enumerate(volume):
            save_path = save_dir / f"{prefix}_slice_{i:03d}.png"
            self.save_processed_slice(slice_data, save_path)
    
    def preprocess_patient(self, patient_data: Dict, output_dir: Path, 
                          method: str = 'center') -> Dict:
        """
        Preprocess all modalities for a patient.
        
        Args:
            patient_data: Patient data dictionary
            output_dir: Directory to save processed data
            method: Method to extract slices ('center' or 'lesion')
            
        Returns:
            Dictionary with paths to processed data
        """
        patient_id = patient_data['patient_id']
        patient_output_dir = output_dir / patient_id
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_info = {
            'patient_id': patient_id,
            'modalities': {},
            'mask': None
        }
        
        # Get mask data if available
        mask_data = patient_data['mask']['data'] if patient_data['mask'] else None
        
        # Process each modality
        for modality, data in patient_data['modalities'].items():
            volume = data['data']
            
            if volume is None:
                print(f"Skipping {modality} for patient {patient_id} - no data")
                continue
            
            # Preprocess volume
            processed = self.preprocess_volume_for_vit(volume, mask_data, method=method)
            
            # Save processed volume
            modality_dir = patient_output_dir / modality
            self.save_processed_volume(processed, modality_dir, f"{modality}")
            
            processed_info['modalities'][modality] = {
                'shape': processed.shape,
                'path': str(modality_dir)
            }
        
        # Process mask if available
        if mask_data is not None:
            # Extract same slices as used for MRI
            if method == 'lesion':
                mask_slices = self.extract_slices_with_lesion(mask_data, mask_data, num_slices=3)
            else:
                mask_slices = self.extract_center_slices(mask_data, num_slices=3)
            
            # Resize mask slices
            resized_mask = np.stack([self.resize_slice(s) for s in mask_slices], axis=0)
            
            # Save mask
            mask_dir = patient_output_dir / 'mask'
            self.save_processed_volume(resized_mask, mask_dir, "mask")
            
            processed_info['mask'] = {
                'shape': resized_mask.shape,
                'path': str(mask_dir)
            }
        
        return processed_info


def main():
    """Main function to test the preprocessor."""
    
    # Path configuration
    dataset_path = str(resolve_dataset_path(r"d:\Stroke-DigitalTwin\dataset"))
    output_dir = Path(r"d:\Stroke-DigitalTwin\outputs\preprocessed")
    
    # Initialize loader and preprocessor
    loader = ISLES2015Loader(dataset_path)
    preprocessor = MRIPreprocessor(target_size=(224, 224))
    
    # Get patient folders
    patient_folders = loader.get_patient_folders(split='training')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Preprocess first 5 patients
    processed_summary = []
    
    for i in range(min(5, len(patient_folders))):
        patient_folder = patient_folders[i]
        print(f"\nPreprocessing patient {patient_folder.name}...")
        
        # Load patient data
        patient_data = loader.load_patient(patient_folder, load_data=True)
        
        # Preprocess patient
        processed_info = preprocessor.preprocess_patient(patient_data, output_dir, method='lesion')
        processed_summary.append(processed_info)
        
        print(f"Preprocessed patient {patient_folder.name}")
        for modality, info in processed_info['modalities'].items():
            print(f"  {modality}: {info['shape']}")
        if processed_info['mask']:
            print(f"  mask: {processed_info['mask']['shape']}")
    
    # Save summary
    summary_path = output_dir / "preprocessing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(processed_summary, f, indent=2)
    
    print(f"\nPreprocessing complete. Summary saved to {summary_path}")
    print(f"Processed data saved to {output_dir}")


if __name__ == "__main__":
    main()

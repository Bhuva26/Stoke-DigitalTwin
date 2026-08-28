"""
Module 2: Visualization
This module visualizes MRI slices, lesion masks, and overlays.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
from loader import ISLES2015Loader


class MRIVisualizer:
    """Visualizer for MRI data and lesion masks."""
    
    def __init__(self, loader: ISLES2015Loader):
        """
        Initialize the visualizer.
        
        Args:
            loader: ISLES2015Loader instance with loaded data
        """
        self.loader = loader
    
    def normalize_slice(self, slice_data: np.ndarray) -> np.ndarray:
        """
        Normalize a slice for visualization.
        
        Args:
            slice_data: 2D array of slice data
            
        Returns:
            Normalized 2D array
        """
        # Clip to reasonable range
        p1, p99 = np.percentile(slice_data, (1, 99))
        slice_data = np.clip(slice_data, p1, p99)
        
        # Normalize to 0-255
        if slice_data.max() > slice_data.min():
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
        
        return slice_data.astype(np.uint8)
    
    def get_middle_slice(self, volume: np.ndarray, axis: int = 2) -> np.ndarray:
        """
        Get the middle slice along a given axis.
        
        Args:
            volume: 3D volume
            axis: Axis to slice along (0, 1, or 2)
            
        Returns:
            2D slice
        """
        middle_idx = volume.shape[axis] // 2
        if axis == 0:
            return volume[middle_idx, :, :]
        elif axis == 1:
            return volume[:, middle_idx, :]
        else:
            return volume[:, :, middle_idx]
    
    def get_slice_with_lesion(self, volume: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Get the slice with the maximum lesion area.
        
        Args:
            volume: 3D MRI volume
            mask: 3D lesion mask
            
        Returns:
            Tuple of (slice, slice_index)
        """
        # Find slice with maximum lesion area
        lesion_areas = np.sum(mask, axis=(0, 1))
        best_slice_idx = np.argmax(lesion_areas)
        
        return volume[:, :, best_slice_idx], best_slice_idx
    
    def plot_slice(self, slice_data: np.ndarray, title: str = "MRI Slice", 
                   cmap: str = 'gray', save_path: Optional[str] = None):
        """
        Plot a single slice.
        
        Args:
            slice_data: 2D slice data
            title: Plot title
            cmap: Colormap
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(8, 8))
        normalized = self.normalize_slice(slice_data)
        plt.imshow(normalized, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_mask(self, mask_data: np.ndarray, title: str = "Lesion Mask",
                  save_path: Optional[str] = None):
        """
        Plot a lesion mask.
        
        Args:
            mask_data: 2D mask data
            title: Plot title
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(mask_data, cmap='hot')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_overlay(self, mri_slice: np.ndarray, mask_slice: np.ndarray,
                     title: str = "MRI with Lesion Overlay", alpha: float = 0.4,
                     save_path: Optional[str] = None):
        """
        Plot MRI slice with lesion mask overlay.
        
        Args:
            mri_slice: 2D MRI slice
            mask_slice: 2D lesion mask
            title: Plot title
            alpha: Transparency of overlay
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(8, 8))
        
        # Normalize MRI slice
        normalized_mri = self.normalize_slice(mri_slice)
        
        # Display MRI
        plt.imshow(normalized_mri, cmap='gray')
        
        # Overlay mask
        if mask_slice.max() > 0:
            mask_colored = plt.cm.hot(mask_slice / mask_slice.max())
        else:
            mask_colored = plt.cm.hot(mask_slice)
        plt.imshow(mask_colored, alpha=alpha)
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_all_modalities(self, patient_data: dict, slice_idx: Optional[int] = None,
                           save_dir: Optional[str] = None):
        """
        Plot all modalities for a patient side by side.
        
        Args:
            patient_data: Patient data dictionary
            slice_idx: Specific slice index (if None, uses middle slice)
            save_dir: Directory to save plots (if None, displays)
        """
        modalities = patient_data['modalities']
        num_modalities = len(modalities)
        
        fig, axes = plt.subplots(2, num_modalities, figsize=(4*num_modalities, 8))
        if num_modalities == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (modality, data) in enumerate(modalities.items()):
            volume = data['data']
            
            # Get slice
            if slice_idx is not None:
                slice_data = volume[:, :, slice_idx]
            else:
                slice_data = self.get_middle_slice(volume)
            
            # Plot MRI
            normalized = self.normalize_slice(slice_data)
            axes[0, idx].imshow(normalized, cmap='gray')
            axes[0, idx].set_title(f'{modality} MRI')
            axes[0, idx].axis('off')
            
            # Plot with mask overlay if available
            if patient_data['mask'] is not None:
                mask_volume = patient_data['mask']['data']
                if slice_idx is not None:
                    mask_slice = mask_volume[:, :, slice_idx]
                else:
                    mask_slice = self.get_middle_slice(mask_volume)
                
                axes[1, idx].imshow(normalized, cmap='gray')
                if mask_slice.max() > 0:
                    mask_colored = plt.cm.hot(mask_slice / mask_slice.max())
                else:
                    mask_colored = plt.cm.hot(mask_slice)
                axes[1, idx].imshow(mask_colored, alpha=0.4)
                axes[1, idx].set_title(f'{modality} + Lesion')
                axes[1, idx].axis('off')
            else:
                axes[1, idx].text(0.5, 0.5, 'No mask available', 
                                  ha='center', va='center', transform=axes[1, idx].transAxes)
                axes[1, idx].axis('off')
        
        plt.suptitle(f"Patient {patient_data['patient_id']} - All Modalities")
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f"patient_{patient_data['patient_id']}_all_modalities.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_slice_series(self, volume: np.ndarray, mask: Optional[np.ndarray] = None,
                          num_slices: int = 9, title: str = "Slice Series",
                          save_path: Optional[str] = None):
        """
        Plot a series of slices through the volume.
        
        Args:
            volume: 3D MRI volume
            mask: Optional 3D lesion mask
            num_slices: Number of slices to display
            title: Plot title
            save_path: Optional path to save the figure
        """
        # Select evenly spaced slices
        total_slices = volume.shape[2]
        slice_indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
        
        fig, axes = plt.subplots(3, num_slices, figsize=(2*num_slices, 6))
        
        for idx, slice_idx in enumerate(slice_indices):
            slice_data = volume[:, :, slice_idx]
            normalized = self.normalize_slice(slice_data)
            
            # Top row: MRI only
            axes[0, idx].imshow(normalized, cmap='gray')
            axes[0, idx].set_title(f'Slice {slice_idx}')
            axes[0, idx].axis('off')
            
            # Middle row: Mask if available
            if mask is not None:
                mask_slice = mask[:, :, slice_idx]
                axes[1, idx].imshow(mask_slice, cmap='hot')
                axes[1, idx].axis('off')
                
                # Bottom row: Overlay
                axes[2, idx].imshow(normalized, cmap='gray')
                if mask_slice.max() > 0:
                    mask_colored = plt.cm.hot(mask_slice / mask_slice.max())
                else:
                    mask_colored = plt.cm.hot(mask_slice)
                axes[2, idx].imshow(mask_colored, alpha=0.4)
                axes[2, idx].axis('off')
            else:
                axes[1, idx].axis('off')
                axes[2, idx].axis('off')
        
        if mask is not None:
            axes[1, 0].set_ylabel('Mask', rotation=0, labelpad=20)
            axes[2, 0].set_ylabel('Overlay', rotation=0, labelpad=20)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_patient(self, patient_data: dict, output_dir: Optional[str] = None):
        """
        Create comprehensive visualizations for a single patient.
        
        Args:
            patient_data: Patient data dictionary
            output_dir: Directory to save visualizations
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Get modality data
        modalities = patient_data['modalities']
        
        # Use DWI as primary modality for detailed views
        if 'DWI' in modalities:
            primary_modality = 'DWI'
        else:
            primary_modality = list(modalities.keys())[0]
        
        volume = modalities[primary_modality]['data']
        mask = patient_data['mask']['data'] if patient_data['mask'] else None
        
        # Get slice with maximum lesion
        if mask is not None:
            best_slice, slice_idx = self.get_slice_with_lesion(volume, mask)
            mask_slice = mask[:, :, slice_idx]
        else:
            best_slice = self.get_middle_slice(volume)
            slice_idx = volume.shape[2] // 2
            mask_slice = None
        
        # Plot individual visualizations
        if output_dir:
            self.plot_slice(best_slice, 
                          title=f"Patient {patient_data['patient_id']} - {primary_modality} (Slice {slice_idx})",
                          save_path=output_dir / f"patient_{patient_data['patient_id']}_slice.png")
            
            if mask_slice is not None:
                self.plot_mask(mask_slice,
                             title=f"Patient {patient_data['patient_id']} - Lesion Mask (Slice {slice_idx})",
                             save_path=output_dir / f"patient_{patient_data['patient_id']}_mask.png")
                
                self.plot_overlay(best_slice, mask_slice,
                                title=f"Patient {patient_data['patient_id']} - Overlay (Slice {slice_idx})",
                                save_path=output_dir / f"patient_{patient_data['patient_id']}_overlay.png")
            
            self.plot_all_modalities(patient_data, slice_idx=slice_idx, save_dir=output_dir)
            
            self.plot_slice_series(volume, mask, num_slices=9,
                                  title=f"Patient {patient_data['patient_id']} - Slice Series",
                                  save_path=output_dir / f"patient_{patient_data['patient_id']}_series.png")
        else:
            self.plot_slice(best_slice, 
                          title=f"Patient {patient_data['patient_id']} - {primary_modality} (Slice {slice_idx})")
            
            if mask_slice is not None:
                self.plot_mask(mask_slice,
                             title=f"Patient {patient_data['patient_id']} - Lesion Mask (Slice {slice_idx})")
                
                self.plot_overlay(best_slice, mask_slice,
                                title=f"Patient {patient_data['patient_id']} - Overlay (Slice {slice_idx})")
            
            self.plot_all_modalities(patient_data, slice_idx=slice_idx)
            
            self.plot_slice_series(volume, mask, num_slices=9,
                                  title=f"Patient {patient_data['patient_id']} - Slice Series")


def main():
    """Main function to test the visualizer."""
    
    # Path configuration
    dataset_path = str(resolve_dataset_path(r"d:\Stroke-DigitalTwin\dataset"))
    output_dir = r"d:\Stroke-DigitalTwin\outputs\visualizations"
    
    # Initialize loader
    loader = ISLES2015Loader(dataset_path)
    
    # Get patient folders
    patient_folders = loader.get_patient_folders(split='training')
    
    # Initialize visualizer
    visualizer = MRIVisualizer(loader)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Visualize first 3 patients (loading data individually)
    for i in range(min(3, len(patient_folders))):
        patient_folder = patient_folders[i]
        print(f"\nVisualizing patient {patient_folder.name}...")
        
        # Load only this patient's data
        patient_data = loader.load_patient(patient_folder, load_data=True)
        
        patient_output_dir = Path(output_dir) / f"patient_{patient_data['patient_id']}"
        visualizer.visualize_patient(patient_data, output_dir=str(patient_output_dir))
    
    print(f"\nVisualizations saved to {output_dir}")


if __name__ == "__main__":
    main()

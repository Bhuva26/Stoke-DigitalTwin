"""
Module: Temporal Analysis (Follow-up MRI Comparison)
This module implements temporal analysis for comparing follow-up MRIs with baseline scans.
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import matplotlib.pyplot as plt
from skimage import measure
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift


class TemporalAnalyzer:
    """Analyzer for temporal changes in MRI scans."""
    
    def __init__(self, baseline_dir: Path, followup_dir: Path):
        """
        Initialize temporal analyzer.
        
        Args:
            baseline_dir: Directory with baseline/preprocessed data
            followup_dir: Directory with follow-up/preprocessed data
        """
        self.baseline_dir = Path(baseline_dir)
        self.followup_dir = Path(followup_dir)
    
    def load_patient_slices(self, patient_dir: Path, modality: str = 'DWI') -> np.ndarray:
        """
        Load slices for a patient and modality.
        
        Args:
            patient_dir: Patient directory
            modality: Modality name
            
        Returns:
            Stacked slices as numpy array (3, H, W)
        """
        modality_dir = patient_dir / modality
        if not modality_dir.exists():
            raise ValueError(f"Modality directory {modality_dir} does not exist")
        
        slice_files = sorted(modality_dir.glob("*.png"))
        slices = []
        for slice_file in slice_files[:3]:
            img = cv2.imread(str(slice_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                slices.append(img)
        
        if len(slices) == 0:
            raise ValueError(f"No slices found for {modality}")
        
        stacked = np.stack(slices, axis=0)
        return stacked
    
    def register_images(self, moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        """
        Register moving image to fixed image using phase correlation.
        
        Args:
            moving: Moving image (H, W)
            fixed: Fixed image (H, W)
            
        Returns:
            Registered moving image
        """
        # Compute phase correlation
        shift_result, error, diffphase = phase_cross_correlation(
            fixed, moving, normalization=None
        )
        
        # Apply shift
        registered = shift(moving, shift_result, mode='constant', cval=0)
        
        return registered
    
    def compute_lesion_change(self, baseline_mask: np.ndarray, 
                            followup_mask: np.ndarray) -> Dict:
        """
        Compute lesion change metrics between baseline and follow-up.
        
        Args:
            baseline_mask: Baseline lesion mask (H, W)
            followup_mask: Follow-up lesion mask (H, W)
            
        Returns:
            Dictionary with change metrics
        """
        # Compute lesion areas
        baseline_area = np.sum(baseline_mask > 0)
        followup_area = np.sum(followup_mask > 0)
        
        # Compute change
        area_change = followup_area - baseline_area
        percent_change = (area_change / baseline_area * 100) if baseline_area > 0 else 0
        
        # Compute overlap (intersection)
        overlap = np.sum((baseline_mask > 0) & (followup_mask > 0))
        
        # Compute new lesion area (in follow-up but not baseline)
        new_lesion = np.sum((baseline_mask == 0) & (followup_mask > 0))
        
        # Compute resolved lesion area (in baseline but not follow-up)
        resolved_lesion = np.sum((baseline_mask > 0) & (followup_mask == 0))
        
        # Dice coefficient
        dice = 2 * overlap / (baseline_area + followup_area) if (baseline_area + followup_area) > 0 else 0
        
        return {
            'baseline_area': int(baseline_area),
            'followup_area': int(followup_area),
            'area_change': int(area_change),
            'percent_change': float(percent_change),
            'overlap': int(overlap),
            'new_lesion': int(new_lesion),
            'resolved_lesion': int(resolved_lesion),
            'dice_coefficient': float(dice)
        }
    
    def analyze_temporal_change(self, patient_id: str, 
                               modality: str = 'DWI') -> Dict:
        """
        Analyze temporal changes for a patient.
        
        Args:
            patient_id: Patient identifier
            modality: Modality to analyze
            
        Returns:
            Dictionary with temporal analysis results
        """
        baseline_patient_dir = self.baseline_dir / patient_id
        followup_patient_dir = self.followup_dir / patient_id
        
        if not baseline_patient_dir.exists():
            raise ValueError(f"Baseline patient directory {baseline_patient_dir} does not exist")
        
        if not followup_patient_dir.exists():
            raise ValueError(f"Follow-up patient directory {followup_patient_dir} does not exist")
        
        results = {
            'patient_id': patient_id,
            'modality': modality,
            'slices': []
        }
        
        # Load baseline and follow-up slices
        try:
            baseline_slices = self.load_patient_slices(baseline_patient_dir, modality)
            followup_slices = self.load_patient_slices(followup_patient_dir, modality)
        except ValueError as e:
            results['error'] = str(e)
            return results
        
        # Load masks
        baseline_mask_dir = baseline_patient_dir / 'mask'
        followup_mask_dir = followup_patient_dir / 'mask'
        
        baseline_masks = None
        followup_masks = None
        
        if baseline_mask_dir.exists():
            mask_files = sorted(baseline_mask_dir.glob("*.png"))
            masks = []
            for mask_file in mask_files[:3]:
                img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    masks.append(img)
            if masks:
                baseline_masks = np.stack(masks, axis=0)
        
        if followup_mask_dir.exists():
            mask_files = sorted(followup_mask_dir.glob("*.png"))
            masks = []
            for mask_file in mask_files[:3]:
                img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    masks.append(img)
            if masks:
                followup_masks = np.stack(masks, axis=0)
        
        # Analyze each slice
        for i in range(min(baseline_slices.shape[0], followup_slices.shape[0])):
            baseline_slice = baseline_slices[i]
            followup_slice = followup_slices[i]
            
            # Register follow-up to baseline
            registered_followup = self.register_images(followup_slice, baseline_slice)
            
            slice_result = {
                'slice_index': i,
                'intensity_change': {
                    'mean_baseline': float(np.mean(baseline_slice)),
                    'mean_followup': float(np.mean(registered_followup)),
                    'mean_diff': float(np.mean(registered_followup) - np.mean(baseline_slice)),
                    'std_baseline': float(np.std(baseline_slice)),
                    'std_followup': float(np.std(registered_followup))
                }
            }
            
            # Analyze lesion change if masks available
            if baseline_masks is not None and followup_masks is not None:
                baseline_mask = baseline_masks[i]
                followup_mask = followup_masks[i]
                
                # Register follow-up mask
                registered_mask = self.register_images(followup_mask.astype(float), baseline_mask.astype(float))
                registered_mask = (registered_mask > 127).astype(np.uint8)
                
                lesion_change = self.compute_lesion_change(baseline_mask, registered_mask)
                slice_result['lesion_change'] = lesion_change
            
            results['slices'].append(slice_result)
        
        # Compute overall summary
        if results['slices']:
            total_baseline_area = sum(
                s.get('lesion_change', {}).get('baseline_area', 0) 
                for s in results['slices']
            )
            total_followup_area = sum(
                s.get('lesion_change', {}).get('followup_area', 0) 
                for s in results['slices']
            )
            
            results['summary'] = {
                'total_baseline_area': total_baseline_area,
                'total_followup_area': total_followup_area,
                'total_area_change': total_followup_area - total_baseline_area,
                'overall_percent_change': (
                    (total_followup_area - total_baseline_area) / total_baseline_area * 100 
                    if total_baseline_area > 0 else 0
                )
            }
        
        return results
    
    def visualize_temporal_change(self, patient_id: str, 
                                 modality: str = 'DWI',
                                 save_path: Optional[str] = None):
        """
        Visualize temporal changes for a patient.
        
        Args:
            patient_id: Patient identifier
            modality: Modality to visualize
            save_path: Optional path to save visualization
        """
        results = self.analyze_temporal_change(patient_id, modality)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        num_slices = len(results['slices'])
        if num_slices == 0:
            print("No slices to visualize")
            return
        
        fig, axes = plt.subplots(num_slices, 4, figsize=(20, 5 * num_slices))
        
        if num_slices == 1:
            axes = axes.reshape(1, -1)
        
        baseline_patient_dir = self.baseline_dir / patient_id
        followup_patient_dir = self.followup_dir / patient_id
        
        baseline_slices = self.load_patient_slices(baseline_patient_dir, modality)
        followup_slices = self.load_patient_slices(followup_patient_dir, modality)
        
        for i, slice_result in enumerate(results['slices']):
            baseline_slice = baseline_slices[i]
            followup_slice = followup_slices[i]
            registered_followup = self.register_images(followup_slice, baseline_slice)
            
            # Baseline
            axes[i, 0].imshow(baseline_slice, cmap='gray')
            axes[i, 0].set_title(f'Baseline Slice {i}')
            axes[i, 0].axis('off')
            
            # Follow-up (registered)
            axes[i, 1].imshow(registered_followup, cmap='gray')
            axes[i, 1].set_title(f'Follow-up Slice {i} (Registered)')
            axes[i, 1].axis('off')
            
            # Difference
            diff = registered_followup.astype(float) - baseline_slice.astype(float)
            axes[i, 2].imshow(diff, cmap='RdBu', vmin=-50, vmax=50)
            axes[i, 2].set_title(f'Difference Slice {i}')
            axes[i, 2].axis('off')
            
            # Summary text
            summary_text = f"Intensity Change:\n"
            summary_text += f"Mean Diff: {slice_result['intensity_change']['mean_diff']:.2f}\n"
            
            if 'lesion_change' in slice_result:
                lc = slice_result['lesion_change']
                summary_text += f"\nLesion Change:\n"
                summary_text += f"Baseline: {lc['baseline_area']} px\n"
                summary_text += f"Follow-up: {lc['followup_area']} px\n"
                summary_text += f"Change: {lc['percent_change']:.1f}%\n"
                summary_text += f"Dice: {lc['dice_coefficient']:.3f}"
            
            axes[i, 3].text(0.1, 0.5, summary_text, 
                           fontsize=10, verticalalignment='center',
                           transform=axes[i, 3].transAxes)
            axes[i, 3].axis('off')
        
        plt.suptitle(f'Temporal Analysis: Patient {patient_id} ({modality})', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return results


class TemporalChangePredictor(nn.Module):
    """Neural network for predicting temporal changes."""
    
    def __init__(self, input_dim: int = 12, hidden_dims: List[int] = [64, 32]):
        """
        Initialize temporal change predictor.
        
        Args:
            input_dim: Input feature dimension (baseline features + follow-up features)
            hidden_dims: List of hidden layer dimensions
        """
        super(TemporalChangePredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output: predict lesion change percentage
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(1)


def main():
    """Main function to test temporal analysis."""
    
    # Configuration
    config = {
        'baseline_dir': r"d:\Stroke-DigitalTwin\outputs\preprocessed_full\training",
        'followup_dir': r"d:\Stroke-DigitalTwin\outputs\preprocessed_full\testing",  # Using test as follow-up for demo
        'output_dir': r"d:\Stroke-DigitalTwin\outputs\temporal_analysis",
        'patient_id': '1',
        'modality': 'DWI'
    }
    
    print("Temporal Analysis Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create analyzer
    analyzer = TemporalAnalyzer(
        baseline_dir=config['baseline_dir'],
        followup_dir=config['followup_dir']
    )
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze temporal change
    print(f"\nAnalyzing temporal change for patient {config['patient_id']}...")
    results = analyzer.analyze_temporal_change(
        patient_id=config['patient_id'],
        modality=config['modality']
    )
    
    # Save results
    result_path = output_dir / f"temporal_analysis_{config['patient_id']}.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {result_path}")
    
    # Print summary
    if 'summary' in results:
        print(f"\nSummary:")
        print(f"  Total baseline area: {results['summary']['total_baseline_area']} px")
        print(f"  Total follow-up area: {results['summary']['total_followup_area']} px")
        print(f"  Total area change: {results['summary']['total_area_change']} px")
        print(f"  Overall percent change: {results['summary']['overall_percent_change']:.1f}%")
    
    # Visualize
    print(f"\nGenerating visualization...")
    viz_path = output_dir / f"temporal_viz_{config['patient_id']}.png"
    analyzer.visualize_temporal_change(
        patient_id=config['patient_id'],
        modality=config['modality'],
        save_path=str(viz_path)
    )
    
    print("\nTemporal analysis completed!")


if __name__ == "__main__":
    main()

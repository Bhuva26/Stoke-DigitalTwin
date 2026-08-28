"""
Evaluation module for lesion segmentation with comprehensive metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import json
import matplotlib.pyplot as plt
import cv2
from typing import Dict, List, Tuple
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.segmentation_model import LesionSegmentationUNet
from models.segmentation_dataset import SegmentationStrokeDataset, get_val_transforms


class SegmentationEvaluator:
    """Comprehensive evaluator for lesion segmentation."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize evaluator.
        
        Args:
            model: Trained segmentation model
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """
        Comprehensive evaluation of segmentation model.
        
        Args:
            dataloader: Test dataloader
            
        Returns:
            Dictionary of evaluation metrics
        """
        all_predictions = []
        all_masks = []
        all_patient_ids = []
        all_slice_indices = []
        
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            patient_ids = batch['patient_id']
            slice_indices = batch['slice_idx']
            
            # Get predictions
            outputs = self.model(images)
            predictions = torch.sigmoid(outputs)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_masks.extend(masks.cpu().numpy())
            all_patient_ids.extend(patient_ids)
            all_slice_indices.extend(slice_indices)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_masks = np.array(all_masks)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_masks)
        
        # Per-slice results
        metrics['per_slice_results'] = []
        for i, (patient_id, slice_idx) in enumerate(zip(all_patient_ids, all_slice_indices)):
            pred = all_predictions[i]
            mask = all_masks[i]
            slice_metrics = self._calculate_single_slice_metrics(pred, mask)
            metrics['per_slice_results'].append({
                'patient_id': patient_id,
                'slice_idx': int(slice_idx),
                'iou': slice_metrics['iou'],
                'dice': slice_metrics['dice'],
                'precision': slice_metrics['precision'],
                'recall': slice_metrics['recall']
            })
        
        return metrics
    
    def _calculate_metrics(self, predictions: np.ndarray, masks: np.ndarray) -> Dict:
        """Calculate overall metrics."""
        # Binary predictions with threshold
        binary_predictions = (predictions > 0.5).astype(np.float32)
        binary_masks = masks.astype(np.float32)
        
        # Flatten arrays
        pred_flat = binary_predictions.flatten()
        mask_flat = binary_masks.flatten()
        
        # True positives, false positives, false negatives
        tp = np.sum((pred_flat == 1) & (mask_flat == 1))
        fp = np.sum((pred_flat == 1) & (mask_flat == 0))
        fn = np.sum((pred_flat == 0) & (mask_flat == 1))
        tn = np.sum((pred_flat == 0) & (mask_flat == 0))
        
        # Basic metrics
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)
        
        # Dice coefficient
        dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
        
        # IoU
        iou = tp / (tp + fp + fn + 1e-6)
        
        # Hausdorff distance (simplified)
        hausdorff = self._calculate_hausdorff_distance(binary_predictions, binary_masks)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'dice': float(dice),
            'iou': float(iou),
            'hausdorff_distance': float(hausdorff),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn)
        }
    
    def _calculate_single_slice_metrics(self, prediction: np.ndarray, mask: np.ndarray) -> Dict:
        """Calculate metrics for a single slice."""
        binary_pred = (prediction > 0.5).astype(np.float32)
        binary_mask = mask.astype(np.float32)
        
        pred_flat = binary_pred.flatten()
        mask_flat = binary_mask.flatten()
        
        tp = np.sum((pred_flat == 1) & (mask_flat == 1))
        fp = np.sum((pred_flat == 1) & (mask_flat == 0))
        fn = np.sum((pred_flat == 0) & (mask_flat == 1))
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'dice': float(dice),
            'iou': float(iou)
        }
    
    def _calculate_hausdorff_distance(self, pred: np.ndarray, mask: np.ndarray) -> float:
        """Calculate Hausdorff distance (simplified version)."""
        # Get contours
        pred_contours = self._get_contours(pred[0])
        mask_contours = self._get_contours(mask[0])
        
        if len(pred_contours) == 0 or len(mask_contours) == 0:
            return 0.0
        
        # Calculate distance between contours
        from scipy.spatial.distance import directed_hausdorff
        try:
            distance = max(
                directed_hausdorff(pred_contours, mask_contours)[0],
                directed_hausdorff(mask_contours, pred_contours)[0]
            )
            return float(distance)
        except:
            return 0.0
    
    def _get_contours(self, binary_mask: np.ndarray) -> np.ndarray:
        """Extract contour points from binary mask."""
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return np.array([])
        
        # Combine all contours
        all_points = np.vstack([c.reshape(-1, 2) for c in contours])
        return all_points
    
    def print_evaluation_report(self, metrics: Dict):
        """Print detailed evaluation report."""
        print("\n" + "="*60)
        print("SEGMENTATION EVALUATION REPORT")
        print("="*60)
        
        print("\nOverall Performance Metrics:")
        print(f"  Dice Coefficient: {metrics['dice']:.4f}")
        print(f"  IoU (Jaccard Index): {metrics['iou']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Hausdorff Distance: {metrics['hausdorff_distance']:.2f} pixels")
        
        print("\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        
        print("\nPer-Slice Statistics:")
        dice_scores = [r['dice'] for r in metrics['per_slice_results']]
        iou_scores = [r['iou'] for r in metrics['per_slice_results']]
        
        print(f"  Dice - Mean: {np.mean(dice_scores):.4f}, Std: {np.std(dice_scores):.4f}")
        print(f"  IoU - Mean: {np.mean(iou_scores):.4f}, Std: {np.std(iou_scores):.4f}")
        
        print("="*60 + "\n")
    
    def visualize_predictions(self, dataloader: DataLoader, save_dir: str, num_samples: int = 5):
        """
        Visualize model predictions.
        
        Args:
            dataloader: Test dataloader
            save_dir: Directory to save visualizations
            num_samples: Number of samples to visualize
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        samples_visualized = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if samples_visualized >= num_samples:
                    break
                
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                patient_ids = batch['patient_id']
                
                # Get predictions
                outputs = self.model(images)
                predictions = torch.sigmoid(outputs)
                
                # Visualize each sample in batch
                for i in range(min(len(images), num_samples - samples_visualized)):
                    image = images[i].cpu().numpy()[0]
                    mask = masks[i].cpu().numpy()[0]
                    pred = predictions[i].cpu().numpy()[0]
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Original image
                    axes[0].imshow(image, cmap='gray')
                    axes[0].set_title('Original Image')
                    axes[0].axis('off')
                    
                    # Ground truth mask
                    axes[1].imshow(mask, cmap='jet')
                    axes[1].set_title('Ground Truth Mask')
                    axes[1].axis('off')
                    
                    # Predicted mask
                    axes[2].imshow(pred, cmap='jet')
                    axes[2].set_title('Predicted Mask')
                    axes[2].axis('off')
                    
                    plt.suptitle(f'Patient {patient_ids[i]} - Slice {i}')
                    
                    save_path = save_dir / f'prediction_{samples_visualized + 1}.png'
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    samples_visualized += 1
        
        print(f"Saved {samples_visualized} prediction visualizations to {save_dir}")
    
    def save_evaluation_results(self, metrics: Dict, save_path: str):
        """Save evaluation results to JSON."""
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Evaluation results saved to {save_path}")


def main():
    """Main evaluation function."""
    
    # Configuration
    config = {
        'model_path': r"d:\Stroke-DigitalTwin\outputs\segmentation_checkpoints\best_segmentation_model.pth",
        'data_dir': r"d:\Stroke-DigitalTwin\outputs\preprocessed",
        'output_dir': r"d:\Stroke-DigitalTwin\outputs\segmentation_evaluation",
        'batch_size': 8,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Segmentation Evaluation Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load model
    print(f"\nLoading model from {config['model_path']}...")
    model = LesionSegmentationUNet(in_channels=1, out_channels=1, bilinear=False)
    
    checkpoint = torch.load(config['model_path'], map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model loaded successfully")
    if 'val_dice' in checkpoint:
        print(f"Previous validation Dice: {checkpoint['val_dice']:.4f}")
    
    # Create test dataset
    print("\nCreating test dataset...")
    from models.segmentation_dataset import create_segmentation_dataloaders
    
    # Use validation split as test set for evaluation
    _, test_loader = create_segmentation_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=0,
        use_all_slices=False
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create evaluator
    evaluator = SegmentationEvaluator(model, config['device'])
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluator.evaluate(test_loader)
    
    # Print report
    evaluator.print_evaluation_report(metrics)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_path = output_dir / "segmentation_evaluation_results.json"
    evaluator.save_evaluation_results(metrics, results_path)
    
    # Visualize predictions
    print("\nGenerating prediction visualizations...")
    evaluator.visualize_predictions(test_loader, str(output_dir), num_samples=5)
    
    print(f"\nEvaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

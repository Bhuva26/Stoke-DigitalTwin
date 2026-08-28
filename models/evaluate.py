"""
Model evaluation with proper metrics for publication.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, auc
)
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency
    sns = None
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.vit import StrokeViT
from models.dataset import StrokeDataset, get_val_transforms


class ModelEvaluator:
    """Comprehensive model evaluation for publication."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """
        Comprehensive evaluation of model performance.
        
        Args:
            dataloader: Test dataloader
            
        Returns:
            Dictionary of evaluation metrics
        """
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_patient_ids = []
        
        for batch in dataloader:
            if batch is None:
                continue
            if 'image' not in batch or 'label' not in batch or 'patient_id' not in batch:
                continue
            # Move data to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            patient_ids = batch['patient_id']
            
            # Get predictions
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_patient_ids.extend(patient_ids)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = {}
        unique_labels = np.unique(all_labels)
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(all_labels, all_predictions)
        if len(unique_labels) > 1:
            metrics['precision'] = precision_score(all_labels, all_predictions, zero_division=0)
            metrics['recall'] = recall_score(all_labels, all_predictions, zero_division=0)
            metrics['f1_score'] = f1_score(all_labels, all_predictions, zero_division=0)
        else:
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1_score'] = 0.0
        metrics['specificity'] = self._calculate_specificity(all_labels, all_predictions)
        
        # AUC-ROC
        if len(unique_labels) > 1:
            metrics['auc_roc'] = roc_auc_score(all_labels, all_probabilities[:, 1])
        else:
            metrics['auc_roc'] = None
        
        # AUC-PR
        if len(unique_labels) > 1:
            precision, recall, _ = precision_recall_curve(all_labels, all_probabilities[:, 1])
            metrics['auc_pr'] = auc(recall, precision)
        else:
            metrics['auc_pr'] = None
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-patient results
        metrics['per_patient_results'] = []
        for i, patient_id in enumerate(all_patient_ids):
            metrics['per_patient_results'].append({
                'patient_id': patient_id,
                'true_label': int(all_labels[i]),
                'predicted_label': int(all_predictions[i]),
                'confidence': float(all_probabilities[i, all_predictions[i]]),
                'stroke_probability': float(all_probabilities[i, 1])
            })
        
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.0
    
    def print_evaluation_report(self, metrics: Dict):
        """Print detailed evaluation report."""
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        print("\nOverall Performance Metrics:")
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  Precision:     {metrics['precision']:.4f}")
        print(f"  Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"  Specificity:   {metrics['specificity']:.4f}")
        print(f"  F1-Score:      {metrics['f1_score']:.4f}")
        
        if metrics['auc_roc'] is not None:
            print(f"  AUC-ROC:       {metrics['auc_roc']:.4f}")
        if metrics['auc_pr'] is not None:
            print(f"  AUC-PR:        {metrics['auc_pr']:.4f}")
        
        print("\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print("  Predicted\\Actual")
        print("    No Stroke  Stroke")
        print(f"No Stroke  {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"Stroke      {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        print("\nDetailed Classification Report:")
        target_names = ['No Stroke', 'Stroke']
        # Reconstruct predictions and labels for classification report
        y_true = [r['true_label'] for r in metrics['per_patient_results']]
        y_pred = [r['predicted_label'] for r in metrics['per_patient_results']]
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
        
        print("="*60 + "\n")
    
    def plot_confusion_matrix(self, metrics: Dict, save_path: str = None):
        """Plot confusion matrix."""
        cm = np.array(metrics['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        if sns is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['No Stroke', 'Stroke'],
                        yticklabels=['No Stroke', 'Stroke'])
        else:
            plt.imshow(cm, cmap='Blues')
            plt.xticks([0, 1], ['No Stroke', 'Stroke'])
            plt.yticks([0, 1], ['No Stroke', 'Stroke'])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(self, metrics: Dict, save_path: str = None):
        """Plot ROC curve."""
        if metrics['auc_roc'] is None:
            print("Cannot plot ROC curve: only one class present")
            return
        
        # Get probabilities and labels
        stroke_probs = [r['stroke_probability'] for r in metrics['per_patient_results']]
        true_labels = [r['true_label'] for r in metrics['per_patient_results']]
        
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(true_labels, stroke_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {metrics["auc_roc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_precision_recall_curve(self, metrics: Dict, save_path: str = None):
        """Plot precision-recall curve."""
        if metrics['auc_pr'] is None:
            print("Cannot plot PR curve: only one class present")
            return
        
        # Get probabilities and labels
        stroke_probs = [r['stroke_probability'] for r in metrics['per_patient_results']]
        true_labels = [r['true_label'] for r in metrics['per_patient_results']]
        
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(true_labels, stroke_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {metrics["auc_pr"]:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_evaluation_results(self, metrics: Dict, save_path: str):
        """Save evaluation results to JSON."""
        # Convert numpy types to Python types
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                metrics_serializable[key] = float(value)
            else:
                metrics_serializable[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"Evaluation results saved to {save_path}")


def main():
    """Main evaluation function."""
    
    # Configuration
    config = {
        'model_path': r"d:\Stroke-DigitalTwin\outputs\checkpoints\best_model.pth",
        'data_dir': r"d:\Stroke-DigitalTwin\outputs\preprocessed",
        'output_dir': r"d:\Stroke-DigitalTwin\outputs\evaluation",
        'batch_size': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_name': 'vit_base_patch16_224'
    }
    
    print("Evaluation Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load model
    print(f"\nLoading model from {config['model_path']}...")
    model = StrokeViT(
        model_name=config['model_name'],
        num_classes=2,
        pretrained=False,
        in_channels=3
    )
    
    checkpoint = torch.load(config['model_path'], map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model loaded successfully")
    if 'val_acc' in checkpoint:
        print(f"Previous validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Create test dataset (using validation split for now)
    print("\nCreating test dataset...")
    from models.dataset import create_dataloaders
    
    # Use validation split as test set
    _, test_loader = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=0,
        modalities=['DWI'],
        multi_modal=False
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, config['device'])
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluator.evaluate(test_loader)
    
    # Print report
    evaluator.print_evaluation_report(metrics)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    evaluator.save_evaluation_results(metrics, results_path)
    
    # Plot visualizations
    cm_path = output_dir / "confusion_matrix.png"
    evaluator.plot_confusion_matrix(metrics, save_path=str(cm_path))
    
    roc_path = output_dir / "roc_curve.png"
    evaluator.plot_roc_curve(metrics, save_path=str(roc_path))
    
    pr_path = output_dir / "precision_recall_curve.png"
    evaluator.plot_precision_recall_curve(metrics, save_path=str(pr_path))
    
    print(f"\nEvaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

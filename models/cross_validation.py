"""
Cross-validation training for robust model evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.simple_model import SimpleStrokeCNN
from models.dataset import StrokeDataset, get_train_transforms, get_val_transforms
from models.evaluate import ModelEvaluator


class CrossValidationTrainer:
    """Trainer with k-fold cross-validation."""
    
    def __init__(self,
                 model_class,
                 model_params: Dict,
                 data_dir: str,
                 n_folds: int = 5,
                 batch_size: int = 4,
                 num_epochs: int = 30,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-3,
                 device: str = 'cuda'):
        """
        Initialize cross-validation trainer.
        
        Args:
            model_class: Model class to instantiate
            model_params: Parameters for model initialization
            data_dir: Directory with preprocessed data
            n_folds: Number of folds for cross-validation
            batch_size: Batch size
            num_epochs: Number of epochs per fold
            learning_rate: Learning rate
            weight_decay: Weight decay
            device: Device to train on
        """
        self.model_class = model_class
        self.model_params = model_params
        self.data_dir = Path(data_dir)
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load full dataset
        self.full_dataset = StrokeDataset(
            data_dir=data_dir,
            split='training',
            modalities=['DWI'],
            transform=None  # Will apply transforms per fold
        )
        
        # Get labels for stratified splitting
        self.labels = [self.full_dataset[i]['label'].item() for i in range(len(self.full_dataset))]
        
        print(f"Full dataset size: {len(self.full_dataset)}")
        print(f"Label distribution: {np.bincount(self.labels)}")
        print(f"Device: {self.device}")
    
    def train_fold(self, train_indices: List[int], val_indices: List[int], fold: int) -> Dict:
        """
        Train a single fold.
        
        Args:
            train_indices: Training indices
            val_indices: Validation indices
            fold: Fold number
            
        Returns:
            Dictionary with fold results
        """
        print(f"\n{'='*60}")
        print(f"Training Fold {fold + 1}/{self.n_folds}")
        print(f"{'='*60}")
        print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
        
        # Create subsets
        train_dataset = Subset(self.full_dataset, train_indices)
        val_dataset = Subset(self.full_dataset, val_indices)
        
        # Apply transforms
        train_dataset.dataset.transform = get_train_transforms()
        val_dataset.dataset.transform = get_val_transforms()
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Create model
        model = self.model_class(**self.model_params).to(self.device)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs,
            eta_min=1e-6
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0.0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(self.num_epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Update scheduler
            scheduler.step()
            
            # Record metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = deepcopy(model.state_dict())
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Load best model for final evaluation
        model.load_state_dict(best_model_state)
        
        # Final evaluation
        evaluator = ModelEvaluator(model, self.device)
        final_metrics = evaluator.evaluate(val_loader)
        
        fold_results = {
            'fold': fold,
            'best_val_acc': best_val_acc,
            'final_metrics': final_metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
        
        print(f"Fold {fold + 1} completed. Best Val Acc: {best_val_acc:.2f}%")
        
        return fold_results
    
    def run_cross_validation(self) -> Dict:
        """
        Run k-fold cross-validation.
        
        Returns:
            Dictionary with all fold results
        """
        print(f"\nStarting {self.n_folds}-fold cross-validation...")
        
        # Stratified k-fold for balanced class distribution
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        all_fold_results = []
        
        for fold, (train_indices, val_indices) in enumerate(skf.split(range(len(self.full_dataset)), self.labels)):
            fold_results = self.train_fold(train_indices, val_indices, fold)
            all_fold_results.append(fold_results)
        
        # Aggregate results
        cv_results = self._aggregate_results(all_fold_results)
        
        return cv_results
    
    def _aggregate_results(self, fold_results: List[Dict]) -> Dict:
        """
        Aggregate results across folds.
        
        Args:
            fold_results: List of fold result dictionaries
            
        Returns:
            Aggregated results
        """
        # Extract metrics
        val_accs = [r['best_val_acc'] for r in fold_results]
        accuracies = [r['final_metrics']['accuracy'] for r in fold_results]
        precisions = [r['final_metrics']['precision'] for r in fold_results]
        recalls = [r['final_metrics']['recall'] for r in fold_results]
        f1_scores = [r['final_metrics']['f1_score'] for r in fold_results]
        specificities = [r['final_metrics']['specificity'] for r in fold_results]
        
        # Calculate statistics
        aggregated = {
            'n_folds': self.n_folds,
            'val_accuracy_mean': np.mean(val_accs),
            'val_accuracy_std': np.std(val_accs),
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'precision_mean': np.mean(precisions),
            'precision_std': np.std(precisions),
            'recall_mean': np.mean(recalls),
            'recall_std': np.std(recalls),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'specificity_mean': np.mean(specificities),
            'specificity_std': np.std(specificities),
            'fold_results': fold_results
        }
        
        return aggregated
    
    def print_cv_results(self, results: Dict):
        """Print cross-validation results."""
        print("\n" + "="*60)
        print("CROSS-VALIDATION RESULTS")
        print("="*60)
        print(f"Number of folds: {results['n_folds']}")
        print(f"\nValidation Accuracy: {results['val_accuracy_mean']:.2f}% ± {results['val_accuracy_std']:.2f}%")
        print(f"Test Accuracy: {results['accuracy_mean']:.2f}% ± {results['accuracy_std']:.2f}%")
        print(f"Precision: {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
        print(f"Recall (Sensitivity): {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
        print(f"Specificity: {results['specificity_mean']:.4f} ± {results['specificity_std']:.4f}")
        print(f"F1-Score: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
        
        print("\nPer-fold results:")
        for i, fold_result in enumerate(results['fold_results']):
            print(f"  Fold {i+1}: Val Acc = {fold_result['best_val_acc']:.2f}%, "
                  f"Test Acc = {fold_result['final_metrics']['accuracy']:.2f}%")
        
        print("="*60 + "\n")
    
    def plot_cv_results(self, results: Dict, save_path: str = None):
        """Plot cross-validation results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract fold metrics
        fold_nums = [i+1 for i in range(results['n_folds'])]
        val_accs = [r['best_val_acc'] for r in results['fold_results']]
        accuracies = [r['final_metrics']['accuracy'] for r in results['fold_results']]
        precisions = [r['final_metrics']['precision'] for r in results['fold_results']]
        recalls = [r['final_metrics']['recall'] for r in results['fold_results']]
        
        # Validation accuracy per fold
        axes[0, 0].bar(fold_nums, val_accs, alpha=0.7)
        axes[0, 0].axhline(y=results['val_accuracy_mean'], color='r', linestyle='--', 
                          label=f'Mean: {results["val_accuracy_mean"]:.2f}%')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('Validation Accuracy (%)')
        axes[0, 0].set_title('Validation Accuracy per Fold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Test accuracy per fold
        axes[0, 1].bar(fold_nums, accuracies, alpha=0.7)
        axes[0, 1].axhline(y=results['accuracy_mean'], color='r', linestyle='--',
                          label=f'Mean: {results["accuracy_mean"]:.2f}%')
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('Test Accuracy (%)')
        axes[0, 1].set_title('Test Accuracy per Fold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision and Recall
        x = np.arange(len(fold_nums))
        width = 0.35
        axes[1, 0].bar(x - width/2, precisions, width, label='Precision', alpha=0.7)
        axes[1, 0].bar(x + width/2, recalls, width, label='Recall', alpha=0.7)
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision and Recall per Fold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(fold_nums)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning curves (average across folds)
        # Get average training curves
        max_epochs = max(len(r['train_losses']) for r in results['fold_results'])
        avg_train_losses = []
        avg_val_losses = []
        
        for epoch in range(max_epochs):
            epoch_train_losses = []
            epoch_val_losses = []
            for fold_result in results['fold_results']:
                if epoch < len(fold_result['train_losses']):
                    epoch_train_losses.append(fold_result['train_losses'][epoch])
                    epoch_val_losses.append(fold_result['val_losses'][epoch])
            
            if epoch_train_losses:
                avg_train_losses.append(np.mean(epoch_train_losses))
                avg_val_losses.append(np.mean(epoch_val_losses))
        
        axes[1, 1].plot(avg_train_losses, label='Train Loss', linewidth=2)
        axes[1, 1].plot(avg_val_losses, label='Val Loss', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Average Learning Curves')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Cross-validation plots saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_results(self, results: Dict, save_path: str):
        """Save cross-validation results."""
        # Convert to serializable format
        results_serializable = {
            'n_folds': results['n_folds'],
            'val_accuracy_mean': float(results['val_accuracy_mean']),
            'val_accuracy_std': float(results['val_accuracy_std']),
            'accuracy_mean': float(results['accuracy_mean']),
            'accuracy_std': float(results['accuracy_std']),
            'precision_mean': float(results['precision_mean']),
            'precision_std': float(results['precision_std']),
            'recall_mean': float(results['recall_mean']),
            'recall_std': float(results['recall_std']),
            'f1_mean': float(results['f1_mean']),
            'f1_std': float(results['f1_std']),
            'specificity_mean': float(results['specificity_mean']),
            'specificity_std': float(results['specificity_std']),
            'fold_results': []
        }
        
        for fold_result in results['fold_results']:
            fold_serializable = {
                'fold': fold_result['fold'],
                'best_val_acc': float(fold_result['best_val_acc']),
                'final_metrics': {
                    'accuracy': float(fold_result['final_metrics']['accuracy']),
                    'precision': float(fold_result['final_metrics']['precision']),
                    'recall': float(fold_result['final_metrics']['recall']),
                    'specificity': float(fold_result['final_metrics']['specificity']),
                    'f1_score': float(fold_result['final_metrics']['f1_score']),
                    'confusion_matrix': fold_result['final_metrics']['confusion_matrix']
                },
                'train_losses': [float(x) for x in fold_result['train_losses']],
                'val_losses': [float(x) for x in fold_result['val_losses']],
                'train_accs': [float(x) for x in fold_result['train_accs']],
                'val_accs': [float(x) for x in fold_result['val_accs']]
            }
            results_serializable['fold_results'].append(fold_serializable)
        
        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Cross-validation results saved to {save_path}")


def main():
    """Main cross-validation function."""
    
    # Configuration
    config = {
        'data_dir': r"d:\Stroke-DigitalTwin\outputs\preprocessed",
        'output_dir': r"d:\Stroke-DigitalTwin\outputs\cross_validation",
        'n_folds': 5,
        'batch_size': 4,
        'num_epochs': 30,
        'learning_rate': 1e-4,
        'weight_decay': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Cross-Validation Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use simpler CNN for better generalization
    model_params = {
        'num_classes': 2,
        'in_channels': 3,
        'dropout': 0.5
    }
    
    # Create trainer
    trainer = CrossValidationTrainer(
        model_class=SimpleStrokeCNN,
        model_params=model_params,
        data_dir=config['data_dir'],
        n_folds=config['n_folds'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=config['device']
    )
    
    # Run cross-validation
    results = trainer.run_cross_validation()
    
    # Print results
    trainer.print_cv_results(results)
    
    # Save results
    results_path = output_dir / "cross_validation_results.json"
    trainer.save_results(results, results_path)
    
    # Plot results
    plot_path = output_dir / "cross_validation_plots.png"
    trainer.plot_cv_results(results, save_path=str(plot_path))
    
    print(f"\nCross-validation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

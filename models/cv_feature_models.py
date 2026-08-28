"""
Cross-validation for feature-based models (severity, recovery, recurrence).
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
from digital_twin.twin import DigitalTwin
from severity_feature_model import SeverityFeatureClassifier, SeverityFeatureDataset
from recovery_model import RecoveryPredictor, RecoveryDataset
from recurrence_model import RecurrencePredictor, RecurrenceDataset


class FeatureModelCV:
    """Cross-validation for feature-based models."""
    
    def __init__(self,
                 model_type: str,
                 digital_twin_dir: str,
                 n_folds: int = 5,
                 num_epochs: int = 50,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 device: str = 'cpu'):
        """
        Initialize cross-validation trainer.
        
        Args:
            model_type: 'severity', 'recovery', or 'recurrence'
            digital_twin_dir: Directory with digital twins
            n_folds: Number of folds
            num_epochs: Epochs per fold
            learning_rate: Learning rate
            weight_decay: Weight decay
            device: Device to train on
        """
        self.model_type = model_type
        self.digital_twin_dir = Path(digital_twin_dir)
        self.n_folds = n_folds
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load digital twins
        self.twins = self._load_twins()
        
        # Create dataset based on model type
        if model_type == 'severity':
            self.dataset = SeverityFeatureDataset(
                digital_twins=self.twins,
                severity_thresholds=(677, 3502)
            )
        elif model_type == 'recovery':
            self.dataset = RecoveryDataset(
                digital_twins=self.twins,
                feature_type='imaging'
            )
        elif model_type == 'recurrence':
            self.dataset = RecurrenceDataset(
                digital_twins=self.twins,
                feature_type='combined'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Get labels for stratified splitting
        if model_type == 'recovery':
            # For regression, use discretized labels for stratification
            labels = np.array([self.dataset[i]['recovery_prob'].item() for i in range(len(self.dataset))])
            labels = (labels * 10).astype(int)  # Discretize to 10 bins
        else:
            labels = np.array([self.dataset[i]['severity_label'].item() if model_type == 'severity' 
                              else self.dataset[i]['recurrence_risk'].item() 
                              for i in range(len(self.dataset))])
        
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Label distribution: {np.bincount(labels.astype(int))}")
        print(f"Device: {self.device}")
    
    def _load_twins(self) -> List[DigitalTwin]:
        """Load digital twins."""
        twin_files = sorted(self.digital_twin_dir.glob("digital_twin_*.json"))
        twins = []
        for twin_file in twin_files:
            try:
                twin = DigitalTwin.load(twin_file)
                twins.append(twin)
            except Exception as e:
                print(f"Error loading {twin_file}: {e}")
        return twins
    
    def _create_model(self, input_dim: int) -> nn.Module:
        """Create model based on type."""
        if self.model_type == 'severity':
            return SeverityFeatureClassifier(
                input_dim=input_dim,
                hidden_dims=[64, 32],
                num_classes=3,
                dropout=0.4
            )
        elif self.model_type == 'recovery':
            return RecoveryPredictor(
                input_dim=input_dim,
                hidden_dims=[128, 64, 32],
                dropout=0.3
            )
        elif self.model_type == 'recurrence':
            return RecurrencePredictor(
                input_dim=input_dim,
                hidden_dims=[64, 32],
                num_classes=3,
                dropout=0.5
            )
    
    def _get_criterion(self) -> nn.Module:
        """Get loss function based on model type."""
        if self.model_type == 'recovery':
            return nn.MSELoss()
        else:
            return nn.CrossEntropyLoss()
    
    def _train_fold(self, train_indices: List[int], val_indices: List[int], fold: int) -> Dict:
        """Train a single fold."""
        print(f"\n{'='*60}")
        print(f"Training Fold {fold + 1}/{self.n_folds}")
        print(f"{'='*60}")
        print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
        
        # Create subsets
        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        # Get input dimension
        sample = self.dataset[0]
        input_dim = sample['features'].shape[0]
        
        # Create model
        model = self._create_model(input_dim).to(self.device)
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        criterion = self._get_criterion()
        
        # Training loop
        best_metric = float('inf') if self.model_type == 'recovery' else 0.0
        train_losses = []
        val_losses = []
        train_metrics = []
        val_metrics = []
        
        for epoch in range(self.num_epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                
                if self.model_type == 'recovery':
                    targets = batch['recovery_prob'].to(self.device)
                else:
                    targets = batch['severity_label'].to(self.device) if self.model_type == 'severity' \
                              else batch['recurrence_risk'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                if self.model_type != 'recovery':
                    _, predicted = outputs.max(1)
                    train_total += targets.size(0)
                    train_correct += predicted.eq(targets).sum().item()
            
            train_loss /= len(train_loader)
            if self.model_type != 'recovery':
                train_acc = 100. * train_correct / train_total
                train_metrics.append(train_acc)
            else:
                train_metrics.append(train_loss)
            
            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    
                    if self.model_type == 'recovery':
                        targets = batch['recovery_prob'].to(self.device)
                    else:
                        targets = batch['severity_label'].to(self.device) if self.model_type == 'severity' \
                                  else batch['recurrence_risk'].to(self.device)
                    
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    if self.model_type != 'recovery':
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
            
            val_loss /= len(val_loader)
            if self.model_type != 'recovery':
                val_acc = 100. * val_correct / val_total
                val_metrics.append(val_acc)
                metric = val_acc
            else:
                val_metrics.append(val_loss)
                metric = val_loss
            
            scheduler.step(val_loss)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Save best model
            if (self.model_type == 'recovery' and metric < best_metric) or \
               (self.model_type != 'recovery' and metric > best_metric):
                best_metric = metric
                best_model_state = deepcopy(model.state_dict())
            
            if (epoch + 1) % 10 == 0:
                metric_name = 'Loss' if self.model_type == 'recovery' else 'Acc'
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train {metric_name}: {train_metrics[-1]:.2f}, "
                      f"Val Loss: {val_loss:.4f}, Val {metric_name}: {val_metrics[-1]:.2f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        fold_results = {
            'fold': fold,
            'best_metric': best_metric,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        print(f"Fold {fold + 1} completed. Best {metric_name}: {best_metric:.2f}")
        
        return fold_results
    
    def run_cv(self) -> Dict:
        """Run k-fold cross-validation."""
        print(f"\nStarting {self.n_folds}-fold cross-validation for {self.model_type} model...")
        
        # Get labels for stratified splitting
        if self.model_type == 'recovery':
            labels = np.array([self.dataset[i]['recovery_prob'].item() for i in range(len(self.dataset))])
            labels = (labels * 10).astype(int)  # Discretize to 10 bins
        else:
            labels = np.array([self.dataset[i]['severity_label'].item() if self.model_type == 'severity' 
                              else self.dataset[i]['recurrence_risk'].item() 
                              for i in range(len(self.dataset))])
        
        # Stratified k-fold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        all_fold_results = []
        
        for fold, (train_indices, val_indices) in enumerate(skf.split(range(len(self.dataset)), labels)):
            fold_results = self._train_fold(train_indices, val_indices, fold)
            all_fold_results.append(fold_results)
        
        # Aggregate results
        cv_results = self._aggregate_results(all_fold_results)
        
        return cv_results
    
    def _aggregate_results(self, fold_results: List[Dict]) -> Dict:
        """Aggregate results across folds."""
        metrics = [r['best_metric'] for r in fold_results]
        
        aggregated = {
            'model_type': self.model_type,
            'n_folds': self.n_folds,
            'metric_mean': np.mean(metrics),
            'metric_std': np.std(metrics),
            'fold_results': fold_results
        }
        
        return aggregated
    
    def print_results(self, results: Dict):
        """Print cross-validation results."""
        metric_name = 'Loss' if self.model_type == 'recovery' else 'Accuracy'
        
        print("\n" + "="*60)
        print(f"CROSS-VALIDATION RESULTS - {self.model_type.upper()}")
        print("="*60)
        print(f"Number of folds: {results['n_folds']}")
        print(f"Mean {metric_name}: {results['metric_mean']:.2f} ± {results['metric_std']:.2f}")
        
        print("\nPer-fold results:")
        for i, fold_result in enumerate(results['fold_results']):
            print(f"  Fold {i+1}: {metric_name} = {fold_result['best_metric']:.2f}")
        
        print("="*60 + "\n")
    
    def save_results(self, results: Dict, save_path: str):
        """Save results to JSON."""
        results_serializable = {
            'model_type': results['model_type'],
            'n_folds': results['n_folds'],
            'metric_mean': float(results['metric_mean']),
            'metric_std': float(results['metric_std']),
            'fold_results': []
        }
        
        for fold_result in results['fold_results']:
            fold_serializable = {
                'fold': fold_result['fold'],
                'best_metric': float(fold_result['best_metric']),
                'train_losses': [float(x) for x in fold_result['train_losses']],
                'val_losses': [float(x) for x in fold_result['val_losses']],
                'train_metrics': [float(x) for x in fold_result['train_metrics']],
                'val_metrics': [float(x) for x in fold_result['val_metrics']]
            }
            results_serializable['fold_results'].append(fold_serializable)
        
        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {save_path}")


def main():
    """Run cross-validation for all models."""
    
    config = {
        'digital_twin_dir': r"d:\Stroke-DigitalTwin\outputs\digital_twins",
        'output_dir': r"d:\Stroke-DigitalTwin\outputs\cv_results",
        'n_folds': 5,
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'device': 'cpu'
    }
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run CV for each model type
    for model_type in ['severity', 'recovery', 'recurrence']:
        print(f"\n{'#'*60}")
        print(f"Running cross-validation for {model_type} model")
        print(f"{'#'*60}")
        
        cv = FeatureModelCV(
            model_type=model_type,
            digital_twin_dir=config['digital_twin_dir'],
            n_folds=config['n_folds'],
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            device=config['device']
        )
        
        results = cv.run_cv()
        cv.print_results(results)
        
        save_path = output_dir / f"{model_type}_cv_results.json"
        cv.save_results(results, save_path)
    
    print(f"\nCross-validation completed for all models. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

"""
Training with proper train/validation/test split using ISLES2015 separate test set.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict
import json
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.loader import ISLES2015Loader, resolve_dataset_path
from preprocessing.preprocess import MRIPreprocessor
from models.simple_model import SimpleStrokeCNN
from models.dataset import StrokeDataset, get_train_transforms, get_val_transforms
from models.train import Trainer
from models.evaluate import ModelEvaluator


def prepare_full_dataset():
    """Preprocess both training and testing datasets from ISLES2015."""
    
    dataset_path = str(resolve_dataset_path(r"d:\Stroke-DigitalTwin\dataset"))
    output_dir = Path(r"d:\Stroke-DigitalTwin\outputs\preprocessed_full")
    
    # Create separate directories for train and test
    train_dir = output_dir / "training"
    test_dir = output_dir / "testing"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    loader = ISLES2015Loader(dataset_path)
    preprocessor = MRIPreprocessor(target_size=(224, 224))
    
    # Preprocess training set
    print("Preprocessing training set...")
    train_folders = loader.get_patient_folders(split='training')
    
    for i, patient_folder in enumerate(train_folders):
        print(f"Processing training patient {patient_folder.name} ({i+1}/{len(train_folders)})...")
        try:
            patient_data = loader.load_patient(patient_folder, load_data=True)
            preprocessor.preprocess_patient(patient_data, train_dir, method='lesion')
        except Exception as e:
            print(f"Error processing {patient_folder.name}: {e}")
    
    # Preprocess testing set
    print("\nPreprocessing testing set...")
    test_folders = loader.get_patient_folders(split='testing')
    
    for i, patient_folder in enumerate(test_folders):
        print(f"Processing test patient {patient_folder.name} ({i+1}/{len(test_folders)})...")
        try:
            patient_data = loader.load_patient(patient_folder, load_data=True)
            preprocessor.preprocess_patient(patient_data, test_dir, method='lesion')
        except Exception as e:
            print(f"Error processing {patient_folder.name}: {e}")
    
    print(f"\nDataset preprocessing complete!")
    print(f"Training patients: {len(train_folders)}")
    print(f"Testing patients: {len(test_folders)}")
    
    return train_dir, test_dir


def create_full_training_loaders(train_dir: Path, test_dir: Path, batch_size: int = 4):
    """
    Create loaders for the full training set and the separate test set.
    
    Args:
        train_dir: Directory with training data
        test_dir: Directory with separate test data
        batch_size: DataLoader batch size
        
    Returns:
        Train and test dataloaders
    """
    full_train_dataset = StrokeDataset(
        data_dir=str(train_dir),
        split='training',
        modalities=['DWI'],
        transform=None
    )

    labels = [full_train_dataset[i]['label'].item() for i in range(len(full_train_dataset))]
    print(f"Training size: {len(full_train_dataset)}")
    print(f"Training label distribution: {np.bincount(labels)}")

    full_train_dataset.transform = get_train_transforms()
    train_loader = DataLoader(
        full_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    test_dataset = StrokeDataset(
        data_dir=str(test_dir),
        split='testing',
        modalities=['DWI'],
        transform=get_val_transforms()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"Test size: {len(test_dataset)}")
    return train_loader, test_loader


def train_with_proper_split():
    """Train model with proper train/val/test split."""
    
    # Configuration
    config = {
        'train_dir': r"d:\Stroke-DigitalTwin\outputs\preprocessed_full\training",
        'test_dir': r"d:\Stroke-DigitalTwin\outputs\preprocessed_full\testing",
        'output_dir': r"d:\Stroke-DigitalTwin\outputs\proper_split_training",
        'batch_size': 4,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Check if preprocessed data exists
    train_path = Path(config['train_dir'])
    test_path = Path(config['test_dir'])
    
    if not train_path.exists() or not test_path.exists():
        print("\nPreprocessed data not found. Preprocessing full dataset...")
        train_dir, test_dir = prepare_full_dataset()
        config['train_dir'] = str(train_dir)
        config['test_dir'] = str(test_dir)
    else:
        print("\nUsing existing preprocessed data")
    
    # Create loaders for all training data and the separate test set
    print("\nCreating training and test loaders...")
    train_loader, eval_loader = create_full_training_loaders(
        Path(config['train_dir']),
        Path(config['test_dir']),
        batch_size=config['batch_size']
    )
    
    # Create model
    print("\nCreating model...")
    model = SimpleStrokeCNN(
        num_classes=2,
        in_channels=3,
        dropout=0.5
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        device=config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        gradient_clip=1.0,
        mixed_precision=config['device'] == 'cuda',
        save_dir=str(output_dir)
    )
    
    # Train with early stopping
    class EarlyStopping:
        def __init__(self, patience=10, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_score = None
            self.early_stop = False
        
        def __call__(self, val_acc):
            if self.best_score is None:
                self.best_score = val_acc
            elif val_acc < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_acc
                self.counter = 0
    
    early_stopping = EarlyStopping(patience=10)
    
    print("\nStarting training with early stopping...")
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = trainer.train_epoch()
        
        trainer.scheduler.step()
        
        trainer.history['train_loss'].append(train_loss)
        trainer.history['train_acc'].append(train_acc)
        trainer.history['val_loss'].append(float('nan'))
        trainer.history['val_acc'].append(float('nan'))
        trainer.history['learning_rate'].append(trainer.optimizer.param_groups[0]['lr'])
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Learning Rate: {trainer.optimizer.param_groups[0]['lr']:.6f}")
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            trainer.save_checkpoint(checkpoint_path, epoch, train_acc)
    
    print("\nTraining completed!")
    trainer.best_model_path = output_dir / 'best_model.pth'
    trainer.save_checkpoint(trainer.best_model_path, config['num_epochs'] - 1, train_acc)
    print("Saved final model after training on all training data")
    
    # Save training history
    trainer.save_history()
    trainer.plot_training_curves()
    
    # Evaluate on test set
    print("\nEvaluating on separate test set...")
    
    # Load best model
    model.load_state_dict(torch.load(trainer.best_model_path, map_location=config['device'])['model_state_dict'])
    
    # Evaluate
    evaluator = ModelEvaluator(model, config['device'])
    test_metrics = evaluator.evaluate(eval_loader)
    
    # Print test results
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"Precision:     {test_metrics['precision']:.4f}")
    print(f"Recall:        {test_metrics['recall']:.4f}")
    print(f"Specificity:   {test_metrics['specificity']:.4f}")
    print(f"F1-Score:      {test_metrics['f1_score']:.4f}")
    if test_metrics['auc_roc'] is not None:
        print(f"AUC-ROC:       {test_metrics['auc_roc']:.4f}")
    if test_metrics['auc_pr'] is not None:
        print(f"AUC-PR:        {test_metrics['auc_pr']:.4f}")
    print("="*60)
    
    # Save test results
    test_results_path = output_dir / "test_results.json"
    evaluator.save_evaluation_results(test_metrics, test_results_path)
    
    # Plot test visualizations
    cm_path = output_dir / "test_confusion_matrix.png"
    evaluator.plot_confusion_matrix(test_metrics, save_path=str(cm_path))
    
    roc_path = output_dir / "test_roc_curve.png"
    evaluator.plot_roc_curve(test_metrics, save_path=str(roc_path))
    
    pr_path = output_dir / "test_precision_recall_curve.png"
    evaluator.plot_precision_recall_curve(test_metrics, save_path=str(pr_path))
    
    print(f"\nTest evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    import numpy as np
    train_with_proper_split()

"""
Module: Recurrence Risk Prediction
This module implements recurrence risk prediction for stroke patients.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).parent.parent))
from digital_twin.twin import DigitalTwin, PatientProfile, ImagingFeatures


class RecurrenceDataset(Dataset):
    """Dataset for recurrence risk prediction based on patient features."""
    
    def __init__(self, 
                 digital_twins: List[DigitalTwin],
                 feature_type: str = 'combined'):
        """
        Initialize recurrence dataset.
        
        Args:
            digital_twins: List of DigitalTwin objects
            feature_type: Type of features to use ('imaging', 'profile', or 'combined')
        """
        self.digital_twins = digital_twins
        self.feature_type = feature_type
        
        # Filter twins with required features
        self.valid_twins = []
        for twin in digital_twins:
            if feature_type == 'imaging' and twin.imaging_features:
                self.valid_twins.append(twin)
            elif feature_type == 'profile' and twin.profile:
                self.valid_twins.append(twin)
            elif feature_type == 'combined' and twin.imaging_features and twin.profile:
                self.valid_twins.append(twin)
        
        print(f"Loaded {len(self.valid_twins)} valid twins for recurrence prediction")
    
    def _extract_features(self, twin: DigitalTwin) -> np.ndarray:
        """Extract features from digital twin."""
        features = []
        
        if self.feature_type in ['imaging', 'combined']:
            if twin.imaging_features:
                feat_dict = {
                    'dwi_intensity_mean': twin.imaging_features.dwi_intensity_mean,
                    'dwi_intensity_std': twin.imaging_features.dwi_intensity_std,
                    'flair_intensity_mean': twin.imaging_features.flair_intensity_mean,
                    'flair_intensity_std': twin.imaging_features.flair_intensity_std,
                    't1_intensity_mean': twin.imaging_features.t1_intensity_mean,
                    't1_intensity_std': twin.imaging_features.t1_intensity_std,
                    't2_intensity_mean': twin.imaging_features.t2_intensity_mean,
                    't2_intensity_std': twin.imaging_features.t2_intensity_std,
                    'lesion_area_pixels': twin.imaging_features.lesion_area_pixels,
                    'lesion_center_x': twin.imaging_features.lesion_center_x,
                    'lesion_center_y': twin.imaging_features.lesion_center_y,
                    'lesion_center_z': twin.imaging_features.lesion_center_z,
                }
                features.extend(list(feat_dict.values()))
        
        if self.feature_type in ['profile', 'combined']:
            if twin.profile:
                feat_dict = {
                    'age': twin.profile.age if twin.profile.age else 0,
                    'nihss_score': twin.profile.nihss_score if twin.profile.nihss_score else 0,
                    'time_since_stroke': twin.profile.time_since_stroke if twin.profile.time_since_stroke else 0,
                }
                features.extend(list(feat_dict.values()))
        
        return np.array(features, dtype=np.float32)
    
    def _generate_recurrence_label(self, twin: DigitalTwin) -> int:
        """
        Generate recurrence risk label (0: Low, 1: Medium, 2: High).
        This is a heuristic - in practice, this would come from actual follow-up data.
        """
        if not twin.imaging_features:
            return 0  # Default to low risk
        
        # Risk factors
        risk_score = 0.0
        
        # Lesion size contributes to risk
        lesion_area = twin.imaging_features.lesion_area_pixels
        if lesion_area > 5000:
            risk_score += 2.0
        elif lesion_area > 2000:
            risk_score += 1.0
        
        # Age contributes to risk
        if twin.profile and twin.profile.age:
            age = twin.profile.age
            if age > 75:
                risk_score += 2.0
            elif age > 65:
                risk_score += 1.0
        
        # NIHSS contributes to risk
        if twin.profile and twin.profile.nihss_score:
            nihss = twin.profile.nihss_score
            if nihss > 15:
                risk_score += 2.0
            elif nihss > 10:
                risk_score += 1.0
        
        # Determine risk category
        if risk_score >= 4.0:
            return 2  # High
        elif risk_score >= 2.0:
            return 1  # Medium
        else:
            return 0  # Low
    
    def __len__(self) -> int:
        return len(self.valid_twins)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        twin = self.valid_twins[idx]
        
        features = self._extract_features(twin)
        recurrence_risk = self._generate_recurrence_label(twin)
        
        return {
            'features': torch.from_numpy(features),
            'recurrence_risk': torch.tensor(recurrence_risk, dtype=torch.long),
            'patient_id': twin.patient_id
        }


class RecurrencePredictor(nn.Module):
    """Neural network for recurrence risk prediction (multi-class)."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64],
                 num_classes: int = 3,
                 dropout: float = 0.3):
        """
        Initialize recurrence predictor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of risk classes (3: Low, Medium, High)
            dropout: Dropout rate
        """
        super(RecurrencePredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class RecurrenceTrainer:
    """Trainer for recurrence risk prediction."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: str = 'cuda',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 class_weights: Optional[torch.Tensor] = None,
                 save_dir: str = 'outputs/recurrence_checkpoints'):
        """
        Initialize recurrence trainer.
        
        Args:
            model: Recurrence predictor model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
            class_weights: Optional class weights for imbalanced data
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.uses_validation = val_loader is not None
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function with class weights
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_path = None
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            features = batch['features'].to(self.device)
            labels = batch['recurrence_risk'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc='Validation')
        for batch in pbar:
            features = batch['features'].to(self.device)
            labels = batch['recurrence_risk'].to(self.device)
            
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int, save_every: int = 10):
        """Train the model for multiple epochs."""
        print(f"Starting recurrence risk prediction training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            train_loss, train_acc = self.train_epoch()
            
            if self.uses_validation:
                val_loss, val_acc = self.validate()
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_model_path = self.save_dir / 'best_recurrence_model.pth'
                    self.save_checkpoint(self.best_model_path, epoch, val_acc)
                    print(f"New best model saved with accuracy: {val_acc:.2f}%")
                
                self.scheduler.step(val_loss)
            else:
                val_loss, val_acc = None, None
                self.history['val_loss'].append(float('nan'))
                self.history['val_acc'].append(float('nan'))
                print("Validation disabled")
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if (epoch + 1) % save_every == 0:
                checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                self.save_checkpoint(checkpoint_path, epoch, train_acc)
                print(f"Checkpoint saved: {checkpoint_path}")
        
        print("\nTraining completed!")
        if self.uses_validation:
            print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        else:
            self.best_model_path = self.save_dir / 'best_recurrence_model.pth'
            self.save_checkpoint(self.best_model_path, num_epochs - 1, train_acc)
        
        self.save_history()
        self.plot_training_curves()
    
    def save_checkpoint(self, path: Path, epoch: int, val_acc: float):
        """Save a model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.save_dir / 'recurrence_training_history.json'
        
        history_serializable = {}
        for key, value in self.history.items():
            history_serializable[key] = [float(v) for v in value]
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"Training history saved to {history_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss curve
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Recurrence Risk: Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy curve
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Recurrence Risk: Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # Learning rate curve
        axes[2].plot(self.history['learning_rate'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        plot_path = self.save_dir / 'recurrence_training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {plot_path}")
        plt.close()


def main():
    """Main function to train recurrence predictor."""
    
    # Configuration
    config = {
        'digital_twin_dir': r"d:\Stroke-DigitalTwin\outputs\digital_twins",
        'batch_size': 8,
        'num_epochs': 150,
        'learning_rate': 0.0005,
        'weight_decay': 0.001,  # Increased for stronger regularization
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': r"d:\Stroke-DigitalTwin\outputs\recurrence_checkpoints",
        'feature_type': 'combined',
        'hidden_dims': [64, 32],  # Reduced model complexity
        'num_classes': 3,
        'train_val_split': 0.2,
        'dropout': 0.5  # Increased dropout
    }
    
    print("Recurrence Risk Prediction Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load digital twins
    print("\nLoading digital twins...")
    twin_dir = Path(config['digital_twin_dir'])
    
    if not twin_dir.exists():
        print(f"Digital twin directory {twin_dir} does not exist.")
        print("Please create digital twins first using digital_twin/twin.py")
        return
    
    twin_files = sorted(twin_dir.glob("digital_twin_*.json"))
    
    if len(twin_files) == 0:
        print(f"No digital twins found in {twin_dir}")
        return
    
    twins = []
    for twin_file in twin_files:
        try:
            twin = DigitalTwin.load(twin_file)
            twins.append(twin)
        except Exception as e:
            print(f"Error loading {twin_file}: {e}")
    
    print(f"Loaded {len(twins)} digital twins")
    
    if len(twins) < 10:
        print(f"Warning: Only {len(twins)} twins available. Consider more data for better training.")
    
    # Create dataset
    print("\nCreating recurrence dataset...")
    dataset = RecurrenceDataset(
        digital_twins=twins,
        feature_type=config['feature_type']
    )
    
    if len(dataset) == 0:
        print("No valid twins found for recurrence prediction")
        return
    
    # Get input dimension
    sample_features = dataset[0]['features']
    input_dim = sample_features.shape[0]
    print(f"Input feature dimension: {input_dim}")
    
    # Calculate class weights for imbalanced data
    labels = [dataset[i]['recurrence_risk'].item() for i in range(len(dataset))]
    label_counts = np.bincount(labels, minlength=3)  # Ensure 3 classes
    total_samples = len(dataset)
    class_weights = total_samples / (3 * (label_counts + 1e-6))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    print(f"Class distribution: Low={label_counts[0]}, Medium={label_counts[1]}, High={label_counts[2]}")
    print(f"Class weights: {class_weights}")
    
    # Split into train/val
    train_size = int((1 - config['train_val_split']) * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    
    train_twins = [dataset.valid_twins[i] for i in indices[:train_size]]
    val_twins = [dataset.valid_twins[i] for i in indices[train_size:]]
    
    # Create separate datasets
    train_dataset = RecurrenceDataset(
        digital_twins=train_twins,
        feature_type=config['feature_type']
    )
    val_dataset = RecurrenceDataset(
        digital_twins=val_twins,
        feature_type=config['feature_type']
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\nCreating recurrence predictor...")
    model = RecurrencePredictor(
        input_dim=input_dim,
        hidden_dims=config['hidden_dims'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.3)
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = RecurrenceTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        class_weights=class_weights,
        save_dir=config['save_dir']
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'], save_every=20)
    
    print("\nRecurrence risk prediction training completed!")


if __name__ == "__main__":
    main()

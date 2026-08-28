"""
Module: Recovery Forecast Prediction
This module implements recovery forecast prediction for stroke patients.
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
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(str(Path(__file__).parent.parent))
from digital_twin.twin import DigitalTwin, PatientProfile, ImagingFeatures


class RecoveryDataset(Dataset):
    """Dataset for recovery prediction based on patient features."""
    
    def __init__(self, 
                 digital_twins: List[DigitalTwin],
                 feature_type: str = 'imaging'):
        """
        Initialize recovery dataset.
        
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
        
        print(f"Loaded {len(self.valid_twins)} valid twins for recovery prediction")
    
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
    
    def _generate_recovery_label(self, twin: DigitalTwin) -> float:
        """
        Generate recovery probability based on features.
        This is a heuristic - in practice, this would come from actual follow-up data.
        """
        if not twin.imaging_features:
            return 0.5
        
        # Recovery probability based on lesion size and location
        lesion_area = twin.imaging_features.lesion_area_pixels
        
        # Smaller lesions have better recovery
        base_recovery = 1.0 / (1.0 + lesion_area / 10000.0)
        
        # Adjust based on NIHSS if available
        if twin.profile and twin.profile.nihss_score:
            nihss = twin.profile.nihss_score
            base_recovery *= (1.0 - nihss / 42.0)  # NIHSS max is 42
        
        # Ensure in [0, 1] range
        recovery = np.clip(base_recovery, 0.1, 0.9)
        
        return float(recovery)
    
    def __len__(self) -> int:
        return len(self.valid_twins)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        twin = self.valid_twins[idx]
        
        features = self._extract_features(twin)
        recovery_prob = self._generate_recovery_label(twin)
        
        return {
            'features': torch.from_numpy(features),
            'recovery_prob': torch.tensor(recovery_prob, dtype=torch.float32),
            'patient_id': twin.patient_id
        }


class RecoveryPredictor(nn.Module):
    """Neural network for recovery prediction."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.3):
        """
        Initialize recovery predictor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super(RecoveryPredictor, self).__init__()
        
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
        
        # Output layer (single value for recovery probability)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(1)


class RecoveryTrainer:
    """Trainer for recovery prediction."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: str = 'cuda',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 save_dir: str = 'outputs/recovery_checkpoints'):
        """
        Initialize recovery trainer.
        
        Args:
            model: Recovery predictor model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.uses_validation = val_loader is not None
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function (MSE for regression)
        self.criterion = nn.MSELoss()
        
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
            'val_loss': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.best_model_path = None
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            features = batch['features'].to(self.device)
            targets = batch['recovery_prob'].to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc='Validation')
        for batch in pbar:
            features = batch['features'].to(self.device)
            targets = batch['recovery_prob'].to(self.device)
            
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)
            
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self, num_epochs: int, save_every: int = 10):
        """Train the model for multiple epochs."""
        print(f"Starting recovery prediction training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            train_loss = self.train_epoch()
            
            if self.uses_validation:
                val_loss = self.validate()
                self.history['val_loss'].append(val_loss)
                print(f"Val Loss: {val_loss:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_path = self.save_dir / 'best_recovery_model.pth'
                    self.save_checkpoint(self.best_model_path, epoch, val_loss)
                    print(f"New best model saved with loss: {val_loss:.4f}")
                
                # Update scheduler
                self.scheduler.step(val_loss)
            else:
                val_loss = None
                self.history['val_loss'].append(float('nan'))
                print("Validation disabled")
            
            self.history['train_loss'].append(train_loss)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if (epoch + 1) % save_every == 0:
                checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                self.save_checkpoint(checkpoint_path, epoch, train_loss)
                print(f"Checkpoint saved: {checkpoint_path}")
        
        print("\nTraining completed!")
        if self.uses_validation:
            print(f"Best validation loss: {self.best_val_loss:.4f}")
        else:
            self.best_model_path = self.save_dir / 'best_recovery_model.pth'
            self.save_checkpoint(self.best_model_path, num_epochs - 1, train_loss)
        
        self.save_history()
        self.plot_training_curves()
    
    def save_checkpoint(self, path: Path, epoch: int, val_loss: float):
        """Save a model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.save_dir / 'recovery_training_history.json'
        
        history_serializable = {}
        for key, value in self.history.items():
            history_serializable[key] = [float(v) for v in value]
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"Training history saved to {history_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curve
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Recovery Prediction: Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Learning rate curve
        axes[1].plot(self.history['learning_rate'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        plot_path = self.save_dir / 'recovery_training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {plot_path}")
        plt.close()


def main():
    """Main function to train recovery predictor."""
    
    # Configuration
    config = {
        'digital_twin_dir': r"d:\Stroke-DigitalTwin\outputs\digital_twins",
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': r"d:\Stroke-DigitalTwin\outputs\recovery_checkpoints",
        'feature_type': 'imaging',
        'hidden_dims': [128, 64, 32],
        'train_val_split': 0.2
    }
    
    print("Recovery Prediction Configuration:")
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
    print("\nCreating recovery dataset...")
    dataset = RecoveryDataset(
        digital_twins=twins,
        feature_type=config['feature_type']
    )
    
    if len(dataset) == 0:
        print("No valid twins found for recovery prediction")
        return
    
    # Get input dimension
    sample_features = dataset[0]['features']
    input_dim = sample_features.shape[0]
    print(f"Input feature dimension: {input_dim}")
    
    # Split into train/val
    train_size = int((1 - config['train_val_split']) * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    
    train_twins = [dataset.valid_twins[i] for i in indices[:train_size]]
    val_twins = [dataset.valid_twins[i] for i in indices[train_size:]]
    
    # Create separate datasets
    train_dataset = RecoveryDataset(
        digital_twins=train_twins,
        feature_type=config['feature_type']
    )
    val_dataset = RecoveryDataset(
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
    print("\nCreating recovery predictor...")
    model = RecoveryPredictor(
        input_dim=input_dim,
        hidden_dims=config['hidden_dims'],
        dropout=0.3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = RecoveryTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        save_dir=config['save_dir']
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'], save_every=20)
    
    print("\nRecovery prediction training completed!")


if __name__ == "__main__":
    main()

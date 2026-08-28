"""
Module: Severity Classification
This module implements multi-class severity classification (Mild/Moderate/Severe) for stroke patients.
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
from vit import StrokeViT, MultiModalViT
from dataset import StrokeDataset, MultiModalStrokeDataset, get_train_transforms, get_val_transforms


class SeverityDataset(Dataset):
    """Dataset for severity classification based on lesion volume."""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'training',
                 modalities: List[str] = ['DWI'],
                 transform=None,
                 severity_thresholds: Tuple[float, float] = (1000, 5000)):
        """
        Initialize severity dataset.
        
        Args:
            data_dir: Directory containing preprocessed data
            split: 'training' or 'testing'
            modalities: List of modalities to use
            transform: Albumentations transform
            severity_thresholds: (mild_threshold, moderate_threshold) in pixels
                - < mild_threshold: Mild
                - mild_threshold to moderate_threshold: Moderate
                - > moderate_threshold: Severe
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.modalities = modalities
        self.transform = transform
        self.mild_threshold, self.moderate_threshold = severity_thresholds
        
        # Load patient list
        self.patients = self._get_patient_list()
        
        # Calculate severity labels
        self.severity_labels = self._calculate_severity_labels()
        
        print(f"Loaded {len(self.patients)} patients for {split} split")
        print(f"Severity distribution: Mild={sum(self.severity_labels==0)}, "
              f"Moderate={sum(self.severity_labels==1)}, Severe={sum(self.severity_labels==2)}")
    
    def _get_patient_list(self) -> List[str]:
        """Get list of patient IDs."""
        patient_dir = self.data_dir
        if not patient_dir.exists():
            raise ValueError(f"Data directory {patient_dir} does not exist")
        
        patients = [p.name for p in patient_dir.iterdir() if p.is_dir()]
        return sorted(patients)
    
    def _calculate_severity_labels(self) -> np.ndarray:
        """Calculate severity labels based on lesion volume."""
        labels = []
        
        for patient_id in self.patients:
            patient_dir = self.data_dir / patient_id
            mask_dir = patient_dir / 'mask'
            
            if mask_dir.exists():
                mask_files = list(mask_dir.glob("*.png"))
                total_lesion_area = 0
                
                for mask_file in mask_files:
                    mask_img = np.array(plt.imread(str(mask_file)))
                    if mask_img.ndim == 3:
                        mask_img = mask_img[:, :, 0]
                    total_lesion_area += np.sum(mask_img > 0)
                
                # Determine severity based on lesion area
                if total_lesion_area < self.mild_threshold:
                    severity = 0  # Mild
                elif total_lesion_area < self.moderate_threshold:
                    severity = 1  # Moderate
                else:
                    severity = 2  # Severe
            else:
                # No mask found, default to mild
                severity = 0
            
            labels.append(severity)
        
        return np.array(labels)
    
    def _load_slices(self, patient_dir: Path, modality: str) -> np.ndarray:
        """Load all slices for a modality."""
        import cv2
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
    
    def __len__(self) -> int:
        return len(self.patients)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patient_id = self.patients[idx]
        patient_dir = self.data_dir / patient_id
        
        # Load modality data
        if len(self.modalities) == 1:
            modality = self.modalities[0]
            try:
                image = self._load_slices(patient_dir, modality)
            except ValueError:
                return {
                    'image': torch.zeros((3, 224, 224), dtype=torch.float32),
                    'label': torch.tensor(0, dtype=torch.long),
                    'patient_id': patient_id
                }
            
            # Apply transforms
            if self.transform:
                import cv2
                image_hwc = np.transpose(image, (1, 2, 0))
                transformed = self.transform(image=image_hwc)
                image = transformed['image']
        else:
            # Multi-modal
            import cv2
            modality_slices = []
            for modality in self.modalities:
                slices = self._load_slices(patient_dir, modality)
                modality_slices.append(slices)
            image = np.concatenate(modality_slices, axis=0)
            
            if self.transform:
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image)
        
        return {
            'image': image,
            'label': torch.tensor(self.severity_labels[idx], dtype=torch.long),
            'patient_id': patient_id
        }


class SeverityClassifier(nn.Module):
    """Multi-class severity classifier."""
    
    def __init__(self,
                 model_name: str = 'vit_small_patch16_224',
                 num_classes: int = 3,
                 pretrained: bool = True,
                 in_channels: int = 3,
                 dropout: float = 0.3):
        """
        Initialize severity classifier.
        
        Args:
            model_name: Name of the ViT model
            num_classes: Number of severity classes (3: Mild, Moderate, Severe)
            pretrained: Whether to use pretrained weights
            in_channels: Number of input channels
            dropout: Dropout rate
        """
        super(SeverityClassifier, self).__init__()
        
        self.backbone = StrokeViT(
            model_name=model_name,
            num_classes=0,  # Remove classification head
            pretrained=pretrained,
            in_channels=in_channels,
            dropout=dropout
        )
        
        feature_dim = self.backbone.feature_dim
        
        # Custom severity classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone.get_features(x)
        logits = self.classifier(features)
        return logits


class SeverityTrainer:
    """Trainer for severity classification."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 class_weights: Optional[torch.Tensor] = None,
                 save_dir: str = 'outputs/severity_checkpoints'):
        """
        Initialize severity trainer.
        
        Args:
            model: Severity classifier model
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
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10,
            eta_min=1e-6
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
            inputs = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
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
            inputs = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(inputs)
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
    
    def train(self, num_epochs: int, save_every: int = 5):
        """Train the model for multiple epochs."""
        print(f"Starting severity classification training for {num_epochs} epochs...")
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
                    self.best_model_path = self.save_dir / 'best_severity_model.pth'
                    self.save_checkpoint(self.best_model_path, epoch, val_acc)
                    print(f"New best model saved with accuracy: {val_acc:.2f}%")
            else:
                val_loss, val_acc = None, None
                self.history['val_loss'].append(float('nan'))
                self.history['val_acc'].append(float('nan'))
            
            self.scheduler.step()
            
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
            self.best_model_path = self.save_dir / 'best_severity_model.pth'
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
        history_path = self.save_dir / 'severity_training_history.json'
        history_serializable = {}
        for key, value in self.history.items():
            history_serializable[key] = [float(v) for v in value]
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"Training history saved to {history_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Severity Classification: Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Severity Classification: Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        axes[2].plot(self.history['learning_rate'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        plot_path = self.save_dir / 'severity_training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {plot_path}")
        plt.close()


def main():
    """Main function to train severity classifier."""
    
    # Configuration
    config = {
        'data_dir': r"d:\Stroke-DigitalTwin\outputs\preprocessed_full\training",
        'test_dir': r"d:\Stroke-DigitalTwin\outputs\preprocessed_full\testing",
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': r"d:\Stroke-DigitalTwin\outputs\severity_checkpoints",
        'modalities': ['DWI'],
        'severity_thresholds': (677, 3502),  # Mild, Moderate thresholds in pixels (based on percentiles)
        'train_val_split': 0.2
    }
    
    print("Severity Classification Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create datasets
    print("\nCreating severity datasets...")
    train_dataset = SeverityDataset(
        data_dir=config['data_dir'],
        split='training',
        modalities=config['modalities'],
        transform=get_train_transforms(),
        severity_thresholds=config['severity_thresholds']
    )
    
    # Calculate class weights for imbalanced data
    label_counts = np.bincount(train_dataset.severity_labels)
    total_samples = len(train_dataset)
    class_weights = total_samples / (len(label_counts) * label_counts + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    print(f"Class weights: {class_weights}")
    
    # Split into train/val using patient IDs
    train_size = int((1 - config['train_val_split']) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    
    train_patient_ids = [train_dataset.patients[i] for i in indices[:train_size]]
    val_patient_ids = [train_dataset.patients[i] for i in indices[train_size:]]
    
    # Create separate datasets with patient IDs
    train_dataset = SeverityDataset(
        data_dir=config['data_dir'],
        split='training',
        modalities=config['modalities'],
        transform=get_train_transforms(),
        severity_thresholds=config['severity_thresholds']
    )
    train_dataset.patients = train_patient_ids
    train_dataset.severity_labels = train_dataset._calculate_severity_labels()
    
    val_dataset = SeverityDataset(
        data_dir=config['data_dir'],
        split='training',
        modalities=config['modalities'],
        transform=get_val_transforms(),
        severity_thresholds=config['severity_thresholds']
    )
    val_dataset.patients = val_patient_ids
    val_dataset.severity_labels = val_dataset._calculate_severity_labels()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating severity classifier...")
    model = SeverityClassifier(
        model_name='vit_small_patch16_224',
        num_classes=3,
        pretrained=True,
        in_channels=3,
        dropout=0.4
    )
    
    # Freeze backbone for transfer learning
    for param in model.backbone.backbone.parameters():
        param.requires_grad = False
    
    # Unfreeze last few layers
    for param in model.backbone.backbone.blocks[-4:].parameters():
        param.requires_grad = True
    
    # Always unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Create trainer
    trainer = SeverityTrainer(
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
    trainer.train(num_epochs=config['num_epochs'], save_every=10)
    
    print("\nSeverity classification training completed!")


if __name__ == "__main__":
    main()

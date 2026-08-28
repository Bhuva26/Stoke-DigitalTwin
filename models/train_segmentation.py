"""
Training script for lesion segmentation with proper metrics and loss functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.segmentation_dataset import create_segmentation_dataloaders
from models.segmentation_model import LesionSegmentationUNet


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE and Dice loss."""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.bce_weight * self.bce(pred, target) + 
                self.dice_weight * self.dice(pred, target))


class SegmentationMetrics:
    """Calculate segmentation metrics."""
    
    @staticmethod
    def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """Calculate Intersection over Union."""
        pred = (torch.sigmoid(pred) > threshold).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.item()
    
    @staticmethod
    def calculate_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """Calculate Dice coefficient."""
        pred = (torch.sigmoid(pred) > threshold).float()
        intersection = (pred * target).sum()
        dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
        return dice.item()
    
    @staticmethod
    def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict:
        """Calculate all metrics."""
        return {
            'iou': SegmentationMetrics.calculate_iou(pred, target),
            'dice': SegmentationMetrics.calculate_dice(pred, target)
        }


class SegmentationTrainer:
    """Trainer for lesion segmentation."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 save_dir: str = 'outputs/checkpoints'):
        """
        Initialize trainer.
        
        Args:
            model: Segmentation model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss and optimizer
        self.criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rate': []
        }
        
        self.best_val_dice = 0.0
        self.best_model_path = None
    
    def train_epoch(self) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        
        for batch in tqdm(self.train_loader, desc='Training'):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = SegmentationMetrics.calculate_metrics(outputs, masks)
                total_iou += metrics['iou']
                total_dice += metrics['dice']
        
        avg_loss = total_loss / len(self.train_loader)
        avg_iou = total_iou / len(self.train_loader)
        avg_dice = total_dice / len(self.train_loader)
        
        return avg_loss, avg_iou, avg_dice
    
    def validate(self) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                
                metrics = SegmentationMetrics.calculate_metrics(outputs, masks)
                total_iou += metrics['iou']
                total_dice += metrics['dice']
        
        avg_loss = total_loss / len(self.val_loader)
        avg_iou = total_iou / len(self.val_loader)
        avg_dice = total_dice / len(self.val_loader)
        
        return avg_loss, avg_iou, avg_dice
    
    def save_checkpoint(self, path: Path, epoch: int, val_dice: float):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_dice': val_dice,
            'history': self.history
        }, path)
    
    def save_history(self):
        """Save training history to JSON."""
        with open(self.save_dir / 'segmentation_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # IoU
        axes[0, 1].plot(self.history['train_iou'], label='Train IoU')
        axes[0, 1].plot(self.history['val_iou'], label='Val IoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].set_title('Training and Validation IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Dice
        axes[1, 0].plot(self.history['train_dice'], label='Train Dice')
        axes[1, 0].plot(self.history['val_dice'], label='Val Dice')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].set_title('Training and Validation Dice Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(self.history['learning_rate'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.save_dir / 'segmentation_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {plot_path}")
        plt.close()


def main():
    """Main training function."""
    
    # Configuration
    config = {
        'data_dir': r"d:\Stroke-DigitalTwin\outputs\preprocessed",
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': r"d:\Stroke-DigitalTwin\outputs\segmentation_checkpoints",
        'use_all_slices': True
    }
    
    print("Segmentation Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_segmentation_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=0,
        use_all_slices=config['use_all_slices']
    )
    
    # Create model
    print("\nCreating U-Net model...")
    model = LesionSegmentationUNet(in_channels=1, out_channels=1, bilinear=False)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        save_dir=config['save_dir']
    )
    
    # Training loop with early stopping
    class EarlyStopping:
        def __init__(self, patience=15, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_score = None
            self.early_stop = False
        
        def __call__(self, val_dice):
            if self.best_score is None:
                self.best_score = val_dice
            elif val_dice < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_dice
                self.counter = 0
    
    early_stopping = EarlyStopping(patience=15)
    
    print("\nStarting training...")
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 50)
        
        # Train
        train_loss, train_iou, train_dice = trainer.train_epoch()
        
        # Validate
        val_loss, val_iou, val_dice = trainer.validate()
        
        # Update learning rate
        trainer.scheduler.step()
        
        # Record history
        trainer.history['train_loss'].append(train_loss)
        trainer.history['val_loss'].append(val_loss)
        trainer.history['train_iou'].append(train_iou)
        trainer.history['val_iou'].append(val_iou)
        trainer.history['train_dice'].append(train_dice)
        trainer.history['val_dice'].append(val_dice)
        trainer.history['learning_rate'].append(trainer.optimizer.param_groups[0]['lr'])
        
        # Print summary
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}")
        print(f"Learning Rate: {trainer.optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_dice > trainer.best_val_dice:
            trainer.best_val_dice = val_dice
            trainer.best_model_path = trainer.save_dir / 'best_segmentation_model.pth'
            trainer.save_checkpoint(trainer.best_model_path, epoch, val_dice)
            print(f"New best model saved with Dice: {val_dice:.4f}")
        
        # Early stopping check
        early_stopping(val_dice)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % 20 == 0:
            checkpoint_path = trainer.save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            trainer.save_checkpoint(checkpoint_path, epoch, val_dice)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    print("\nTraining completed!")
    print(f"Best validation Dice: {trainer.best_val_dice:.4f}")
    
    # Save training history
    trainer.save_history()
    trainer.plot_training_curves()


if __name__ == "__main__":
    main()

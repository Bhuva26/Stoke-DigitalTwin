"""
Feature augmentation for better generalization of feature-based models.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
import torch.nn as nn


class FeatureAugmenter:
    """Augment features for better model generalization."""
    
    def __init__(self,
                 noise_std: float = 0.05,
                 dropout_rate: float = 0.1,
                 mixup_alpha: float = 0.2):
        """
        Initialize feature augmenter.
        
        Args:
            noise_std: Standard deviation for Gaussian noise
            dropout_rate: Dropout rate for feature dropout
            mixup_alpha: Alpha parameter for mixup augmentation
        """
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
        self.mixup_alpha = mixup_alpha
    
    def add_gaussian_noise(self, features: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to features."""
        noise = np.random.normal(0, self.noise_std, features.shape)
        return features + noise
    
    def feature_dropout(self, features: np.ndarray) -> np.ndarray:
        """Randomly zero out some features."""
        mask = np.random.random(features.shape) > self.dropout_rate
        return features * mask
    
    def feature_scaling(self, features: np.ndarray) -> np.ndarray:
        """Randomly scale features."""
        scale_factor = np.random.uniform(0.9, 1.1, features.shape)
        return features * scale_factor
    
    def mixup(self, features: np.ndarray, labels: np.ndarray, 
              features2: np.ndarray, labels2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup augmentation: blend two samples.
        
        Args:
            features: First sample features
            labels: First sample labels
            features2: Second sample features
            labels2: Second sample labels
            
        Returns:
            Tuple of (mixed_features, mixed_labels)
        """
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        mixed_features = lam * features + (1 - lam) * features2
        mixed_labels = lam * labels + (1 - lam) * labels2
        return mixed_features, mixed_labels
    
    def augment(self, features: np.ndarray, n_augmentations: int = 3) -> List[np.ndarray]:
        """
        Apply random augmentation to features.
        
        Args:
            features: Original features
            n_augmentations: Number of augmented versions to create
            
        Returns:
            List of augmented feature arrays
        """
        augmented = [features.copy()]
        
        for _ in range(n_augmentations):
            aug_features = features.copy()
            
            # Randomly choose augmentation
            aug_type = np.random.choice(['noise', 'dropout', 'scaling'])
            
            if aug_type == 'noise':
                aug_features = self.add_gaussian_noise(aug_features)
            elif aug_type == 'dropout':
                aug_features = self.feature_dropout(aug_features)
            elif aug_type == 'scaling':
                aug_features = self.feature_scaling(aug_features)
            
            augmented.append(aug_features)
        
        return augmented


class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper with feature augmentation."""
    
    def __init__(self, base_dataset, augmenter: FeatureAugmenter, 
                 n_augmentations: int = 2, augment_prob: float = 0.5):
        """
        Initialize augmented dataset.
        
        Args:
            base_dataset: Base dataset to augment
            augmenter: FeatureAugmenter instance
            n_augmentations: Number of augmentations per sample
            augment_prob: Probability of applying augmentation
        """
        self.base_dataset = base_dataset
        self.augmenter = augmenter
        self.n_augmentations = n_augmentations
        self.augment_prob = augment_prob
    
    def __len__(self) -> int:
        return len(self.base_dataset) * (self.n_augmentations + 1)
    
    def __getitem__(self, idx: int) -> Dict:
        base_idx = idx // (self.n_augmentations + 1)
        aug_idx = idx % (self.n_augmentations + 1)
        
        sample = self.base_dataset[base_idx]
        
        if aug_idx > 0 and np.random.random() < self.augment_prob:
            features = sample['features'].numpy()
            augmented_features = self.augmenter.augment(features, n_augmentations=1)[aug_idx - 1]
            sample['features'] = torch.from_numpy(augmented_features).float()
        
        return sample


def test_augmentation():
    """Test feature augmentation."""
    print("Testing feature augmentation...")
    
    # Create sample features
    features = np.array([0.5, 0.3, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1000, 100, 100, 1])
    
    augmenter = FeatureAugmenter(noise_std=0.05, dropout_rate=0.1, mixup_alpha=0.2)
    
    # Test each augmentation
    print("Original features:", features[:5])
    
    noisy = augmenter.add_gaussian_noise(features)
    print("With noise:", noisy[:5])
    
    dropped = augmenter.feature_dropout(features)
    print("With dropout:", dropped[:5])
    
    scaled = augmenter.feature_scaling(features)
    print("With scaling:", scaled[:5])
    
    # Test mixup
    features2 = np.array([0.6, 0.4, 0.8, 0.7, 0.3, 0.5, 0.7, 0.9, 2000, 150, 120, 2])
    labels = np.array([1.0, 0.0, 0.0])
    labels2 = np.array([0.0, 1.0, 0.0])
    
    mixed_features, mixed_labels = augmenter.mixup(features, labels, features2, labels2)
    print("Mixed features:", mixed_features[:5])
    print("Mixed labels:", mixed_labels)
    
    # Test augment
    augmented = augmenter.augment(features, n_augmentations=3)
    print(f"Created {len(augmented)} augmented versions")
    for i, aug in enumerate(augmented):
        print(f"  Version {i}: {aug[:5]}")
    
    print("Feature augmentation test completed!")


if __name__ == "__main__":
    test_augmentation()

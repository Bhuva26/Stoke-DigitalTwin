"""
Module 4: Vision Transformer
This module implements a Vision Transformer for stroke classification using pretrained models from timm.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Dict, List
import numpy as np


class StrokeViT(nn.Module):
    """Vision Transformer for stroke classification."""
    
    def __init__(self, 
                 model_name: str = 'vit_small_patch16_224',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 in_channels: int = 3,
                 dropout: float = 0.1):
        """
        Initialize the Vision Transformer for stroke classification.
        
        Args:
            model_name: Name of the ViT model from timm
            num_classes: Number of output classes (binary: stroke vs no stroke)
            pretrained: Whether to use pretrained weights
            in_channels: Number of input channels (3 for RGB-like multi-slice input)
            dropout: Dropout rate
        """
        super(StrokeViT, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Load pretrained ViT model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'  # Use global average pooling
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Handle different input channels if needed
        if in_channels != 3:
            self.modify_first_conv()
    
    def modify_first_conv(self):
        """Modify the first convolutional layer to accept different number of input channels."""
        # Get the first convolutional layer
        first_conv = self.backbone.patch_embed.proj
        
        # Create new convolution with adjusted input channels
        if isinstance(first_conv, nn.Conv2d):
            old_weights = first_conv.weight.data
            in_channels = self.in_channels
            
            # Create new conv layer
            new_conv = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            
            # Initialize weights
            if in_channels < 3:
                # Copy weights from first few channels
                new_conv.weight.data[:, :in_channels, :, :] = old_weights[:, :in_channels, :, :]
            elif in_channels > 3:
                # Repeat weights for additional channels
                for i in range(in_channels):
                    new_conv.weight.data[:, i, :, :] = old_weights[:, i % 3, :, :]
            else:
                new_conv.weight.data = old_weights
            
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data
            
            # Replace the first conv
            self.backbone.patch_embed.proj = new_conv
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Extract features
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x):
        """
        Extract features without classification.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Features of shape (batch_size, feature_dim)
        """
        return self.backbone(x)
    
    def get_attention_maps(self, x):
        """
        Get attention maps for explainability.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Attention maps from the last transformer block
        """
        # This requires accessing internal attention weights
        # For now, we'll return features
        # A full implementation would need to hook into the transformer blocks
        return self.get_features(x)


class MultiModalViT(nn.Module):
    """Multi-modal Vision Transformer that processes multiple MRI modalities separately."""
    
    def __init__(self,
                 model_name: str = 'vit_base_patch16_224',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 modalities: List[str] = ['DWI', 'Flair', 'T1', 'T2'],
                 dropout: float = 0.1):
        """
        Initialize multi-modal ViT.
        
        Args:
            model_name: Name of the ViT model from timm
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            modalities: List of modality names
            dropout: Dropout rate
        """
        super(MultiModalViT, self).__init__()
        
        self.modalities = modalities
        self.num_modalities = len(modalities)
        
        # Create separate ViT for each modality
        self.modality_encoders = nn.ModuleDict({
            mod: StrokeViT(model_name, num_classes=0, pretrained=pretrained, 
                          in_channels=3, dropout=dropout)
            for mod in modalities
        })
        
        # Get feature dimension from one encoder
        feature_dim = self.modality_encoders[modalities[0]].feature_dim
        total_feature_dim = feature_dim * self.num_modalities
        
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x_dict: Dict[str, torch.Tensor]):
        """
        Forward pass for multi-modal input.
        
        Args:
            x_dict: Dictionary of modality names to tensors
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features_list = []
        
        # Extract features from each modality
        for modality in self.modalities:
            if modality in x_dict:
                features = self.modality_encoders[modality].get_features(x_dict[modality])
                features_list.append(features)
        
        # Concatenate features
        combined_features = torch.cat(features_list, dim=1)
        
        # Classify
        logits = self.classifier(combined_features)
        
        return logits
    
    def get_modality_features(self, x_dict: Dict[str, torch.Tensor]):
        """
        Get features from each modality separately.
        
        Args:
            x_dict: Dictionary of modality names to tensors
            
        Returns:
            Dictionary of modality features
        """
        features_dict = {}
        for modality in self.modalities:
            if modality in x_dict:
                features_dict[modality] = self.modality_encoders[modality].get_features(x_dict[modality])
        
        return features_dict


def create_model(model_type: str = 'single', **kwargs) -> nn.Module:
    """
    Factory function to create a ViT model.
    
    Args:
        model_type: Type of model ('single' or 'multi')
        **kwargs: Additional arguments for the model
        
    Returns:
        Initialized model
    """
    if model_type == 'single':
        return StrokeViT(**kwargs)
    elif model_type == 'multi':
        return MultiModalViT(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def test_model():
    """Test the model with random input."""
    print("Testing StrokeViT model...")
    
    # Create model
    model = StrokeViT(
        model_name='vit_base_patch16_224',
        num_classes=2,
        pretrained=False,
        in_channels=3
    )
    
    # Print model info
    print(f"Model: {model.model_name}")
    print(f"Feature dimension: {model.feature_dim}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        logits = model(x)
        features = model.get_features(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    print("Single model test passed!")
    
    # Test multi-modal model
    print("\nTesting MultiModalViT model...")
    multi_model = MultiModalViT(
        model_name='vit_base_patch16_224',
        num_classes=2,
        pretrained=False,
        modalities=['DWI', 'Flair', 'T1', 'T2']
    )
    
    print(f"Number of parameters: {sum(p.numel() for p in multi_model.parameters()):,}")
    
    # Test forward pass
    x_dict = {
        'DWI': torch.randn(batch_size, 3, 224, 224),
        'Flair': torch.randn(batch_size, 3, 224, 224),
        'T1': torch.randn(batch_size, 3, 224, 224),
        'T2': torch.randn(batch_size, 3, 224, 224)
    }
    
    with torch.no_grad():
        logits = multi_model(x_dict)
        features_dict = multi_model.get_modality_features(x_dict)
    
    print(f"Logits shape: {logits.shape}")
    for mod, features in features_dict.items():
        print(f"{mod} features shape: {features.shape}")
    print("Multi-modal model test passed!")


if __name__ == "__main__":
    test_model()

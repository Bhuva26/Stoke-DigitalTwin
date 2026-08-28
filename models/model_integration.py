"""
Module: Model Integration
This module integrates trained models (severity, recovery, recurrence) for digital twin predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from severity_feature_model import SeverityFeatureClassifier
from recovery_model import RecoveryPredictor
from recurrence_model import RecurrencePredictor


class ModelIntegrator:
    """Integrates all trained models for digital twin predictions."""
    
    def __init__(self,
                 severity_model_path: str,
                 recovery_model_path: str,
                 recurrence_model_path: str,
                 device: str = 'cpu'):
        """
        Initialize model integrator.
        
        Args:
            severity_model_path: Path to trained severity model
            recovery_model_path: Path to trained recovery model
            recurrence_model_path: Path to trained recurrence model
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading models on {self.device}...")
        
        # Load severity model
        self.severity_model = self._load_severity_model(severity_model_path)
        
        # Load recovery model
        self.recovery_model = self._load_recovery_model(recovery_model_path)
        
        # Load recurrence model
        self.recurrence_model = self._load_recurrence_model(recurrence_model_path)
        
        print("All models loaded successfully!")
    
    def _load_severity_model(self, model_path: str) -> nn.Module:
        """Load trained severity classification model (feature-based)."""
        print(f"Loading severity feature model from {model_path}...")
        
        # Use feature-based classifier instead of ViT
        from severity_feature_model import SeverityFeatureClassifier
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine input dimension from checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Input dimension is the first layer's weight shape
            input_dim = state_dict['network.0.weight'].shape[1]
        else:
            input_dim = 12  # Default
        
        model = SeverityFeatureClassifier(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            num_classes=3,
            dropout=0.4
        )
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_recovery_model(self, model_path: str) -> nn.Module:
        """Load trained recovery prediction model."""
        print(f"Loading recovery model from {model_path}...")
        
        # First, we need to determine input dimension from the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model with default dimensions
        model = RecoveryPredictor(
            input_dim=12,  # Will be adjusted if needed
            hidden_dims=[128, 64, 32],
            dropout=0.3
        )
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_recurrence_model(self, model_path: str) -> nn.Module:
        """Load trained recurrence risk prediction model."""
        print(f"Loading recurrence model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine hidden dimensions from checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Hidden dimensions from state dict
            if 'network.0.weight' in state_dict:
                hidden_dim1 = state_dict['network.0.weight'].shape[0]
                if 'network.4.weight' in state_dict:
                    hidden_dim2 = state_dict['network.4.weight'].shape[0]
                    hidden_dims = [hidden_dim1, hidden_dim2]
                else:
                    hidden_dims = [hidden_dim1]
            else:
                hidden_dims = [64, 32]  # Default
        else:
            hidden_dims = [64, 32]  # Default
        
        # Create model with correct dimensions
        model = RecurrencePredictor(
            input_dim=15,
            hidden_dims=hidden_dims,
            num_classes=3,
            dropout=0.5
        )
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    @torch.no_grad()
    def predict_severity(self, features: np.ndarray, n_samples: int = 100) -> Tuple[int, float, Dict, Dict]:
        """
        Predict stroke severity from imaging features with uncertainty estimation.
        
        Args:
            features: Feature array from digital twin
            n_samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            Tuple of (predicted_class, confidence, class_probabilities, uncertainty_intervals)
        """
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        # Monte Carlo dropout for uncertainty estimation
        # Enable dropout but keep BatchNorm in eval mode
        self.severity_model.eval()
        
        def enable_dropout(m):
            if type(m) == nn.Dropout:
                m.train()
        
        all_probabilities = []
        
        for _ in range(n_samples):
            self.severity_model.apply(enable_dropout)
            outputs = self.severity_model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            all_probabilities.append(probabilities.cpu().numpy())
        
        # Calculate mean and std across samples
        all_probabilities = np.array(all_probabilities)
        mean_probs = np.mean(all_probabilities, axis=0)
        std_probs = np.std(all_probabilities, axis=0)
        
        confidence, predicted_class = torch.max(torch.from_numpy(mean_probs), dim=0)
        
        class_names = ['Mild', 'Moderate', 'Severe']
        class_probs = {
            class_names[i]: float(mean_probs[i])
            for i in range(len(class_names))
        }
        
        # Calculate confidence intervals (95%)
        uncertainty_intervals = {
            class_names[i]: {
                'mean': float(mean_probs[i]),
                'std': float(std_probs[i]),
                'ci_95': (
                    float(np.clip(mean_probs[i] - 1.96 * std_probs[i], 0.0, 1.0)),
                    float(np.clip(mean_probs[i] + 1.96 * std_probs[i], 0.0, 1.0))
                )
            }
            for i in range(len(class_names))
        }
        
        return int(predicted_class.item()), float(confidence.item()), class_probs, uncertainty_intervals
    
    @torch.no_grad()
    def predict_recovery(self, features: np.ndarray, n_samples: int = 100) -> Tuple[float, Dict]:
        """
        Predict recovery probability from features with uncertainty estimation.
        
        Args:
            features: Feature array from digital twin
            n_samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            Tuple of (recovery_probability, uncertainty_intervals)
        """
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        # Monte Carlo dropout for uncertainty estimation
        # Enable dropout but keep BatchNorm in eval mode
        self.recovery_model.eval()
        
        def enable_dropout(m):
            if type(m) == nn.Dropout:
                m.train()
        
        all_predictions = []
        
        for _ in range(n_samples):
            self.recovery_model.apply(enable_dropout)
            recovery_prob = self.recovery_model(features_tensor)
            all_predictions.append(recovery_prob.item())
        
        # Calculate statistics
        all_predictions = np.array(all_predictions)
        mean_prob = np.mean(all_predictions)
        std_prob = np.std(all_predictions)
        
        # Calculate confidence intervals (95%)
        uncertainty_intervals = {
            'mean': float(mean_prob),
            'std': float(std_prob),
            'ci_95': (
                float(np.clip(mean_prob - 1.96 * std_prob, 0.0, 1.0)),
                float(np.clip(mean_prob + 1.96 * std_prob, 0.0, 1.0))
            ),
            'ci_90': (
                float(np.clip(mean_prob - 1.645 * std_prob, 0.0, 1.0)),
                float(np.clip(mean_prob + 1.645 * std_prob, 0.0, 1.0))
            )
        }
        
        return float(mean_prob), uncertainty_intervals
    
    @torch.no_grad()
    def predict_recurrence_risk(self, features: np.ndarray, n_samples: int = 100) -> Tuple[int, float, Dict, Dict]:
        """
        Predict recurrence risk from features with uncertainty estimation.
        
        Args:
            features: Feature array from digital twin
            n_samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            Tuple of (predicted_class, confidence, class_probabilities, uncertainty_intervals)
        """
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        # Monte Carlo dropout for uncertainty estimation
        # Enable dropout but keep BatchNorm in eval mode
        self.recurrence_model.eval()
        
        def enable_dropout(m):
            if type(m) == nn.Dropout:
                m.train()
        
        all_probabilities = []
        
        for _ in range(n_samples):
            self.recurrence_model.apply(enable_dropout)
            outputs = self.recurrence_model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            all_probabilities.append(probabilities.cpu().numpy())
        
        # Calculate mean and std across samples
        all_probabilities = np.array(all_probabilities)
        mean_probs = np.mean(all_probabilities, axis=0)
        std_probs = np.std(all_probabilities, axis=0)
        
        confidence, predicted_class = torch.max(torch.from_numpy(mean_probs), dim=0)
        
        class_names = ['Low', 'Medium', 'High']
        class_probs = {
            class_names[i]: float(mean_probs[i])
            for i in range(len(class_names))
        }
        
        # Calculate confidence intervals (95%)
        uncertainty_intervals = {
            class_names[i]: {
                'mean': float(mean_probs[i]),
                'std': float(std_probs[i]),
                'ci_95': (
                    float(np.clip(mean_probs[i] - 1.96 * std_probs[i], 0.0, 1.0)),
                    float(np.clip(mean_probs[i] + 1.96 * std_probs[i], 0.0, 1.0))
                )
            }
            for i in range(len(class_names))
        }
        
        return int(predicted_class.item()), float(confidence.item()), class_probs, uncertainty_intervals
    
    def predict_all(self, 
                   imaging_features: Optional[Dict] = None,
                   profile_features: Optional[Dict] = None) -> Dict:
        """
        Make predictions using all available models with uncertainty estimation.
        
        Args:
            imaging_features: Imaging features dictionary
            profile_features: Patient profile features dictionary
            
        Returns:
            Dictionary with all predictions including uncertainty intervals
        """
        predictions = {}
        
        # Severity prediction (requires imaging features)
        if imaging_features is not None:
            severity_features = self._extract_severity_features(imaging_features)
            severity_class, severity_conf, severity_probs, severity_uncertainty = self.predict_severity(severity_features)
            predictions['severity'] = {
                'class': severity_class,
                'class_name': ['Mild', 'Moderate', 'Severe'][severity_class],
                'confidence': severity_conf,
                'probabilities': severity_probs,
                'uncertainty_intervals': severity_uncertainty
            }
        
        # Recovery prediction (requires features)
        if imaging_features is not None:
            # Extract features for recovery model
            recovery_features = self._extract_recovery_features(imaging_features, profile_features)
            recovery_prob, recovery_uncertainty = self.predict_recovery(recovery_features)
            predictions['recovery'] = {
                'probability': recovery_prob,
                'uncertainty_intervals': recovery_uncertainty,
                'interpretation': self._interpret_recovery(recovery_prob)
            }
        
        # Recurrence prediction (requires features)
        if imaging_features is not None and profile_features is not None:
            # Extract features for recurrence model
            recurrence_features = self._extract_recurrence_features(imaging_features, profile_features)
            recurrence_class, recurrence_conf, recurrence_probs, recurrence_uncertainty = self.predict_recurrence_risk(recurrence_features)
            predictions['recurrence'] = {
                'class': recurrence_class,
                'class_name': ['Low', 'Medium', 'High'][recurrence_class],
                'confidence': recurrence_conf,
                'probabilities': recurrence_probs,
                'uncertainty_intervals': recurrence_uncertainty
            }
        
        return predictions
    
    def _extract_severity_features(self, imaging_features: Dict) -> np.ndarray:
        """Extract features for severity prediction."""
        features = []
        
        # Imaging features (same as recovery)
        feat_dict = {
            'dwi_intensity_mean': imaging_features.get('dwi_intensity_mean', 0),
            'dwi_intensity_std': imaging_features.get('dwi_intensity_std', 0),
            'flair_intensity_mean': imaging_features.get('flair_intensity_mean', 0),
            'flair_intensity_std': imaging_features.get('flair_intensity_std', 0),
            't1_intensity_mean': imaging_features.get('t1_intensity_mean', 0),
            't1_intensity_std': imaging_features.get('t1_intensity_std', 0),
            't2_intensity_mean': imaging_features.get('t2_intensity_mean', 0),
            't2_intensity_std': imaging_features.get('t2_intensity_std', 0),
            'lesion_area_pixels': imaging_features.get('lesion_area_pixels', 0),
            'lesion_center_x': imaging_features.get('lesion_center_x', 0),
            'lesion_center_y': imaging_features.get('lesion_center_y', 0),
            'lesion_center_z': imaging_features.get('lesion_center_z', 0),
        }
        features.extend(list(feat_dict.values()))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_recovery_features(self, imaging_features: Dict, profile_features: Optional[Dict]) -> np.ndarray:
        """Extract features for recovery prediction."""
        features = []
        
        # Imaging features
        feat_dict = {
            'dwi_intensity_mean': imaging_features.get('dwi_intensity_mean', 0),
            'dwi_intensity_std': imaging_features.get('dwi_intensity_std', 0),
            'flair_intensity_mean': imaging_features.get('flair_intensity_mean', 0),
            'flair_intensity_std': imaging_features.get('flair_intensity_std', 0),
            't1_intensity_mean': imaging_features.get('t1_intensity_mean', 0),
            't1_intensity_std': imaging_features.get('t1_intensity_std', 0),
            't2_intensity_mean': imaging_features.get('t2_intensity_mean', 0),
            't2_intensity_std': imaging_features.get('t2_intensity_std', 0),
            'lesion_area_pixels': imaging_features.get('lesion_area_pixels', 0),
            'lesion_center_x': imaging_features.get('lesion_center_x', 0),
            'lesion_center_y': imaging_features.get('lesion_center_y', 0),
            'lesion_center_z': imaging_features.get('lesion_center_z', 0),
        }
        features.extend(list(feat_dict.values()))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_recurrence_features(self, imaging_features: Dict, profile_features: Dict) -> np.ndarray:
        """Extract features for recurrence prediction."""
        features = []
        
        # Imaging features
        feat_dict = {
            'dwi_intensity_mean': imaging_features.get('dwi_intensity_mean', 0),
            'dwi_intensity_std': imaging_features.get('dwi_intensity_std', 0),
            'flair_intensity_mean': imaging_features.get('flair_intensity_mean', 0),
            'flair_intensity_std': imaging_features.get('flair_intensity_std', 0),
            't1_intensity_mean': imaging_features.get('t1_intensity_mean', 0),
            't1_intensity_std': imaging_features.get('t1_intensity_std', 0),
            't2_intensity_mean': imaging_features.get('t2_intensity_mean', 0),
            't2_intensity_std': imaging_features.get('t2_intensity_std', 0),
            'lesion_area_pixels': imaging_features.get('lesion_area_pixels', 0),
            'lesion_center_x': imaging_features.get('lesion_center_x', 0),
            'lesion_center_y': imaging_features.get('lesion_center_y', 0),
            'lesion_center_z': imaging_features.get('lesion_center_z', 0),
        }
        features.extend(list(feat_dict.values()))
        
        # Profile features
        profile_dict = {
            'age': profile_features.get('age', 0),
            'nihss_score': profile_features.get('nihss_score', 0),
            'time_since_stroke': profile_features.get('time_since_stroke', 0),
        }
        features.extend(list(profile_dict.values()))
        
        return np.array(features, dtype=np.float32)
    
    def _interpret_recovery(self, recovery_prob: float) -> str:
        """Interpret recovery probability."""
        if recovery_prob > 0.7:
            return "Good recovery expected"
        elif recovery_prob > 0.4:
            return "Moderate recovery expected"
        else:
            return "Poor recovery expected"


def main():
    """Test model integration."""
    print("Testing Model Integration...")
    
    # Configuration
    config = {
        'severity_model_path': r"d:\Stroke-DigitalTwin\outputs\severity_checkpoints\best_severity_model.pth",
        'recovery_model_path': r"d:\Stroke-DigitalTwin\outputs\recovery_checkpoints\best_recovery_model.pth",
        'recurrence_model_path': r"d:\Stroke-DigitalTwin\outputs\recurrence_checkpoints\best_recurrence_model.pth",
        'device': 'cpu'
    }
    
    # Create integrator
    integrator = ModelIntegrator(
        severity_model_path=config['severity_model_path'],
        recovery_model_path=config['recovery_model_path'],
        recurrence_model_path=config['recurrence_model_path'],
        device=config['device']
    )
    
    print("\nModel integration test completed!")
    print("Models are ready for digital twin predictions.")


if __name__ == "__main__":
    main()

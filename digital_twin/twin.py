"""
Module 8: Digital Twin
This module creates a digital twin representation of stroke patients.
"""

import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


@dataclass
class PatientProfile:
    """Patient profile data structure."""
    patient_id: str
    age: Optional[float] = None
    gender: Optional[str] = None
    stroke_type: Optional[str] = None
    lesion_volume: Optional[float] = None
    lesion_location: Optional[str] = None
    nihss_score: Optional[float] = None  # National Institutes of Health Stroke Scale
    time_since_stroke: Optional[float] = None  # In hours
    treatment_received: Optional[List[str]] = None
    risk_factors: Optional[List[str]] = None
    segmentation_dice: Optional[float] = None  # Segmentation quality metric
    segmentation_iou: Optional[float] = None  # Segmentation IoU metric


@dataclass
class ImagingFeatures:
    """Imaging features extracted from MRI."""
    dwi_intensity_mean: float
    dwi_intensity_std: float
    flair_intensity_mean: float
    flair_intensity_std: float
    t1_intensity_mean: float
    t1_intensity_std: float
    t2_intensity_mean: float
    t2_intensity_std: float
    lesion_area_pixels: float
    lesion_center_x: float
    lesion_center_y: float
    lesion_center_z: float


class DigitalTwin:
    """Digital Twin for stroke patient."""
    
    def __init__(self, 
                 patient_id: str,
                 patient_profile: Optional[PatientProfile] = None,
                 imaging_features: Optional[ImagingFeatures] = None):
        """
        Initialize a digital twin.
        
        Args:
            patient_id: Unique patient identifier
            patient_profile: Patient profile data
            imaging_features: Extracted imaging features
        """
        self.patient_id = patient_id
        self.profile = patient_profile or PatientProfile(patient_id=patient_id)
        self.imaging_features = imaging_features
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Segmentation results
        self.segmentation_results = {}
        self.lesion_features = {}
        
        # Simulation history
        self.simulation_history = []
    
    def update_profile(self, **kwargs):
        """Update patient profile."""
        for key, value in kwargs.items():
            if hasattr(self.profile, key):
                setattr(self.profile, key, value)
        self.updated_at = datetime.now()
    
    def add_segmentation_result(self, segmentation_result: Dict):
        """Add segmentation result."""
        self.segmentation_results.update(segmentation_result)
        self.updated_at = datetime.now()
    
    def add_lesion_features(self, lesion_features: Dict):
        """Add lesion features from segmentation."""
        self.lesion_features.update(lesion_features)
        self.updated_at = datetime.now()
    
    def add_simulation_result(self, simulation_result: Dict):
        """Add a simulation result."""
        self.simulation_history.append({
            'timestamp': datetime.now().isoformat(),
            'result': simulation_result
        })
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert digital twin to dictionary."""
        return {
            'patient_id': self.patient_id,
            'profile': asdict(self.profile),
            'imaging_features': asdict(self.imaging_features) if self.imaging_features else None,
            'segmentation_results': self.segmentation_results,
            'lesion_features': self.lesion_features,
            'simulation_history': self.simulation_history,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def save(self, save_dir: Path):
        """Save digital twin to JSON file."""
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"digital_twin_{self.patient_id}.json"
        
        with open(save_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return save_path
    
    @classmethod
    def load(cls, load_path: Path):
        """Load digital twin from JSON file."""
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct objects
        profile = PatientProfile(**data['profile'])
        imaging_features = ImagingFeatures(**data['imaging_features']) if data['imaging_features'] else None
        
        twin = cls(
            patient_id=data['patient_id'],
            patient_profile=profile,
            imaging_features=imaging_features
        )
        
        twin.predictions = data['predictions']
        twin.explanations = data['explanations']
        twin.simulation_history = data['simulation_history']
        twin.created_at = datetime.fromisoformat(data['created_at'])
        twin.updated_at = datetime.fromisoformat(data['updated_at'])
        
        return twin


class DigitalTwinFactory:
    """Factory for creating digital twins from patient data."""
    
    def __init__(self, 
                 preprocessed_data_dir: Path):
        """
        Initialize the factory.
        
        Args:
            preprocessed_data_dir: Directory with preprocessed patient data
        """
        self.preprocessed_data_dir = Path(preprocessed_data_dir)
    
    def extract_imaging_features(self, patient_dir: Path) -> ImagingFeatures:
        """
        Extract imaging features from preprocessed patient data.
        
        Args:
            patient_dir: Patient directory with preprocessed data
            
        Returns:
            ImagingFeatures object
        """
        features = {}
        
        # Extract features from each modality
        for modality in ['DWI', 'Flair', 'T1', 'T2']:
            modality_dir = patient_dir / modality
            if modality_dir.exists():
                slice_files = sorted(modality_dir.glob("*.png"))
                if slice_files:
                    # Load all slices
                    slices = []
                    for slice_file in slice_files[:3]:
                        img = cv2.imread(str(slice_file), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            slices.append(img)
                    
                    if slices:
                        stacked = np.stack(slices, axis=0)
                        features[f'{modality.lower()}_intensity_mean'] = float(np.mean(stacked))
                        features[f'{modality.lower()}_intensity_std'] = float(np.std(stacked))
        
        # Extract lesion features
        mask_dir = patient_dir / 'mask'
        if mask_dir.exists():
            mask_files = sorted(mask_dir.glob("*.png"))
            if mask_files:
                masks = []
                centers_x, centers_y, centers_z = [], [], []
                
                for i, mask_file in enumerate(mask_files[:3]):
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        masks.append(mask)
                        
                        # Find lesion center
                        moments = cv2.moments(mask)
                        if moments['m00'] > 0:
                            cx = moments['m10'] / moments['m00']
                            cy = moments['m01'] / moments['m00']
                            centers_x.append(cx)
                            centers_y.append(cy)
                            centers_z.append(i)
                
                if masks:
                    stacked_masks = np.stack(masks, axis=0)
                    features['lesion_area_pixels'] = float(np.sum(stacked_masks > 0))
                    features['lesion_center_x'] = float(np.mean(centers_x)) if centers_x else 0.0
                    features['lesion_center_y'] = float(np.mean(centers_y)) if centers_y else 0.0
                    features['lesion_center_z'] = float(np.mean(centers_z)) if centers_z else 0.0
        
        # Fill missing features with defaults
        defaults = {
            'dwi_intensity_mean': 0.0, 'dwi_intensity_std': 0.0,
            'flair_intensity_mean': 0.0, 'flair_intensity_std': 0.0,
            't1_intensity_mean': 0.0, 't1_intensity_std': 0.0,
            't2_intensity_mean': 0.0, 't2_intensity_std': 0.0,
            'lesion_area_pixels': 0.0,
            'lesion_center_x': 0.0, 'lesion_center_y': 0.0, 'lesion_center_z': 0.0
        }
        
        for key, default in defaults.items():
            if key not in features:
                features[key] = default
        
        return ImagingFeatures(**features)
    
    def create_digital_twin(self, 
                           patient_id: str,
                           patient_profile: Optional[PatientProfile] = None) -> DigitalTwin:
        """
        Create a digital twin for a patient using the dataset-provided lesion mask.
        
        Args:
            patient_id: Patient identifier
            patient_profile: Optional patient profile data
            
        Returns:
            DigitalTwin object
        """
        patient_dir = self.preprocessed_data_dir / patient_id
        
        if not patient_dir.exists():
            raise ValueError(f"Patient directory {patient_dir} does not exist")
        
        # Extract imaging features
        imaging_features = self.extract_imaging_features(patient_dir)
        
        # Create digital twin
        twin = DigitalTwin(
            patient_id=patient_id,
            patient_profile=patient_profile,
            imaging_features=imaging_features
        )
        
        return twin
    
    def create_batch_twins(self, 
                          patient_ids: List[str],
                          profiles: Optional[Dict[str, PatientProfile]] = None) -> List[DigitalTwin]:
        """
        Create digital twins for multiple patients.
        
        Args:
            patient_ids: List of patient identifiers
            profiles: Optional dictionary of patient profiles
            
        Returns:
            List of DigitalTwin objects
        """
        twins = []
        profiles = profiles or {}
        
        for patient_id in patient_ids:
            print(f"Creating digital twin for patient {patient_id}...")
            profile = profiles.get(patient_id)
            
            try:
                twin = self.create_digital_twin(patient_id, profile)
                twins.append(twin)
            except Exception as e:
                print(f"Error creating twin for patient {patient_id}: {e}")
        
        return twins


class DigitalTwinAnalyzer:
    """Analyzer for digital twin data."""
    
    def __init__(self, twins: List[DigitalTwin]):
        """
        Initialize analyzer.
        
        Args:
            twins: List of DigitalTwin objects
        """
        self.twins = twins
    
    def analyze_feature_distribution(self):
        """Analyze distribution of imaging features."""
        # Collect features
        feature_matrix = []
        feature_names = None
        
        for twin in self.twins:
            if twin.imaging_features:
                features = asdict(twin.imaging_features)
                if feature_names is None:
                    feature_names = list(features.keys())
                feature_matrix.append([features[name] for name in feature_names])
        
        if not feature_matrix:
            print("No imaging features found")
            return None
        
        feature_matrix = np.array(feature_matrix)
        
        # Compute statistics
        stats = {}
        for i, name in enumerate(feature_names):
            stats[name] = {
                'mean': float(np.mean(feature_matrix[:, i])),
                'std': float(np.std(feature_matrix[:, i])),
                'min': float(np.min(feature_matrix[:, i])),
                'max': float(np.max(feature_matrix[:, i]))
            }
        
        return stats
    
    def cluster_patients(self, n_clusters: int = 3):
        """
        Cluster patients based on imaging features.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Cluster labels and cluster centers
        """
        # Collect features
        feature_matrix = []
        valid_twins = []
        
        for twin in self.twins:
            if twin.imaging_features:
                features = asdict(twin.imaging_features)
                feature_matrix.append(list(features.values()))
                valid_twins.append(twin)
        
        if len(feature_matrix) < n_clusters:
            print(f"Not enough patients for {n_clusters} clusters")
            return None, None
        
        feature_matrix = np.array(feature_matrix)
        
        # Normalize features
        feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-8)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(feature_matrix)
        
        # Assign clusters to twins
        for twin, label in zip(valid_twins, labels):
            twin.cluster_id = int(label)
        
        return labels, kmeans.cluster_centers_
    
    def visualize_feature_pca(self, save_path: Optional[str] = None):
        """
        Visualize patients in PCA space.
        
        Args:
            save_path: Optional path to save visualization
        """
        # Collect features
        feature_matrix = []
        patient_ids = []
        
        for twin in self.twins:
            if twin.imaging_features:
                features = asdict(twin.imaging_features)
                feature_matrix.append(list(features.values()))
                patient_ids.append(twin.patient_id)
        
        if len(feature_matrix) < 2:
            print("Not enough patients for PCA visualization")
            return
        
        feature_matrix = np.array(feature_matrix)
        
        # Normalize
        feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-8)
        
        # PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(feature_matrix)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c='blue', alpha=0.6)
        
        # Annotate points
        for i, patient_id in enumerate(patient_ids):
            plt.annotate(patient_id, (features_2d[i, 0], features_2d[i, 1]), 
                        fontsize=8, alpha=0.7)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title('Patient Digital Twins in PCA Space')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"PCA visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main function to test digital twin creation."""
    
    # Configuration
    config = {
        'preprocessed_data_dir': r"d:\Stroke-DigitalTwin\outputs\preprocessed",
        'output_dir': r"d:\Stroke-DigitalTwin\outputs\digital_twins",
        'patient_ids': ['1', '2', '3', '4', '5']
    }
    
    print("Digital Twin Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create factory
    factory = DigitalTwinFactory(
        preprocessed_data_dir=config['preprocessed_data_dir']
    )
    
    # Create digital twins
    twins = factory.create_batch_twins(config['patient_ids'])
    
    print(f"\nCreated {len(twins)} digital twins")
    
    # Save twins
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for twin in twins:
        save_path = twin.save(output_dir)
        print(f"Saved digital twin for patient {twin.patient_id} to {save_path}")
    
    # Analyze twins
    analyzer = DigitalTwinAnalyzer(twins)
    
    # Feature distribution
    print("\nFeature Distribution:")
    stats = analyzer.analyze_feature_distribution()
    if stats:
        for feature, values in stats.items():
            print(f"  {feature}: mean={values['mean']:.2f}, std={values['std']:.2f}")
    
    # PCA visualization
    pca_path = output_dir / "pca_visualization.png"
    analyzer.visualize_feature_pca(save_path=str(pca_path))
    
    print("\nDigital twin creation completed!")


if __name__ == "__main__":
    main()

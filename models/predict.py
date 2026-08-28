"""
Module 6: Prediction
This module makes predictions using the trained Vision Transformer model.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.segmentation_model import SimpleSegmentationUNet
from models.dataset import StrokeDataset, get_val_transforms
from preprocessing.loader import ISLES2015Loader
from preprocessing.preprocess import MRIPreprocessor


class StrokePredictor:
    """Predictor for lesion segmentation and detection."""
    
    def __init__(self,
                 model_path: str,
                 model_type: str = 'single',
                 model_name: str = 'unet',
                 modalities: List[str] = ['DWI'],
                 device: str = 'cuda'):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model ('single' or 'multi')
            model_name: Name of the architecture
            modalities: List of modalities to use
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.modalities = modalities
        
        print(f"Loading model from {model_path}...")
        print(f"Device: {self.device}")
        
        # Create model (SimpleSegmentationUNet expects 3 in_channels for 3 slices stacked as channels)
        self.model = SimpleSegmentationUNet(in_channels=3)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
        
        # Load metadata if available
        self.metadata = checkpoint.get('metadata', {}) if isinstance(checkpoint, dict) else {}
        if isinstance(checkpoint, dict) and 'val_loss' in checkpoint:
            print(f"Model validation loss: {checkpoint['val_loss'][-1] if isinstance(checkpoint['val_loss'], list) else checkpoint['val_loss']:.4f}")
    
    @torch.no_grad()
    def predict_single(self, image: Union[np.ndarray, torch.Tensor]) -> Dict:
        """
        Make prediction on a single image.
        
        Args:
            image: Input image (C, H, W) where C=3 (slices as channels)
            
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess
        if isinstance(image, np.ndarray):
            # Ensure CHW format
            if image.ndim == 3 and image.shape[0] != 3:
                # If not CHW, convert
                if image.shape[2] == 3:  # HWC
                    image = np.transpose(image, (2, 0, 1))
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension: (1, C, H, W)
            image = torch.from_numpy(image).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.unsqueeze(0)  # Add batch dimension
            # Normalize if needed
            if image.max() > 1.0:
                image = image / 255.0
        
        image = image.to(self.device)
        
        # Predict logits
        outputs = self.model(image)  # Shape: (1, 1, H, W)
        probabilities = torch.sigmoid(outputs)  # Shape: (1, 1, H, W)
        
        # Threshold to get binary mask
        predicted_mask = (probabilities > 0.5).float()
        
        # Determine if lesion is detected
        lesion_pixels = predicted_mask.sum().item()
        predicted_class = 1 if lesion_pixels > 0 else 0
        
        # Calculate confidence
        if predicted_class == 1:
            confidence = probabilities[predicted_mask == 1].mean().item()
        else:
            confidence = 1.0 - probabilities.mean().item()
            
        probs_list = [1.0 - confidence, confidence]
        
        # Convert predicted mask to numpy (H, W)
        mask_np = predicted_mask.squeeze(0).squeeze(0).cpu().numpy()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probs_list,
            'class_names': ['No Lesion', 'Lesion'],
            'predicted_mask': mask_np,
            'predicted_volume': lesion_pixels
        }
    
    @torch.no_grad()
    def predict_multi_modal(self, modality_images: Dict[str, np.ndarray]) -> Dict:
        """
        Make prediction on multi-modal input.
        
        Args:
            modality_images: Dictionary of modality names to images
            
        Returns:
            Dictionary containing prediction results
        """
        if 'DWI' in modality_images:
            return self.predict_single(modality_images['DWI'])
        else:
            # Fallback
            for key, val in modality_images.items():
                return self.predict_single(val)
            return self.predict_single(np.zeros((3, 224, 224), dtype=np.uint8))
    
    def predict_from_preprocessed(self, patient_dir: Path) -> Dict:
        """
        Make prediction from preprocessed patient data.
        
        Args:
            patient_dir: Directory containing preprocessed patient data
            
        Returns:
            Dictionary containing prediction results
        """
        if self.model_type == 'single':
            # Load single modality
            modality = self.modalities[0]
            modality_dir = patient_dir / modality
            
            if not modality_dir.exists():
                raise ValueError(f"Modality directory {modality_dir} does not exist")
            
            # Load and stack slices
            slice_files = sorted(modality_dir.glob("*.png"))
            slices = []
            for slice_file in slice_files[:3]:
                img = cv2.imread(str(slice_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    slices.append(img)
            
            if len(slices) == 0:
                raise ValueError(f"No slices found for {modality}")
            
            # Stack as channels (3, H, W) - each slice becomes a channel
            image = np.stack(slices, axis=0)  # Shape: (3, H, W)
            
            # Debug: print shape
            print(f"Loaded image shape: {image.shape}")
            
            return self.predict_single(image)
        else:
            # Load multiple modalities
            modality_images = {}
            for modality in self.modalities:
                modality_dir = patient_dir / modality
                
                if not modality_dir.exists():
                    print(f"Warning: {modality} directory not found, using zeros")
                    modality_images[modality] = np.zeros((3, 224, 224), dtype=np.uint8)
                    continue
                
                slice_files = sorted(modality_dir.glob("*.png"))
                slices = []
                for slice_file in slice_files[:3]:
                    img = cv2.imread(str(slice_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        slices.append(img)
                
                if len(slices) == 0:
                    modality_images[modality] = np.zeros((3, 224, 224), dtype=np.uint8)
                else:
                    modality_images[modality] = np.stack(slices, axis=0)
            
            return self.predict_multi_modal(modality_images)
    
    def predict_from_raw(self, patient_folder: Path, preprocessor: MRIPreprocessor) -> Dict:
        """
        Make prediction from raw patient data (requires preprocessing).
        
        Args:
            patient_folder: Path to raw patient folder
            preprocessor: MRIPreprocessor instance
            
        Returns:
            Dictionary containing prediction results
        """
        # Load raw data
        loader = ISLES2015Loader(str(patient_folder.parent.parent))
        patient_data = loader.load_patient(patient_folder, load_data=True)
        
        # Preprocess
        temp_dir = Path("temp_preprocessed")
        temp_dir.mkdir(exist_ok=True)
        
        processed_info = preprocessor.preprocess_patient(
            patient_data, 
            temp_dir, 
            method='lesion'
        )
        
        # Predict
        result = self.predict_from_preprocessed(temp_dir / patient_data['patient_id'])
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        return result
    
    def batch_predict(self, data_dir: Path, output_file: Optional[str] = None) -> List[Dict]:
        """
        Make predictions on all patients in a directory.
        
        Args:
            data_dir: Directory containing preprocessed patient data
            output_file: Optional path to save results as JSON
            
        Returns:
            List of prediction results
        """
        results = []
        
        patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        
        print(f"Making predictions on {len(patient_dirs)} patients...")
        
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            print(f"Predicting for patient {patient_id}...")
            
            try:
                result = self.predict_from_preprocessed(patient_dir)
                result['patient_id'] = patient_id
                results.append(result)
            except Exception as e:
                print(f"Error predicting for patient {patient_id}: {e}")
                results.append({
                    'patient_id': patient_id,
                    'error': str(e)
                })
        
        # Save results if requested
        if output_file:
            # Create a serializable copy of results without the large numpy mask
            serializable_results = []
            for r in results:
                r_copy = r.copy()
                if 'predicted_mask' in r_copy:
                    del r_copy['predicted_mask']
                serializable_results.append(r_copy)
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results
    
    def visualize_prediction(self, image: np.ndarray, result: Dict, save_path: Optional[str] = None):
        """
        Visualize prediction with image.
        
        Args:
            image: Input image
            result: Prediction result dictionary
            save_path: Optional path to save visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display image
        if image.ndim == 3:
            # Take middle slice if 3D
            if image.shape[0] == 3:  # CHW format
                display_img = image[1]  # Middle slice
            else:
                display_img = image[image.shape[0] // 2]
        else:
            display_img = image
        
        axes[0].imshow(display_img, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Display prediction
        class_names = result['class_names']
        probabilities = result['probabilities']
        
        bars = axes[1].bar(class_names, probabilities)
        axes[1].set_ylim([0, 1])
        axes[1].set_ylabel('Probability')
        axes[1].set_title(f'Prediction: {class_names[result["predicted_class"]]} (Confidence: {result["confidence"]:.2f})')
        
        # Color bars
        for i, bar in enumerate(bars):
            if i == result['predicted_class']:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main function to test prediction."""
    
    # Configuration
    config = {
        'model_path': r"d:\Stroke-DigitalTwin\outputs\segmentation_model.pth",
        'model_type': 'single',
        'model_name': 'unet',
        'modalities': ['DWI'],
        'data_dir': r"d:\Stroke-DigitalTwin\outputs\preprocessed",
        'output_file': r"d:\Stroke-DigitalTwin\outputs\predictions.json"
    }
    
    print("Prediction Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create predictor
    predictor = StrokePredictor(
        model_path=config['model_path'],
        model_type=config['model_type'],
        model_name=config['model_name'],
        modalities=config['modalities']
    )
    
    # Batch predict
    data_dir = Path(config['data_dir'])
    if data_dir.exists():
        results = predictor.batch_predict(data_dir, config['output_file'])
        
        # Print summary
        print("\nPrediction Summary:")
        lesion_count = sum(1 for r in results if r.get('predicted_class') == 1)
        no_lesion_count = sum(1 for r in results if r.get('predicted_class') == 0)
        
        print(f"Total predictions: {len(results)}")
        print(f"Lesion: {lesion_count}")
        print(f"No Lesion: {no_lesion_count}")
        
        # Show detailed results for first few patients
        print("\nDetailed Results:")
        for result in results[:5]:
            if 'error' not in result:
                print(f"Patient {result['patient_id']}: {result['class_names'][result['predicted_class']]} (Confidence: {result['confidence']:.3f}, Volume: {result.get('predicted_volume', 0)} px)")
    else:
        print(f"Data directory {data_dir} does not exist")


if __name__ == "__main__":
    main()

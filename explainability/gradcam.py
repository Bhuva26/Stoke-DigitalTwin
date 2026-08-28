"""
Module 7: Explainable AI
This module implements Grad-CAM for explaining model predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.segmentation_model import SimpleSegmentationUNet


class GradCAM:
    """Grad-CAM for visualizing model attention."""
    
    def __init__(self, model: nn.Module, target_layer: str):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The model to analyze
            target_layer: Name of the target layer for Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Target layer {self.target_layer} not found")
        
        self.forward_hook = target_module.register_forward_hook(forward_hook)
        self.backward_hook = target_module.register_backward_hook(backward_hook)
    
    def remove_hooks(self):
        """Remove the registered hooks."""
        self.forward_hook.remove()
        self.backward_hook.remove()
    
    def generate_cam(self, 
                    input_tensor: torch.Tensor, 
                    target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class for Grad-CAM. If None, uses predicted class.
            
        Returns:
            CAM as numpy array (H, W)
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        output[0, target_class].backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    def overlay_cam(self, 
                    image: np.ndarray, 
                    cam: np.ndarray, 
                    alpha: float = 0.4) -> np.ndarray:
        """
        Overlay CAM on original image.
        
        Args:
            image: Original image (H, W) or (H, W, C)
            cam: CAM (H, W)
            alpha: Transparency of overlay
            
        Returns:
            Overlayed image
        """
        # Resize CAM to match image size
        if image.ndim == 2:
            h, w = image.shape
        else:
            h, w = image.shape[:2]
        
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap
        cam_colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # Convert image to RGB if needed
        if image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        # Normalize image to 0-255
        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        
        # Overlay
        overlay = cv2.addWeighted(image_rgb, 1 - alpha, cam_colored, alpha, 0)
        
        return overlay


class ViTAttentionRollout:
    """Attention Rollout for Vision Transformer explainability."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize Attention Rollout.
        
        Args:
            model: Vision Transformer model
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.attention_maps = []
        
        # Register hooks to capture attention maps
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture attention maps."""
        def get_attention_hook(module, input, output):
            # output is typically (batch_size, num_heads, seq_len, seq_len)
            self.attention_maps.append(output)
        
        # Register hooks for all transformer blocks
        for name, module in self.model.named_modules():
            if 'attn' in name and 'qkv' not in name and 'proj' not in name:
                # This is a rough heuristic - adjust based on actual model structure
                if hasattr(module, 'weight'):
                    try:
                        module.register_forward_hook(get_attention_hook)
                    except:
                        pass
    
    def generate_attention_rollout(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate attention rollout heatmap.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            
        Returns:
            Attention heatmap as numpy array
        """
        self.attention_maps = []
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if not self.attention_maps:
            # Fallback: use gradient-based method
            return self._gradient_based_attention(input_tensor)
        
        # Compute attention rollout
        # Start with identity matrix
        rollout = torch.eye(self.attention_maps[0].shape[-1]).to(self.device)
        
        for attn_map in self.attention_maps:
            # Average across heads
            attn_avg = attn_map.mean(dim=1)  # (batch_size, seq_len, seq_len)
            
            # Add residual connection
            attn_avg = attn_avg + torch.eye(attn_avg.shape[-1]).to(self.device) / attn_avg.shape[-1]
            
            # Normalize
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            
            # Multiply
            rollout = rollout @ attn_avg
        
        # Get attention to CLS token
        cls_attention = rollout[0, 0, 1:]  # Exclude CLS token itself
        
        # Reshape to spatial dimensions
        # For ViT with patch size 16, 224x224 image -> 14x14 patches
        num_patches = int(np.sqrt(cls_attention.shape[0]))
        heatmap = cls_attention.reshape(num_patches, num_patches).cpu().numpy()
        
        # Upsample to original image size
        heatmap = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[3]))
        
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def _gradient_based_attention(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Fallback gradient-based attention method.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            
        Returns:
            Attention heatmap
        """
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get predicted class
        target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients
        gradients = input_tensor.grad[0]  # (C, H, W)
        
        # Compute magnitude
        gradient_mag = torch.sqrt(torch.sum(gradients ** 2, dim=0))  # (H, W)
        
        # Normalize
        gradient_mag = (gradient_mag - gradient_mag.min()) / (gradient_mag.max() - gradient_mag.min() + 1e-8)
        
        return gradient_mag.cpu().numpy()
    
    def overlay_cam(self, 
                    image: np.ndarray, 
                    cam: np.ndarray, 
                    alpha: float = 0.4) -> np.ndarray:
        """
        Overlay CAM on original image.
        
        Args:
            image: Original image (H, W) or (H, W, C)
            cam: CAM (H, W)
            alpha: Transparency of overlay
            
        Returns:
            Overlayed image
        """
        # Resize CAM to match image size
        if image.ndim == 2:
            h, w = image.shape
        else:
            h, w = image.shape[:2]
        
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap
        cam_colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # Convert image to RGB if needed
        if image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        # Normalize image to 0-255
        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        
        # Overlay
        overlay = cv2.addWeighted(image_rgb, 1 - alpha, cam_colored, alpha, 0)
        
        return overlay


class StrokeExplainer:
    """Explainer for lesion segmentation model, generating visual overlays of predictions."""
    
    def __init__(self, 
                 model_path: str,
                 model_type: str = 'single',
                 model_name: str = 'unet',
                 modalities: List[str] = ['DWI'],
                 device: str = 'cuda',
                 method: str = 'segmentation_overlay'):
        """
        Initialize the explainer.
        
        Args:
            model_path: Path to trained model
            model_type: Type of model
            model_name: Name of architecture
            modalities: List of modalities
            device: Device to run on
            method: Explanation method
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.method = method
        self.modalities = modalities
        
        # Load model
        self.model = SimpleSegmentationUNet(in_channels=3)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
        print("Explainer model loaded successfully!")
    
    def explain_image(self, 
                      image: np.ndarray, 
                      save_path: Optional[str] = None) -> Dict:
        """
        Generate explanation (predicted segmentation heatmap and overlay) for an image.
        
        Args:
            image: Input image (C, H, W)
            save_path: Optional path to save visualization
            
        Returns:
            Dictionary containing explanation results
        """
        # Prepare input
        if isinstance(image, np.ndarray):
            # Ensure CHW format
            if image.ndim == 3 and image.shape[0] != 3:
                if image.shape[2] == 3:  # HWC
                    image = np.transpose(image, (2, 0, 1))
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
        else:
            image_tensor = image.unsqueeze(0)
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
        
        image_tensor = image_tensor.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)  # (1, 1, H, W)
            probabilities = torch.sigmoid(outputs)  # (1, 1, H, W)
            predicted_mask = (probabilities > 0.5).float()
            
        # Determine if lesion is detected
        lesion_pixels = predicted_mask.sum().item()
        predicted_class = 1 if lesion_pixels > 0 else 0
        
        # Calculate confidence
        if predicted_class == 1:
            confidence = probabilities[predicted_mask == 1].mean().item()
        else:
            confidence = 1.0 - probabilities.mean().item()
            
        # Heatmap is the 2D probabilities array (H, W)
        cam = probabilities.squeeze(0).squeeze(0).cpu().numpy()
        
        # Take middle slice (slice 1) for original image display
        if isinstance(image, np.ndarray):
            display_image = image[1] if image.shape[0] == 3 else image
        else:
            display_image = image[1].cpu().numpy() if image.shape[0] == 3 else image.cpu().numpy()
            
        # Ensure display image is in [0, 1] range for visualization functions
        if display_image.max() > 1.0:
            display_image = display_image / 255.0
            
        # Overlay
        # Custom helper to overlay the mask
        overlay = self.overlay_cam(display_image, cam, alpha=0.4)
        
        # Visualize
        if save_path:
            self._visualize_explanation(display_image, cam, overlay, 
                                        predicted_class, confidence, 
                                        save_path)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'heatmap': cam,
            'overlay': overlay
        }
    
    def overlay_cam(self, 
                    image: np.ndarray, 
                    cam: np.ndarray, 
                    alpha: float = 0.4) -> np.ndarray:
        """
        Overlay CAM (probabilities) on original image.
        """
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap (JET color map for highlights)
        cam_colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # Convert image to RGB if needed
        if image.ndim == 2:
            image_rgb = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Overlay
        overlay = cv2.addWeighted(image_rgb, 1 - alpha, cam_colored, alpha, 0)
        return overlay

    def _visualize_explanation(self, 
                               image: np.ndarray, 
                               heatmap: np.ndarray, 
                               overlay: np.ndarray,
                               predicted_class: int,
                               confidence: float,
                               save_path: str):
        """
        Visualize and save explanation.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original DWI Slice')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='hot', vmin=0.0, vmax=1.0)
        axes[1].set_title('Lesion Probability Map')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        class_name = 'Lesion Detected' if predicted_class == 1 else 'No Lesion Detected'
        axes[2].set_title(f'{class_name}\nConfidence: {confidence:.3f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main function to test the explainer."""
    
    # Configuration
    config = {
        'model_path': r"d:\Stroke-DigitalTwin\outputs\segmentation_model.pth",
        'model_type': 'single',
        'model_name': 'unet',
        'modalities': ['DWI'],
        'device': 'cpu',
        'method': 'segmentation_overlay',
        'data_dir': r"d:\Stroke-DigitalTwin\outputs\preprocessed",
        'output_dir': r"d:\Stroke-DigitalTwin\outputs\explanations"
    }
    
    print("Explainability Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create explainer
    explainer = StrokeExplainer(
        model_path=config['model_path'],
        model_type=config['model_type'],
        model_name=config['model_name'],
        modalities=config['modalities'],
        device=config['device'],
        method=config['method']
    )
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Explain first patient
    data_dir = Path(config['data_dir'])
    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if patient_dirs:
        patient_dir = patient_dirs[0]
        print(f"\nExplaining patient {patient_dir.name}...")
        
        # Load image
        modality_dir = patient_dir / config['modalities'][0]
        slice_files = sorted(modality_dir.glob("*.png"))
        slices = []
        for slice_file in slice_files[:3]:
            img = cv2.imread(str(slice_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                slices.append(img)
        
        if slices:
            image = np.stack(slices, axis=0)
            
            # Generate explanation
            save_path = output_dir / f"explanation_{patient_dir.name}.png"
            result = explainer.explain_image(image, save_path=str(save_path))
            
            print(f"Explanation saved to {save_path}")
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.3f}")
    else:
        print(f"No patient data found in {data_dir}")


if __name__ == "__main__":
    main()

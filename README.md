# Stroke-DigitalTwin

An AI-based stroke detection and personalized prognosis system using Vision Transformers and Digital Twin technology.

## Project Overview

This project implements a complete pipeline for stroke detection and prognosis using:
- **Vision Transformer (ViT)** for MRI-based stroke classification
- **Digital Twin** technology for patient-specific modeling
- **Explainable AI** for model interpretation
- **Simulation** capabilities for treatment response prediction

## Project Structure

```
Stroke-DigitalTwin/
│
├── dataset/
│   └── ISLES2015_SISS/          # ISLES2015 stroke dataset
│
├── preprocessing/
│   ├── loader.py                # Dataset loading and verification
│   ├── visualize.py             # MRI visualization
│   └── preprocess.py            # Data preprocessing
│
├── models/
│   ├── dataset.py               # PyTorch dataset implementation
│   ├── vit.py                   # Vision Transformer model
│   ├── train.py                 # Training pipeline
│   └── predict.py               # Prediction interface
│
├── explainability/
│   └── gradcam.py               # Explainable AI (Grad-CAM, Attention Rollout)
│
├── digital_twin/
│   ├── twin.py                  # Digital Twin creation
│   └── simulation.py            # What-if analysis and simulation
│
├── outputs/                     # Output files
│   ├── checkpoints/             # Model checkpoints
│   ├── preprocessed/            # Preprocessed data
│   ├── visualizations/          # Visualization outputs
│   ├── digital_twins/           # Digital twin files
│   └── simulations/             # Simulation results
│
├── notebooks/                   # Jupyter notebooks
├── requirements.txt             # Python dependencies
├── main.py                      # Main entry point
└── README.md                    # This file
```

## Completed Modules

### Module 1: Dataset Loader (`preprocessing/loader.py`)
- Loads ISLES2015 dataset
- Detects MRI modalities (DWI, FLAIR, T1, T2)
- Loads lesion masks
- Verifies dataset integrity
- Generates summary reports

### Module 2: Visualization (`preprocessing/visualize.py`)
- Displays MRI slices
- Displays lesion masks
- Creates overlay visualizations
- Generates slice series views
- Multi-modality visualization

### Module 3: Preprocessing (`preprocessing/preprocess.py`)
- Normalizes MRI volumes (z-score, min-max)
- Resizes images to 224x224
- Extracts center slices or lesion-focused slices
- Saves processed images for model input

### Module 4: Vision Transformer (`models/vit.py`)
- Implements ViT for stroke classification
- Supports single and multi-modal inputs
- Uses pretrained models from timm library
- Custom classification heads
- Feature extraction capabilities

### Module 5: Training (`models/train.py`)
- Complete training pipeline
- Mixed precision training support
- Learning rate scheduling
- Gradient clipping
- Checkpoint saving
- Training curve visualization

### Module 6: Prediction (`models/predict.py`)
- Model inference interface
- Batch prediction support
- Prediction visualization
- Confidence scoring
- Multi-modal prediction support

### Module 7: Explainable AI (`explainability/gradcam.py`)
- Grad-CAM implementation
- Attention Rollout for ViT
- Gradient-based attention methods
- Visualization of model attention
- Model interpretation tools

### Module 8: Digital Twin (`digital_twin/twin.py`)
- Patient profile management
- Imaging feature extraction
- Digital twin creation and storage
- Patient clustering analysis
- PCA visualization
- Feature distribution analysis

### Module 9: Simulation (`digital_twin/simulation.py`)
- Treatment response simulation
- Lesion evolution modeling
- Modality importance analysis
- What-if scenario analysis
- Simulation visualization

## Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup
1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Complete Pipeline

Run the complete pipeline from data loading to simulation:

```bash
python main.py
```

### Individual Modules

#### Dataset Loading and Verification
```bash
python preprocessing/loader.py
```

#### Visualization
```bash
python preprocessing/visualize.py
```

#### Preprocessing
```bash
python preprocessing/preprocess.py
```

#### Training
```bash
python models/train.py
```

#### Prediction
```bash
python models/predict.py
```

#### Explainability
```bash
python explainability/gradcam.py
```

#### Digital Twin Creation
```bash
python digital_twin/twin.py
```

#### Simulation
```bash
python digital_twin/simulation.py
```

## Dataset

The project uses the ISLES2015 (Ischemic Stroke Lesion Segmentation) dataset. Place the dataset in the following structure:

```
dataset/
└── ISLES2015_SISS/
    ├── training/
    │   ├── 1/
    │   ├── 2/
    │   └── ...
    └── testing/
        ├── 1/
        ├── 2/
        └── ...
```

## Model Performance

### For Publication-Ready Results

**Important**: The initial 100% accuracy was due to using only 5 samples for training. For publication-quality results, use one of these approaches:

#### Option 1: Cross-Validation (Recommended)
```bash
python models/cross_validation.py
```
- Uses 5-fold stratified cross-validation
- Simpler CNN model (~2M parameters) for better generalization
- Provides mean ± std metrics across folds
- More robust for small datasets

#### Option 2: Proper Train/Val/Test Split
```bash
python models/train_test_split.py
```
- Uses separate ISLES2015 test set
- Proper train/validation split from training data
- Final evaluation on unseen test data
- Most realistic performance estimate

### Initial Results (Not for Publication)

The initial Vision Transformer model achieved:
- **Training Accuracy**: 100%
- **Validation Accuracy**: 100%
- **Model Parameters**: 86M (ViT-Base)
- **Input Size**: 224x224x3 (3 slices as channels)

**Note**: These results are not publication-ready due to:
- Small dataset size (5 samples)
- Potential overfitting
- Lack of proper validation
- No separate test set

## Key Features

### Vision Transformer
- Pretrained ViT-Base architecture
- Transfer learning from ImageNet
- Custom classification head for binary stroke detection
- Support for multi-modal MRI inputs

### Digital Twin
- Patient-specific modeling
- Imaging feature extraction
- Prediction integration
- Simulation history tracking

### Explainability
- Attention visualization
- Grad-CAM heatmaps
- Model interpretation tools
- Modality importance analysis

### Simulation
- Treatment response modeling
- Lesion evolution prediction
- What-if scenario analysis
- Outcome visualization

## Output Files

### Checkpoints
- `best_model.pth` - Best performing model
- `checkpoint_epoch_N.pth` - Periodic checkpoints
- `training_history.json` - Training metrics
- `training_curves.png` - Training visualization

### Preprocessed Data
- Patient-wise preprocessed MRI slices
- Normalized and resized images
- Ready for model input

### Digital Twins
- `digital_twin_{patient_id}.json` - Patient digital twins
- `pca_visualization.png` - Patient clustering
- Feature analysis reports

### Explainability
Generate a feature-based explanation for a saved patient prediction:
```bash
python explain_patient.py 1
```
This ranks the patient features that locally increase or decrease each model's selected output. It explains model sensitivity, not clinical causation.

### Important Model Limitation
Severity labels are derived from lesion-area thresholds, while recovery and recurrence labels are heuristic when real follow-up outcomes are unavailable. The reported metrics therefore measure performance against surrogate labels and should not be treated as clinical validation.

### Simulations
- Treatment response simulations
- Lesion evolution models
- Modality importance analysis
- Simulation visualizations

## Future Work

### Model Improvements
- Larger dataset training
- Ensemble methods
- 3D CNN integration
- Multi-task learning

### Clinical Integration
- DICOM support
- PACS integration
- Clinical validation
- Regulatory compliance

## Technical Details

### Model Architecture
- **Backbone**: ViT-Base (Patch16, 224)
- **Parameters**: 86M
- **Input**: 224x224x3 (3 MRI slices)
- **Output**: Binary classification (Stroke/No Stroke)

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Scheduler**: Cosine Annealing
- **Batch Size**: 2 (adjust based on GPU memory)
- **Epochs**: 10
- **Mixed Precision**: Enabled

### Preprocessing Pipeline
1. Load NIfTI files
2. Normalize (z-score)
3. Extract 3 center/lesion slices
4. Resize to 224x224
5. Stack as channels
6. Save as PNG

## Citation

If you use this code, please cite the ISLES2015 dataset:
```
ISLES Challenge: Ischemic Stroke Lesion Segmentation
https://www.isles-challenge.org/
```

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please open an issue on the project repository.

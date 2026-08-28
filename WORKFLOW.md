# Stroke Digital Twin ML Workflow

## Final Architecture (Stroke-Only Dataset)

```
ISLES 2015 SISS (28 patients)
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
       DWI         FLAIR         T2/T1
        │            │            │
        └────────────┼────────────┘
                     ↓
              Preprocessing
                     ↓
        ┌────────────┼────────────┐
        ↓            ↓            ↓
  Feature       Image
 Extraction    Augmentation
        │            │
        └────────────┼────────────┘
                     ↓
        ┌────────────┴────────────┐
        ↓                         ↓
  Feature-Based Models    Cross-Validation
        │                         │
        ┌────────┬────────┬───────┘
        ↓        ↓        ↓
  Severity  Recovery  Recurrence
Classifier Predictor  Predictor
(83.33%)  (0.05)   (83.33%)
        │        │        │
        └────────┼────────┘
                 ↓
        Model Integration
        (with Uncertainty)
                 ↓
        ┌────────┴────────┐
        ↓                 ↓
   Monte Carlo     95% CI
   Dropout         Estimation
        │                 │
        └────────┬────────┘
                 ↓
              Digital Twin
        (28 patients with predictions)
                 │
        ┌────────┴────────┐
        ↓                 ↓
   Explainability   Simulations
   (Grad-CAM)      (Treatment/Lesion)
```

## Component Descriptions

### 1. Data Preprocessing
- **Location**: `preprocessing/preprocess.py`
- **Function**: Normalization, resizing, slice extraction
- **Modalities**: DWI, FLAIR, T1, T2
- **Patients**: 28 patients (full dataset)

### 2. Feature Extraction
- **Location**: `digital_twin/twin.py`
- **Function**: Extract imaging features from preprocessed data
- **Features**: Intensity statistics (mean, std) for each modality, lesion area, lesion center coordinates
- **Dimensions**: 12 features per patient

### 3. Lesion Features
- **Location**: `digital_twin/twin.py`
- **Function**: Reads the lesion masks supplied with ISLES2015
- **Output**: Lesion area and center coordinates
- **Usage**: Severity, recovery, recurrence, and digital-twin features

### 4. Feature Augmentation
- **Location**: `models/feature_augmentation.py`
- **Function**: Gaussian noise, feature dropout, scaling, mixup
- **Purpose**: Improve generalization for small dataset
- **Application**: Training data augmentation (30% probability)

### 5. Severity Classification (Feature-Based)
- **Location**: `models/severity_feature_model.py`
- **Function**: Multi-class severity prediction (Mild/Moderate/Severe)
- **Architecture**: Neural network with hidden dims [64, 32]
- **Basis**: Lesion volume thresholds (677, 3502 pixels)
- **Performance**: 83.33% validation accuracy
- **Cross-Validation**: 92.67% ± 9.04% (5-fold)

### 6. Recovery Forecast Prediction
- **Location**: `models/recovery_model.py`
- **Function**: Predict recovery probability (0-1)
- **Architecture**: Neural network with hidden dims [128, 64, 32]
- **Basis**: Imaging features + patient profile
- **Performance**: 0.1047 validation loss
- **Cross-Validation**: 0.05 ± 0.05 loss (5-fold)

### 7. Recurrence Risk Prediction
- **Location**: `models/recurrence_model.py`
- **Function**: Predict recurrence risk (Low/Medium/High)
- **Architecture**: Neural network with hidden dims [64, 32]
- **Basis**: Lesion size, age, NIHSS score
- **Performance**: 83.33% validation accuracy
- **Cross-Validation**: 92.67% ± 9.04% (5-fold)
- **Regularization**: Increased dropout (0.5), reduced complexity

### 8. Cross-Validation
- **Location**: `models/cv_feature_models.py`
- **Function**: 5-fold stratified cross-validation
- **Purpose**: Robust model evaluation
- **Results**: Saved to `outputs/cv_results/`

### 9. Model Integration
- **Location**: `models/model_integration.py`
- **Function**: Load and use all trained models
- **Features**: Unified prediction interface
- **Uncertainty**: Monte Carlo dropout for confidence intervals

### 10. Uncertainty Estimation
- **Method**: Monte Carlo dropout (100 samples)
- **Output**: 95% confidence intervals for all predictions
- **Implementation**: Dropout enabled during inference, BatchNorm in eval mode

### 11. Digital Twin Generation
- **Location**: `digital_twin/twin.py`
- **Function**: Create patient digital twins with integrated predictions
- **Output**: 28 digital twins with severity, recovery, recurrence predictions + uncertainty
- **Format**: JSON files with patient profiles, features, and predictions

### 12. Explainability (Deferred)
- **Location**: `explainability/gradcam.py`
- **Status**: Deferred because the active feature-based models do not consume image tensors and automated UNet segmentation is not part of this workflow

### 13. Simulation & Analysis
- **Location**: `digital_twin/simulation.py`
- **Function**: Treatment simulation, lesion evolution, and modality importance using extracted lesion features
- **Output**: Simulation results and visualizations

### 14. Temporal Analysis
- **Location**: `models/temporal_analysis.py`
- **Function**: Compare baseline vs follow-up MRIs
- **Methods**: Image registration, lesion change metrics
- **Output**: Change analysis and visualizations

## Important Notes

### Binary Classification Deprecated
- **Reason**: Dataset contains only stroke patients (no healthy controls)
- **Action**: Binary stroke detection removed from workflow
- **Files Deprecated**: 
  - `models/train.py` → `models/train_binary_deprecated.py`
  - `models/train_multimodal.py` → `models/train_multimodal_deprecated.py`

### Feature-Based Approach
- **Severity Classification**: Switched from ViT to feature-based classifier (83.33% vs 66.67%)
- **Rationale**: Severity determined by lesion characteristics, not image patterns
- **Benefits**: Better accuracy, faster training, less overfitting

### Uncertainty Quantification
- **Method**: Monte Carlo dropout with 100 samples
- **Output**: 95% confidence intervals for all predictions
- **Implementation**: Dropout enabled during inference, BatchNorm in eval mode

## Running the Pipeline

### Step 1: Preprocess Data
```bash
python preprocessing/preprocess.py
```
- Input: Raw ISLES 2015 SISS data
- Output: `outputs/preprocessed_full/training/`

### Step 2: Train Models
```bash
# Severity Classification (Feature-Based)
python models/severity_feature_model.py

# Recovery Prediction (requires digital twins)
python models/recovery_model.py

# Recurrence Risk Prediction (requires digital twins)
python models/recurrence_model.py
```

### Step 3: Cross-Validation (Optional)
```bash
# Run 5-fold cross-validation for all models
python models/cv_feature_models.py
```
- Output: `outputs/cv_results/`

### Step 4: Create Digital Twins
```bash
python digital_twin/twin.py
```
- Input: Preprocessed data + trained models
- Output: 28 digital twins with predictions + uncertainty
- Location: `outputs/digital_twins/`

### Step 5: Analysis & Visualization
```bash
# Temporal Analysis
python models/temporal_analysis.py

# Simulations
python digital_twin/simulation.py

# Explainability
python explainability/gradcam.py
```

## Output Directories

- `outputs/preprocessed_full/` - Preprocessed MRI data (28 patients)
- `outputs/severity_feature_checkpoints/` - Severity model checkpoints
- `outputs/recovery_checkpoints/` - Recovery model checkpoints
- `outputs/recurrence_checkpoints/` - Recurrence model checkpoints
- `outputs/cv_results/` - Cross-validation results
- `outputs/digital_twins/` - Digital twin JSON files (28 patients)
- `outputs/explanations/` - Grad-CAM visualizations
- `outputs/simulations/` - Simulation results
- `outputs/temporal_analysis/` - Temporal analysis results

## Model Performance Summary

| Model | Type | Validation Accuracy | Cross-Validation | Key Features |
|-------|------|-------------------|------------------|--------------|
| Severity | Classification | 83.33% | 92.67% ± 9.04% | Feature-based, 12-dim input |
| Recovery | Regression | 0.1047 loss | 0.05 ± 0.05 loss | 15-dim input (imaging + profile) |
| Recurrence | Classification | 83.33% | 92.67% ± 9.04% | 15-dim input, high regularization |

## Key Improvements Made

1. **Severity Classification**: 66.67% → 83.33% (feature-based approach)
2. **Recurrence Overfitting**: 100% train vs 83% val → 90% train vs 83% val
3. **Cross-Validation**: Implemented 5-fold CV for robust evaluation
4. **Uncertainty Estimation**: Monte Carlo dropout with 95% CI
5. **Data Augmentation**: Feature augmentation for better generalization
6. **Model Integration**: Unified prediction interface with uncertainty

# Stroke-DigitalTwin Project Alignment Check

## Proposed Workflow (from diagram) vs. Current Implementation

| Stage | Component | Diagram Proposal | Current Status | Notes |
|-------|-----------|------------------|-----------------|-------|
| **1** | **INPUT DATA** | ISLES 2017 Dataset with DWI, ADC, Lesion Masks | ✅ IMPLEMENTED | Using ISLES2015 instead of 2017. Dataset contains only stroke-positive cases (28 training, 36 testing); **no healthy controls**. |
| **2** | **DATA PREPROCESSING** | Skull stripping, noise removal, normalization, resizing (224×224), augmentation | ✅ IMPLEMENTED | `preprocessing/loader.py`, `preprocess.py`, `visualize.py`. Covers normalization, resizing, augmentation. Skull stripping not explicitly implemented. |
| **3** | **MRI FEATURE EXTRACTION** | Vision Transformer (ViT) with patch embedding and transformer blocks | ✅ IMPLEMENTED | `models/vit.py`: `StrokeViT` and `MultiModalViT` using timm pretrained models. Extracts lesion features via ViT. |
| **4** | **STROKE PREDICTION MODEL** | Deep learning classifier for Stroke/Normal, Severity, Confidence | ⚠️ PARTIAL | `models/train.py`, `models/vit.py`. Binary classifier is not valid for the current single-class dataset. Current prognosis models use features from the provided lesion masks. |
| **5** | **EXPLAINABLE AI** | Grad-CAM & ViT attention maps to highlight stroke regions | ✅ IMPLEMENTED | `explainability/gradcam.py`: `GradCAM`, `ViTAttentionRollout`, `StrokeExplainer`. Heatmaps and attention visualization included. |
| **6** | **DIGITAL TWIN GENERATION** | Patient-specific modeling and profiling | ✅ IMPLEMENTED | `digital_twin/twin.py`: `DigitalTwin`, `PatientProfile`, `ImagingFeatures`, `DigitalTwinFactory`. Captures patient imaging features, predictions, and simulation history. |
| **7** | **SIMULATION & WHAT-IF ANALYSIS** | Lesion growth, reduction, recovery prediction scenarios | ✅ IMPLEMENTED | `digital_twin/simulation.py`: `TreatmentSimulator`, `LesionEvolutionSimulator`, `ModalityImportanceAnalyzer`, `WhatIfAnalyzer`. Full simulation framework in place. |
| **8** | **OUTPUT PREDICTIONS** | Stroke detection, severity, recovery forecast, recurrence risk, confidence scores | ⚠️ PARTIAL | `models/predict.py` for classification predictions. Segmentation predictions not yet integrated. Severity and recurrence risk not fully developed. |

---

## Critical Issues

### 1. **Dataset Limitation** (Blocks Stage 4)
- **Problem**: All 28 training patients are stroke-positive; no healthy controls exist.
- **Impact**: Binary stroke-vs-no-stroke classification is not meaningful (100% accuracy trivially).
- **Current approach**: Use the lesion masks supplied by ISLES2015 to extract features. Automated segmentation is deferred.

### 2. **Model Training Failure** (Stage 4)
- **Status**: `train.py` fails with `ValueError: training split contains only one class`.
- **Current**: Code includes a fail-fast guard to prevent misleading results.

---

## Alignment Summary

| Phase | Coverage |
|-------|----------|
| **Data Input & Preprocessing** (Stages 1-2) | ✅ ~90% complete |
| **Model & Feature Extraction** (Stages 3-4) | ⚠️ ~70% complete (classification unavailable; mask-based features used) |
| **Explainability** (Stage 5) | ✅ ~95% complete |
| **Digital Twin & Simulation** (Stages 6-7) | ✅ ~95% complete |
| **Prediction Output** (Stage 8) | ⚠️ ~50% complete (severity/risk still require validation) |

---

## Recommended Next Steps

1. **Extend dataset** - Add healthy control cases if binary stroke detection is required.
2. **Integrate severity/risk prediction** - Add stroke severity classification and recurrence risk scoring (Stage 8).
3. **End-to-end testing** - Verify the ML pipeline from preprocessing through predictions.

---

## Conclusion

The project implements **~70% of the proposed workflow** with strong coverage of preprocessing, feature extraction, explainability, and digital twin simulation. The main gaps are:
- Binary classification blocked by dataset limitation; current features use the provided lesion masks.
- Severity/risk predictions not yet developed.

**Current focus**: Validating the mask-based feature and prognosis pipeline while keeping automated segmentation deferred.

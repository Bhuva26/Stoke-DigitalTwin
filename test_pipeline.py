"""
Comprehensive test script for the Stroke Digital Twin pipeline.
Tests all components end-to-end.
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_data_preprocessing():
    """Test if preprocessed data exists."""
    print("="*60)
    print("TEST 1: Data Preprocessing")
    print("="*60)
    
    preprocessed_dir = Path("outputs/preprocessed_full/training")
    
    if not preprocessed_dir.exists():
        print("❌ Preprocessed data directory not found")
        return False
    
    patient_dirs = [d for d in preprocessed_dir.iterdir() if d.is_dir()]
    print(f"✓ Found {len(patient_dirs)} patient directories")
    
    if len(patient_dirs) < 28:
        print(f"⚠ Warning: Expected 28 patients, found {len(patient_dirs)}")
    
    # Check modalities for first patient
    if patient_dirs:
        first_patient = patient_dirs[0]
        modalities = ['DWI', 'FLAIR', 'T1', 'T2']
        found_modalities = []
        for mod in modalities:
            if (first_patient / mod).exists():
                found_modalities.append(mod)
        print(f"✓ Patient 1 has modalities: {found_modalities}")
    
    print("✓ Data preprocessing test passed\n")
    return True


def test_model_checkpoints():
    """Test if model checkpoints exist."""
    print("="*60)
    print("TEST 2: Model Checkpoints")
    print("="*60)
    
    checkpoints = {
        'Severity': Path("outputs/severity_feature_checkpoints/best_severity_feature_model.pth"),
        'Recovery': Path("outputs/recovery_checkpoints/best_recovery_model.pth"),
        'Recurrence': Path("outputs/recurrence_checkpoints/best_recurrence_model.pth")
    }
    
    all_exist = True
    for name, path in checkpoints.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ {name} checkpoint exists ({size_mb:.2f} MB)")
        else:
            print(f"❌ {name} checkpoint not found at {path}")
            all_exist = False
    
    if all_exist:
        print("✓ All model checkpoints test passed\n")
    else:
        print("❌ Model checkpoints test failed\n")
    
    return all_exist


def test_digital_twins():
    """Test if digital twins exist and have correct structure."""
    print("="*60)
    print("TEST 3: Digital Twins")
    print("="*60)
    
    twins_dir = Path("outputs/digital_twins")
    
    if not twins_dir.exists():
        print("❌ Digital twins directory not found")
        return False
    
    twin_files = sorted(twins_dir.glob("digital_twin_*.json"))
    print(f"✓ Found {len(twin_files)} digital twin files")
    
    if len(twin_files) < 28:
        print(f"⚠ Warning: Expected 28 twins, found {len(twin_files)}")
    
    # Check structure of first twin
    if twin_files:
        with open(twin_files[0], 'r') as f:
            twin_data = json.load(f)
        
        required_keys = ['patient_id', 'profile', 'imaging_features', 'model_predictions']
        missing_keys = [k for k in required_keys if k not in twin_data]
        
        if missing_keys:
            print(f"❌ Missing keys in digital twin: {missing_keys}")
            return False
        
        print(f"✓ Digital twin has required keys: {required_keys}")
        
        # Check model predictions
        if 'model_predictions' in twin_data:
            predictions = twin_data['model_predictions']
            required_predictions = ['severity', 'recovery', 'recurrence']
            missing_preds = [p for p in required_predictions if p not in predictions]
            
            if missing_preds:
                print(f"❌ Missing predictions: {missing_preds}")
                return False
            
            print(f"✓ Digital twin has predictions: {required_predictions}")
            
            # Check uncertainty intervals
            has_uncertainty = all(
                'uncertainty_intervals' in predictions[p] 
                for p in required_predictions
            )
            
            if has_uncertainty:
                print("✓ Predictions include uncertainty intervals")
            else:
                print("⚠ Warning: Predictions missing uncertainty intervals")
        
        print("✓ Digital twins test passed\n")
        return True
    
    print("❌ No digital twins found\n")
    return False


def test_cross_validation_results():
    """Test if cross-validation results exist."""
    print("="*60)
    print("TEST 4: Cross-Validation Results")
    print("="*60)
    
    cv_dir = Path("outputs/cv_results")
    
    if not cv_dir.exists():
        print("⚠ Cross-validation results directory not found (optional)")
        return True
    
    cv_files = {
        'Severity': cv_dir / "severity_cv_results.json",
        'Recovery': cv_dir / "recovery_cv_results.json",
        'Recurrence': cv_dir / "recurrence_cv_results.json"
    }
    
    all_exist = True
    for name, path in cv_files.items():
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"✓ {name} CV results exist")
            if 'metric_mean' in data:
                print(f"  Mean metric: {data['metric_mean']:.4f}")
            if 'metric_std' in data:
                print(f"  Std metric: {data['metric_std']:.4f}")
        else:
            print(f"⚠ {name} CV results not found (optional)")
    
    print("✓ Cross-validation results test passed\n")
    return True


def test_model_integration():
    """Test model integration by loading and making predictions."""
    print("="*60)
    print("TEST 5: Model Integration")
    print("="*60)
    
    try:
        import torch
        from models.model_integration import ModelIntegrator
        
        config = {
            'severity_model_path': "outputs/severity_feature_checkpoints/best_severity_feature_model.pth",
            'recovery_model_path': "outputs/recovery_checkpoints/best_recovery_model.pth",
            'recurrence_model_path': "outputs/recurrence_checkpoints/best_recurrence_model.pth",
            'device': 'cpu'
        }
        
        print("Loading ModelIntegrator...")
        integrator = ModelIntegrator(
            severity_model_path=config['severity_model_path'],
            recovery_model_path=config['recovery_model_path'],
            recurrence_model_path=config['recurrence_model_path'],
            device=config['device']
        )
        
        print("✓ ModelIntegrator loaded successfully")
        
        # Test prediction with dummy features
        dummy_imaging_features = {
            'dwi_intensity_mean': 0.5,
            'dwi_intensity_std': 1.0,
            'flair_intensity_mean': 0.5,
            'flair_intensity_std': 1.0,
            't1_intensity_mean': 0.5,
            't1_intensity_std': 1.0,
            't2_intensity_mean': 0.5,
            't2_intensity_std': 1.0,
            'lesion_area_pixels': 1000,
            'lesion_center_x': 100,
            'lesion_center_y': 100,
            'lesion_center_z': 1
        }
        
        dummy_profile_features = {
            'age': 65,
            'nihss_score': 10,
            'time_since_stroke': 7
        }
        
        print("Making test predictions...")
        predictions = integrator.predict_all(
            imaging_features=dummy_imaging_features,
            profile_features=dummy_profile_features
        )
        
        print("✓ Predictions generated successfully")
        
        # Check prediction structure
        required_keys = ['severity', 'recovery', 'recurrence']
        for key in required_keys:
            if key in predictions:
                print(f"✓ {key} prediction: {predictions[key]['class_name'] if 'class_name' in predictions[key] else 'N/A'}")
                if 'uncertainty_intervals' in predictions[key]:
                    print(f"  ✓ Has uncertainty intervals")
            else:
                print(f"❌ {key} prediction missing")
                return False
        
        print("✓ Model integration test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_augmentation():
    """Test feature augmentation module."""
    print("="*60)
    print("TEST 6: Feature Augmentation")
    print("="*60)
    
    try:
        from models.feature_augmentation import FeatureAugmenter
        
        augmenter = FeatureAugmenter(noise_std=0.05, dropout_rate=0.1, mixup_alpha=0.2)
        
        # Test with dummy features
        features = np.array([0.5, 0.3, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1000, 100, 100, 1])
        
        augmented = augmenter.augment(features, n_augmentations=3)
        
        print(f"✓ Created {len(augmented)} augmented versions")
        print(f"✓ Original shape: {features.shape}")
        print(f"✓ Augmented shape: {augmented[0].shape}")
        
        print("✓ Feature augmentation test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Feature augmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("STROKE DIGITAL TWIN PIPELINE TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        ("Data Preprocessing", test_data_preprocessing),
        ("Model Checkpoints", test_model_checkpoints),
        ("Digital Twins", test_digital_twins),
        ("Cross-Validation Results", test_cross_validation_results),
        ("Model Integration", test_model_integration),
        ("Feature Augmentation", test_feature_augmentation)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results[name] = False
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The pipeline is ready.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please review the output above.")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

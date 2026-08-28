"""
Module 1: Dataset Loader
This module loads the ISLES2015 dataset, detects MRI modalities, loads lesion masks, and verifies the dataset.
"""

import os
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


def resolve_dataset_path(dataset_path: Optional[str] = None) -> Path:
    """Resolve the ISLES dataset root from either a direct path or the repository layout."""
    repo_root = Path(__file__).resolve().parent.parent

    candidates = []
    if dataset_path:
        candidates.append(Path(dataset_path).expanduser())

    candidates.extend([
        repo_root / "dataset" / "ISLES2015" / "ISLES2015_SISS",
        repo_root / "dataset" / "ISLES2015" / "ISLES2015_SPES",
        repo_root / "dataset" / "ISLES2015_SISS",
        repo_root / "dataset" / "ISLES2015_SPES",
        repo_root / "dataset" / "ISLES2015",
        repo_root / "dataset",
        repo_root,
    ])

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.exists():
            continue

        if (resolved / "training").exists() and (resolved / "testing").exists():
            return resolved
        if (resolved / "ISLES2015_SISS").exists() and (resolved / "ISLES2015_SISS" / "training").exists():
            return resolved / "ISLES2015_SISS"
        if (resolved / "ISLES2015_SPES").exists() and (resolved / "ISLES2015_SPES" / "training").exists():
            return resolved / "ISLES2015_SPES"

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"Could not find an ISLES2015 dataset path. Checked: {', '.join(str(c) for c in candidates)}"
    )


class ISLES2015Loader:
    """Loader for ISLES2015 stroke dataset."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path: Path to the ISLES2015_SISS directory
        """
        self.dataset_path = resolve_dataset_path(dataset_path)
        self.modalities = ['DWI', 'Flair', 'T1', 'T2']
        self.mask_suffix = 'OT'
        self.patients_data = {}
        
    def get_patient_folders(self, split: str = 'training') -> List[Path]:
        """
        Get all patient folders for a given split.
        
        Args:
            split: 'training' or 'testing'
            
        Returns:
            List of patient folder paths
        """
        split_path = self.dataset_path / split
        if not split_path.exists():
            raise ValueError(f"Split path {split_path} does not exist")
        
        patient_folders = [f for f in split_path.iterdir() if f.is_dir() and not f.name.startswith('.')]
        return sorted(patient_folders, key=lambda x: int(x.name) if x.name.isdigit() else x.name)
    
    def detect_modalities(self, patient_folder: Path) -> Dict[str, Optional[Path]]:
        """
        Detect available MRI modalities and lesion mask for a patient.
        
        Args:
            patient_folder: Path to patient folder
            
        Returns:
            Dictionary mapping modality names to their file paths
        """
        modality_files = {}
        
        # Search for each modality
        for modality in self.modalities:
            # Look for folders containing the modality name
            for item in patient_folder.iterdir():
                if item.is_dir() and modality.upper() in item.name.upper():
                    # Find the .nii file inside
                    for file in item.iterdir():
                        if file.suffix in ['.nii', '.nii.gz']:
                            modality_files[modality] = file
                            break
                    break
        
        # Search for lesion mask (OT)
        for item in patient_folder.iterdir():
            if item.is_dir() and self.mask_suffix in item.name.upper():
                for file in item.iterdir():
                    if file.suffix in ['.nii', '.nii.gz']:
                        modality_files['mask'] = file
                        break
                break
        
        return modality_files
    
    def load_nifti(self, file_path: Path) -> np.ndarray:
        """
        Load a NIfTI file and return the data array.
        
        Args:
            file_path: Path to NIfTI file
            
        Returns:
            3D numpy array of the image data
        """
        try:
            img = nib.load(str(file_path))
            data = img.get_fdata()
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_patient(self, patient_folder: Path, load_data: bool = False) -> Dict:
        """
        Load all available data for a single patient.
        
        Args:
            patient_folder: Path to patient folder
            load_data: If True, load actual image data. If False, only load metadata.
            
        Returns:
            Dictionary containing patient data
        """
        patient_id = patient_folder.name
        modality_files = self.detect_modalities(patient_folder)
        
        patient_data = {
            'patient_id': patient_id,
            'folder': str(patient_folder),
            'modalities': {},
            'mask': None,
            'shape': None,
            'missing_modalities': []
        }
        
        # Load each modality
        for modality in self.modalities:
            if modality in modality_files:
                if load_data:
                    data = self.load_nifti(modality_files[modality])
                    if data is not None:
                        patient_data['modalities'][modality] = {
                            'file': str(modality_files[modality]),
                            'shape': data.shape,
                            'data': data
                        }
                        if patient_data['shape'] is None:
                            patient_data['shape'] = data.shape
                    else:
                        patient_data['missing_modalities'].append(modality)
                else:
                    # Only load metadata
                    try:
                        img = nib.load(str(modality_files[modality]))
                        shape = img.shape
                        patient_data['modalities'][modality] = {
                            'file': str(modality_files[modality]),
                            'shape': shape,
                            'data': None
                        }
                        if patient_data['shape'] is None:
                            patient_data['shape'] = shape
                    except Exception as e:
                        print(f"Error loading metadata for {modality_files[modality]}: {e}")
                        patient_data['missing_modalities'].append(modality)
            else:
                patient_data['missing_modalities'].append(modality)
        
        # Load lesion mask
        if 'mask' in modality_files:
            if load_data:
                mask_data = self.load_nifti(modality_files['mask'])
                if mask_data is not None:
                    patient_data['mask'] = {
                        'file': str(modality_files['mask']),
                        'shape': mask_data.shape,
                        'data': mask_data
                    }
                else:
                    patient_data['missing_modalities'].append('mask')
            else:
                # Only load metadata
                try:
                    img = nib.load(str(modality_files['mask']))
                    shape = img.shape
                    patient_data['mask'] = {
                        'file': str(modality_files['mask']),
                        'shape': shape,
                        'data': None
                    }
                except Exception as e:
                    print(f"Error loading mask metadata for {modality_files['mask']}: {e}")
                    patient_data['missing_modalities'].append('mask')
        else:
            patient_data['missing_modalities'].append('mask')
        
        return patient_data
    
    def load_dataset(self, split: str = 'training', load_data: bool = False) -> Dict:
        """
        Load the entire dataset for a given split.
        
        Args:
            split: 'training' or 'testing'
            load_data: If True, load actual image data. If False, only load metadata.
            
        Returns:
            Dictionary containing all patient data
        """
        print(f"Loading {split} dataset (load_data={load_data})...")
        patient_folders = self.get_patient_folders(split)
        
        dataset = {
            'split': split,
            'num_patients': len(patient_folders),
            'patients': []
        }
        
        for patient_folder in patient_folders:
            print(f"Loading patient {patient_folder.name}...")
            patient_data = self.load_patient(patient_folder, load_data=load_data)
            dataset['patients'].append(patient_data)
        
        return dataset
    
    def verify_dataset(self, dataset: Dict) -> Dict:
        """
        Verify the dataset and generate a summary report.
        
        Args:
            dataset: Dataset dictionary from load_dataset
            
        Returns:
            Verification report
        """
        report = {
            'total_patients': dataset['num_patients'],
            'patients_with_all_modalities': 0,
            'patients_with_mask': 0,
            'modality_counts': {mod: 0 for mod in self.modalities},
            'mask_count': 0,
            'shape_distribution': {},
            'issues': []
        }
        
        for patient in dataset['patients']:
            # Check modalities
            available_modalities = list(patient['modalities'].keys())
            missing = patient['missing_modalities']
            
            if len(available_modalities) == len(self.modalities):
                report['patients_with_all_modalities'] += 1
            
            for modality in self.modalities:
                if modality in available_modalities:
                    report['modality_counts'][modality] += 1
            
            # Check mask
            if patient['mask'] is not None:
                report['patients_with_mask'] += 1
                report['mask_count'] += 1
            
            # Track shapes
            if patient['shape']:
                shape_str = str(patient['shape'])
                report['shape_distribution'][shape_str] = report['shape_distribution'].get(shape_str, 0) + 1
            
            # Log issues
            if missing:
                report['issues'].append({
                    'patient_id': patient['patient_id'],
                    'missing': missing
                })
        
        return report
    
    def print_summary(self, report: Dict):
        """
        Print a formatted summary of the dataset verification.
        
        Args:
            report: Verification report from verify_dataset
        """
        print("\n" + "="*60)
        print("DATASET VERIFICATION SUMMARY")
        print("="*60)
        print(f"Total patients: {report['total_patients']}")
        print(f"Patients with all modalities: {report['patients_with_all_modalities']}")
        print(f"Patients with lesion mask: {report['patients_with_mask']}")
        print("\nModality counts:")
        for modality, count in report['modality_counts'].items():
            print(f"  {modality}: {count}")
        print(f"  Mask: {report['mask_count']}")
        print("\nShape distribution:")
        for shape, count in report['shape_distribution'].items():
            print(f"  {shape}: {count} patients")
        
        if report['issues']:
            print(f"\nIssues found: {len(report['issues'])}")
            for issue in report['issues'][:5]:  # Show first 5 issues
                print(f"  Patient {issue['patient_id']}: missing {issue['missing']}")
            if len(report['issues']) > 5:
                print(f"  ... and {len(report['issues']) - 5} more")
        else:
            print("\nNo issues found!")
        
        print("="*60 + "\n")
    
    def save_summary(self, report: Dict, output_path: str):
        """
        Save the verification report to a JSON file.
        
        Args:
            report: Verification report
            output_path: Path to save the JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Summary saved to {output_path}")


def main():
    """Main function to test the dataset loader."""
    
    # Path configuration
    dataset_path = str(resolve_dataset_path(r"d:\Stroke-DigitalTwin\dataset"))
    
    # Initialize loader
    loader = ISLES2015Loader(dataset_path)
    
    # Load training dataset
    dataset = loader.load_dataset(split='training')
    
    # Verify dataset
    report = loader.verify_dataset(dataset)
    
    # Print summary
    loader.print_summary(report)
    
    # Save summary
    output_dir = Path(r"d:\Stroke-DigitalTwin\outputs")
    output_dir.mkdir(exist_ok=True)
    loader.save_summary(report, output_dir / "dataset_summary.json")


if __name__ == "__main__":
    main()

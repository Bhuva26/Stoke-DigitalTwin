"""
Check dataset composition and labels.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from models.dataset import StrokeDataset, validate_label_distribution
from preprocessing.loader import ISLES2015Loader, resolve_dataset_path

# Load preprocessed dataset
preprocessed_dir = r"d:\Stroke-DigitalTwin\outputs\preprocessed"
dataset = StrokeDataset(
    data_dir=preprocessed_dir,
    split='training',
    modalities=['DWI'],
    transform=None
)

print(f"Total preprocessed patients: {len(dataset)}")
print(f"Patient IDs: {dataset.patients}")

labels = []
for i in range(len(dataset)):
    sample = dataset[i]
    labels.append(sample['label'].item())
    print(f"Patient {dataset.patients[i]}: Label = {sample['label'].item()}")

import numpy as np
counts = np.bincount(labels)
print(f"\nLabel distribution: {counts}")
print(f"Class 0 (No Stroke): {counts[0] if len(counts) > 0 else 0}")
print(f"Class 1 (Stroke): {counts[1] if len(counts) > 1 else 0}")
try:
    validate_label_distribution(dataset, 'preprocessed dataset')
except ValueError as exc:
    print(f"\n{exc}")

# Check the raw uploaded dataset for healthy/non-lesion cases
raw_dataset_path = resolve_dataset_path(r"d:\Stroke-DigitalTwin\dataset")
loader = ISLES2015Loader(str(raw_dataset_path))
print(f"\nResolved raw dataset root: {raw_dataset_path}")
for split in ['training', 'testing']:
    patient_folders = loader.get_patient_folders(split=split)
    lesion_cases = 0
    missing_ground_truth = 0
    for patient_folder in patient_folders:
        lesion_files = [
            item for item in patient_folder.iterdir()
            if item.is_dir() and 'OT' in item.name.upper()
        ]
        if lesion_files:
            lesion_cases += 1
        else:
            missing_ground_truth += 1
    print(f"{split}: patients={len(patient_folders)}, lesion_cases={lesion_cases}, missing_ground_truth={missing_ground_truth}")

healthy_keywords = ['healthy', 'control', 'normal', 'nonstroke', 'non-stroke']
healthy_matches = []
for path in raw_dataset_path.rglob('*'):
    if path.is_dir() and any(keyword in path.name.lower() for keyword in healthy_keywords):
        healthy_matches.append(str(path))

if healthy_matches:
    print(f"Healthy/control-like folders found: {healthy_matches[:10]}")
else:
    print("No healthy/control-like folders were found in the uploaded dataset.")

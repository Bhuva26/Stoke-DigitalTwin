"""
Analyze severity distribution and adjust thresholds.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from severity_model import SeverityDataset

# Load dataset
ds = SeverityDataset(
    r"d:\Stroke-DigitalTwin\outputs\preprocessed_full\training",
    modalities=['DWI'],
    transform=None
)

print(f"Current severity distribution:")
print(f"Mild (<1000): {sum(ds.severity_labels==0)}")
print(f"Moderate (1000-5000): {sum(ds.severity_labels==1)}")
print(f"Severe (>5000): {sum(ds.severity_labels==2)}")

# Calculate actual lesion areas
areas = []
for i in range(len(ds)):
    patient_dir = ds.data_dir / ds.patients[i]
    mask_dir = patient_dir / 'mask'
    if mask_dir.exists():
        total = 0
        for f in mask_dir.glob('*.png'):
            img = np.array(plt.imread(str(f)))
            if img.ndim == 3:
                img = img[:, :, 0]
            total += np.sum(img > 0)
        areas.append(total)

print(f"\nLesion area statistics:")
print(f"Min: {min(areas)}")
print(f"Max: {max(areas)}")
print(f"Mean: {np.mean(areas):.0f}")
print(f"Median: {np.median(areas):.0f}")
print(f"Std: {np.std(areas):.0f}")

# Suggest new thresholds based on percentiles
areas_sorted = sorted(areas)
p33 = areas_sorted[int(len(areas) * 0.33)]
p66 = areas_sorted[int(len(areas) * 0.66)]

print(f"\nSuggested thresholds based on percentiles:")
print(f"33rd percentile (Mild/Moderate): {p33}")
print(f"66th percentile (Moderate/Severe): {p66}")

# Test with new thresholds
new_mild_threshold = p33
new_moderate_threshold = p66

new_labels = []
for area in areas:
    if area < new_mild_threshold:
        new_labels.append(0)
    elif area < new_moderate_threshold:
        new_labels.append(1)
    else:
        new_labels.append(2)

print(f"\nNew severity distribution with adjusted thresholds:")
print(f"Mild (<{new_mild_threshold}): {sum(np.array(new_labels)==0)}")
print(f"Moderate ({new_mild_threshold}-{new_moderate_threshold}): {sum(np.array(new_labels)==1)}")
print(f"Severe (>{new_moderate_threshold}): {sum(np.array(new_labels)==2)}")

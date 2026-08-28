"""Feature-based explanations for a saved digital twin prediction."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from models.model_integration import ModelIntegrator


IMAGING_FEATURE_NAMES = [
    "dwi_intensity_mean", "dwi_intensity_std",
    "flair_intensity_mean", "flair_intensity_std",
    "t1_intensity_mean", "t1_intensity_std",
    "t2_intensity_mean", "t2_intensity_std",
    "lesion_area_pixels", "lesion_center_x",
    "lesion_center_y", "lesion_center_z",
]
PROFILE_FEATURE_NAMES = ["age", "nihss_score", "time_since_stroke"]


def _score(model: torch.nn.Module, features: np.ndarray, classification: bool, target: int = 0) -> float:
    """Return a model score for one feature vector."""
    tensor = torch.from_numpy(features).float().unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        if classification:
            return float(torch.softmax(output, dim=1)[0, target].item())
        return float(output.squeeze().item())


def _attributions(
    model: torch.nn.Module,
    features: np.ndarray,
    names: List[str],
    classification: bool,
    target: int = 0,
) -> List[Dict]:
    """Estimate local feature effects using symmetric perturbations."""
    model.eval()
    baseline = _score(model, features, classification, target)
    results = []

    for index, name in enumerate(names):
        step = max(abs(float(features[index])) * 0.10, 0.01)
        plus = features.copy()
        minus = features.copy()
        plus[index] += step
        minus[index] = max(0.0, minus[index] - step) if name in {"age", "nihss_score", "time_since_stroke", "lesion_area_pixels"} else minus[index] - step
        plus_score = _score(model, plus, classification, target)
        minus_score = _score(model, minus, classification, target)
        effect = (plus_score - minus_score) / 2.0
        results.append({
            "feature": name,
            "value": float(features[index]),
            "step": step,
            "score_minus": minus_score,
            "score_plus": plus_score,
            "effect": effect,
            "direction": "increases" if effect >= 0 else "decreases",
            "baseline_score": baseline,
        })

    return sorted(results, key=lambda item: abs(item["effect"]), reverse=True)


def explain_patient(patient_id: str, twin_path: Path, model_dir: Path) -> Dict[str, List[Dict]]:
    """Create local sensitivity explanations for one saved patient."""
    with twin_path.open() as file:
        twin = json.load(file)

    output_dir = model_dir / "outputs"
    integrator = ModelIntegrator(
        severity_model_path=str(output_dir / "severity_feature_checkpoints" / "best_severity_feature_model.pth"),
        recovery_model_path=str(output_dir / "recovery_checkpoints" / "best_recovery_model.pth"),
        recurrence_model_path=str(output_dir / "recurrence_checkpoints" / "best_recurrence_model.pth"),
        device="cpu",
    )

    imaging = twin.get("imaging_features", {})
    profile = twin.get("profile", {})
    imaging_values = np.array([imaging.get(name, 0.0) or 0.0 for name in IMAGING_FEATURE_NAMES], dtype=np.float32)
    profile_values = np.array([profile.get(name, 0.0) or 0.0 for name in PROFILE_FEATURE_NAMES], dtype=np.float32)
    combined_values = np.concatenate([imaging_values, profile_values])
    combined_names = IMAGING_FEATURE_NAMES + PROFILE_FEATURE_NAMES

    results = {
        "severity": _attributions(
            integrator.severity_model,
            imaging_values,
            IMAGING_FEATURE_NAMES,
            classification=True,
            target=twin.get("model_predictions", {}).get("severity", {}).get("class", 0),
        ),
        "recovery": _attributions(
            integrator.recovery_model,
            imaging_values,
            IMAGING_FEATURE_NAMES,
            classification=False,
        ),
    }

    if twin.get("model_predictions", {}).get("recurrence") and any(profile_values):
        results["recurrence"] = _attributions(
            integrator.recurrence_model,
            combined_values,
            combined_names,
            classification=True,
            target=twin["model_predictions"]["recurrence"].get("class", 0),
        )
    else:
        results["recurrence_note"] = [
            {"message": "Recurrence explanation unavailable: age, NIHSS score, or time since stroke is missing."}
        ]

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain a patient's feature-based predictions.")
    parser.add_argument("patient_id", help="Patient ID, for example 1")
    args = parser.parse_args()
    root = Path(__file__).parent
    twin_path = root / "outputs" / "digital_twins" / f"digital_twin_{args.patient_id}.json"
    if not twin_path.exists():
        raise SystemExit(f"Digital twin not found: {twin_path}")

    explanations = explain_patient(args.patient_id, twin_path, root)
    print(f"Feature explanation for Patient {args.patient_id}")
    print("Local sensitivity only; this shows model response, not clinical causation.\n")
    for model_name, items in explanations.items():
        if model_name == "recurrence_note":
            print(items[0]["message"])
            continue
        print(f"{model_name.title()} factors:")
        for item in items[:5]:
            print(
                f"  {item['feature']}={item['value']:.4g}: "
                f"{item['direction']} target score by {item['effect']:+.4f} "
                f"(minus={item['score_minus']:.4f}, plus={item['score_plus']:.4f})"
            )
        print()


if __name__ == "__main__":
    main()

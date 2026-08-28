"""
Interpret digital twin results in user-friendly language.
Translates technical predictions into understandable explanations.
"""

import json
from pathlib import Path
from typing import Dict


class ResultInterpreter:
    """Interpret digital twin predictions for non-technical users."""
    
    def __init__(self, digital_twin_path: str):
        """Load digital twin file."""
        with open(digital_twin_path, 'r') as f:
            self.twin = json.load(f)
    
    def interpret_severity(self) -> Dict:
        """Interpret severity prediction."""
        severity = self.twin['model_predictions'].get('severity', {})
        
        class_name = severity.get('class_name', 'Unknown')
        confidence = severity.get('confidence', 0)
        uncertainty = severity.get('uncertainty_intervals', {}).get(class_name, {})
        ci_95 = uncertainty.get('ci_95', (0, 0))
        
        # Plain language interpretation
        interpretations = {
            'Mild': {
                'description': "The stroke damage is relatively mild.",
                'impact': "Brain function is likely to be mostly preserved.",
                'recovery_outlook': "Good recovery prospects with proper treatment.",
                'daily_life': "Most daily activities should be manageable.",
                'medical_attention': "Regular follow-ups recommended, but not intensive care."
            },
            'Moderate': {
                'description': "The stroke damage is moderate in severity.",
                'impact': "Some brain function may be affected.",
                'recovery_outlook': "Recovery is possible but may require rehabilitation.",
                'daily_life': "Some daily activities may be challenging.",
                'medical_attention': "Regular medical monitoring and rehabilitation therapy recommended."
            },
            'Severe': {
                'description': "The stroke damage is severe.",
                'impact': "Significant brain function may be affected.",
                'recovery_outlook': "Recovery may be challenging and require extensive rehabilitation.",
                'daily_life': "Many daily activities may require assistance.",
                'medical_attention': "Intensive medical care and long-term rehabilitation recommended."
            }
        }
        
        interpretation = interpretations.get(class_name, interpretations['Mild'])
        
        return {
            'severity_level': class_name,
            'confidence_percent': f"{confidence * 100:.1f}%",
            'confidence_range': f"{ci_95[0] * 100:.1f}% to {ci_95[1] * 100:.1f}%",
            'plain_explanation': interpretation['description'],
            'brain_impact': interpretation['impact'],
            'recovery_outlook': interpretation['recovery_outlook'],
            'daily_life': interpretation['daily_life'],
            'medical_attention': interpretation['medical_attention']
        }
    
    def interpret_recovery(self) -> Dict:
        """Interpret recovery prediction."""
        recovery = self.twin['model_predictions'].get('recovery', {})
        
        probability = recovery.get('probability', 0.5)
        interpretation_text = recovery.get('interpretation', 'Unknown')
        uncertainty = recovery.get('uncertainty_intervals', {})
        ci_95 = uncertainty.get('ci_95', (0, 0))
        
        # Plain language interpretation
        if probability > 0.7:
            recovery_level = "Good"
            explanation = "High likelihood of good recovery."
            timeline = "Recovery expected within 3-6 months with proper care."
            therapy = "Standard rehabilitation therapy should be sufficient."
        elif probability > 0.4:
            recovery_level = "Moderate"
            explanation = "Moderate recovery expected."
            timeline = "Recovery may take 6-12 months with dedicated rehabilitation."
            therapy = "Regular rehabilitation therapy and consistent effort needed."
        else:
            recovery_level = "Challenging"
            explanation = "Recovery may be challenging."
            timeline = "Recovery may take 12+ months with intensive rehabilitation."
            therapy = "Intensive and long-term rehabilitation therapy required."
        
        return {
            'recovery_probability': f"{probability * 100:.1f}%",
            'probability_range': f"{ci_95[0] * 100:.1f}% to {ci_95[1] * 100:.1f}%",
            'recovery_level': recovery_level,
            'plain_explanation': explanation,
            'expected_timeline': timeline,
            'therapy_needs': therapy
        }
    
    def interpret_recurrence(self) -> Dict:
        """Interpret recurrence risk prediction."""
        recurrence = self.twin['model_predictions'].get('recurrence', {})
        
        class_name = recurrence.get('class_name', 'Unknown')
        confidence = recurrence.get('confidence', 0)
        uncertainty = recurrence.get('uncertainty_intervals', {}).get(class_name, {})
        ci_95 = uncertainty.get('ci_95', (0, 0))
        
        # Plain language interpretation
        interpretations = {
            'Low': {
                'description': "Low risk of another stroke.",
                'likelihood': "Less than 20% chance of recurrence.",
                'prevention': "Maintain healthy lifestyle and regular check-ups.",
                'monitoring': "Annual check-ups recommended.",
                'lifestyle': "Continue current healthy habits."
            },
            'Medium': {
                'description': "Moderate risk of another stroke.",
                'likelihood': "20-50% chance of recurrence.",
                'prevention': "Lifestyle changes and medication may be needed.",
                'monitoring': "Regular check-ups every 6 months recommended.",
                'lifestyle': "Focus on diet, exercise, and stress management."
            },
            'High': {
                'description': "High risk of another stroke.",
                'likelihood': "More than 50% chance of recurrence.",
                'prevention': "Aggressive prevention measures and close monitoring needed.",
                'monitoring': "Frequent check-ups every 3 months recommended.",
                'lifestyle': "Strict lifestyle changes and medication adherence required."
            }
        }
        
        interpretation = interpretations.get(class_name, interpretations['Low'])
        
        return {
            'recurrence_risk': class_name,
            'confidence_percent': f"{confidence * 100:.1f}%",
            'confidence_range': f"{ci_95[0] * 100:.1f}% to {ci_95[1] * 100:.1f}%",
            'plain_explanation': interpretation['description'],
            'likelihood': interpretation['likelihood'],
            'prevention_measures': interpretation['prevention'],
            'monitoring_needs': interpretation['monitoring'],
            'lifestyle_changes': interpretation['lifestyle']
        }
    
    def interpret_imaging_features(self) -> Dict:
        """Interpret imaging features in plain language."""
        features = self.twin.get('imaging_features', {})
        
        lesion_area = features.get('lesion_area_pixels', 0)
        
        # Convert to approximate mm² (assuming 1mm pixel spacing)
        lesion_area_mm2 = lesion_area
        
        # Plain language interpretation
        if lesion_area_mm2 < 1000:
            size_description = "small"
            impact = "Limited brain tissue affected"
        elif lesion_area_mm2 < 5000:
            size_description = "moderate"
            impact = "Moderate amount of brain tissue affected"
        else:
            size_description = "large"
            impact = "Significant amount of brain tissue affected"
        
        return {
            'lesion_size': size_description,
            'lesion_area': f"{lesion_area_mm2:.0f} pixels",
            'brain_impact': impact,
            'location': f"Center at ({features.get('lesion_center_x', 0):.0f}, {features.get('lesion_center_y', 0):.0f})"
        }
    
    def generate_report(self) -> str:
        """Generate a user-friendly report."""
        patient_id = self.twin.get('patient_id', 'Unknown')
        
        severity = self.interpret_severity()
        recovery = self.interpret_recovery()
        recurrence = self.interpret_recurrence()
        imaging = self.interpret_imaging_features()
        
        report = f"""
{'='*70}
STROKE DIGITAL TWIN REPORT - Patient {patient_id}
{'='*70}

BRAIN IMAGING ANALYSIS
{'-'*70}
- Lesion Size: {imaging['lesion_size']} ({imaging['lesion_area']})
- Impact: {imaging['brain_impact']}
- Location: {imaging['location']}

SEVERITY ASSESSMENT
{'-'*70}
- Severity Level: {severity['severity_level']}
- Confidence: {severity['confidence_percent']} (range: {severity['confidence_range']})
- What this means: {severity['plain_explanation']}
- Brain Impact: {severity['brain_impact']}
- Recovery Outlook: {severity['recovery_outlook']}
- Daily Life: {severity['daily_life']}
- Medical Attention: {severity['medical_attention']}

RECOVERY PREDICTION
{'-'*70}
- Recovery Probability: {recovery['recovery_probability']} (range: {recovery['probability_range']})
- Recovery Level: {recovery['recovery_level']}
- What this means: {recovery['plain_explanation']}
- Expected Timeline: {recovery['expected_timeline']}
- Therapy Needs: {recovery['therapy_needs']}

RECURRENCE RISK
{'-'*70}
- Risk Level: {recurrence['recurrence_risk']}
- Confidence: {recurrence['confidence_percent']} (range: {recurrence['confidence_range']})
- What this means: {recurrence['plain_explanation']}
- Likelihood: {recurrence['likelihood']}
- Prevention: {recurrence['prevention_measures']}
- Monitoring: {recurrence['monitoring_needs']}
- Lifestyle: {recurrence['lifestyle_changes']}

{'='*70}
IMPORTANT: This report is based on AI predictions and should be used
as a supplementary tool. Always consult with your healthcare provider
for medical decisions.
{'='*70}
"""
        return report


def main():
    """Generate user-friendly reports for all digital twins."""
    twins_dir = Path("outputs/digital_twins")
    
    if not twins_dir.exists():
        print("Digital twins directory not found")
        return
    
    twin_files = sorted(twins_dir.glob("digital_twin_*.json"))
    
    print(f"Found {len(twin_files)} digital twins\n")
    
    # Generate report for first patient as example
    if twin_files:
        print("Generating report for first patient (Patient 1)...\n")
        
        interpreter = ResultInterpreter(twin_files[0])
        report = interpreter.generate_report()
        
        print(report)
        
        # Save report to file
        report_path = Path("outputs/digital_twins") / f"patient_report_{twin_files[0].stem.replace('digital_twin_', '')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {report_path}")
    
    # Ask if user wants to see another patient
    print("\n" + "="*70)
    print("To view reports for other patients, run:")
    print("python interpret_results.py <patient_id>")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Generate report for specific patient
        patient_id = sys.argv[1]
        twin_path = Path(f"outputs/digital_twins/digital_twin_{patient_id}.json")
        
        if not twin_path.exists():
            print(f"Digital twin for patient {patient_id} not found")
            print(f"Available patients: {[f.stem.replace('digital_twin_', '') for f in Path('outputs/digital_twins').glob('digital_twin_*.json')]}")
        else:
            interpreter = ResultInterpreter(twin_path)
            report = interpreter.generate_report()
            print(report)
            
            report_path = Path("outputs/digital_twins") / f"patient_report_{patient_id}.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {report_path}")
    else:
        main()

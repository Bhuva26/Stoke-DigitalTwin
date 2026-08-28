"""
Module 9: Simulation
This module implements what-if analysis and simulation for digital twins based on lesion segmentation.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import matplotlib.pyplot as plt
import sys
try:
    from .twin import DigitalTwin, PatientProfile
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from digital_twin.twin import DigitalTwin, PatientProfile


class SimulationType(Enum):
    """Types of simulations for lesion segmentation."""
    LESION_GROWTH = "lesion_growth"
    LESION_SHRINKAGE = "lesion_shrinkage"
    TREATMENT_RESPONSE = "treatment_response"
    MODALITY_COMPARISON = "modality_comparison"
    SEGMENTATION_QUALITY = "segmentation_quality"


@dataclass
class SimulationConfig:
    """Configuration for simulation."""
    simulation_type: SimulationType
    parameters: Dict
    num_steps: int = 10
    time_horizon: float = 24.0  # hours


@dataclass
class SimulationResult:
    """Result of a simulation."""
    simulation_id: str
    config: SimulationConfig
    outcomes: List[Dict]
    metrics: Dict
    timestamp: str


class TreatmentSimulator:
    """Simulator for treatment response."""
    
    def __init__(self, predictor):
        """
        Initialize treatment simulator.
        
        Args:
            predictor: Stroke predictor instance
        """
        self.predictor = predictor
    
    def simulate_treatment(self, 
                          digital_twin: DigitalTwin,
                          treatment_type: str,
                          efficacy: float = 0.7) -> SimulationResult:
        """
        Simulate treatment response.
        
        Args:
            digital_twin: Patient digital twin
            treatment_type: Type of treatment (e.g., 'thrombolysis', 'thrombectomy')
            efficacy: Treatment efficacy (0-1)
            
        Returns:
            Simulation result
        """
        simulation_id = f"{digital_twin.patient_id}_{treatment_type}_{efficacy}"
        
        # Get baseline prediction
        baseline = digital_twin.predictions
        baseline_confidence = baseline.get('confidence', 0.5)
        
        # Simulate treatment effect
        # Treatment improves outcome by reducing stroke severity
        outcomes = []
        
        for time_step in range(10):
            time_hours = time_step * 2.4  # 24 hours total
            
            # Treatment effect increases over time
            treatment_effect = efficacy * (time_step / 10.0)
            
            # Simulate improvement in prediction
            if baseline.get('predicted_class') == 1:  # Stroke
                # Stroke confidence decreases with treatment
                new_confidence = baseline_confidence * (1 - treatment_effect * 0.5)
                predicted_class = 1 if new_confidence > 0.5 else 0
            else:
                # No stroke, remains stable
                new_confidence = baseline_confidence
                predicted_class = 0
            
            outcome = {
                'time_hours': time_hours,
                'treatment_effect': treatment_effect,
                'predicted_class': predicted_class,
                'confidence': new_confidence,
                'treatment_type': treatment_type
            }
            
            outcomes.append(outcome)
        
        # Calculate metrics
        final_outcome = outcomes[-1]
        metrics = {
            'final_confidence': final_outcome['confidence'],
            'confidence_reduction': baseline_confidence - final_outcome['confidence'],
            'treatment_efficacy': efficacy,
            'improvement_rate': (baseline_confidence - final_outcome['confidence']) / baseline_confidence if baseline_confidence > 0 else 0
        }
        
        result = SimulationResult(
            simulation_id=simulation_id,
            config=SimulationConfig(
                simulation_type=SimulationType.TREATMENT_RESPONSE,
                parameters={'treatment_type': treatment_type, 'efficacy': efficacy}
            ),
            outcomes=outcomes,
            metrics=metrics,
            timestamp=np.datetime64('now').astype(str)
        )
        
        return result


class LesionEvolutionSimulator:
    """Simulator for lesion evolution over time based on segmentation."""
    
    def __init__(self):
        """Initialize lesion evolution simulator."""
        pass
    
    def simulate_lesion_growth(self, 
                               initial_area: float,
                               growth_rate: float = 0.05,
                               time_steps: int = 24) -> SimulationResult:
        """
        Simulate lesion growth over time.
        
        Args:
            initial_area: Initial lesion area in pixels
            growth_rate: Growth rate per hour
            time_steps: Number of time steps (hours)
            
        Returns:
            Simulation result
        """
        simulation_id = f"lesion_growth_{initial_area}_{growth_rate}"
        
        outcomes = []
        current_area = initial_area
        
        for hour in range(time_steps):
            # Lesion grows according to logistic growth model
            max_area = initial_area * 2.5  # Maximum 2.5x growth
            growth = growth_rate * current_area * (1 - current_area / max_area)
            current_area += growth
            
            outcome = {
                'time_hours': hour,
                'lesion_area': current_area,
                'growth_rate': growth_rate,
                'area_change': current_area - initial_area,
                'growth_percentage': (current_area - initial_area) / initial_area * 100 if initial_area > 0 else 0
            }
            
            outcomes.append(outcome)
        
        # Calculate metrics
        final_area = outcomes[-1]['lesion_area']
        metrics = {
            'initial_area': initial_area,
            'final_area': final_area,
            'total_growth': final_area - initial_area,
            'growth_percentage': (final_area - initial_area) / initial_area * 100 if initial_area > 0 else 0
        }
        
        result = SimulationResult(
            simulation_id=simulation_id,
            config=SimulationConfig(
                simulation_type=SimulationType.LESION_GROWTH,
                parameters={'initial_area': initial_area, 'growth_rate': growth_rate}
            ),
            outcomes=outcomes,
            metrics=metrics,
            timestamp=np.datetime64('now').astype(str)
        )
        
        return result
    
    def simulate_lesion_shrinkage(self,
                                  initial_area: float,
                                  shrinkage_rate: float = 0.08,
                                  time_steps: int = 24) -> SimulationResult:
        """
        Simulate lesion shrinkage (treatment response).
        
        Args:
            initial_area: Initial lesion area in pixels
            shrinkage_rate: Shrinkage rate per hour
            time_steps: Number of time steps (hours)
            
        Returns:
            Simulation result
        """
        simulation_id = f"lesion_shrinkage_{initial_area}_{shrinkage_rate}"
        
        outcomes = []
        current_area = initial_area
        
        for hour in range(time_steps):
            # Lesion shrinks according to exponential decay
            shrinkage = shrinkage_rate * current_area
            current_area = max(0, current_area - shrinkage)
            
            outcome = {
                'time_hours': hour,
                'lesion_area': current_area,
                'shrinkage_rate': shrinkage_rate,
                'area_change': initial_area - current_area,
                'shrinkage_percentage': (initial_area - current_area) / initial_area * 100 if initial_area > 0 else 0
            }
            
            outcomes.append(outcome)
        
        # Calculate metrics
        final_area = outcomes[-1]['lesion_area']
        metrics = {
            'initial_area': initial_area,
            'final_area': final_area,
            'total_shrinkage': initial_area - final_area,
            'shrinkage_percentage': (initial_area - final_area) / initial_area * 100 if initial_area > 0 else 0
        }
        
        result = SimulationResult(
            simulation_id=simulation_id,
            config=SimulationConfig(
                simulation_type=SimulationType.LESION_SHRINKAGE,
                parameters={'initial_area': initial_area, 'shrinkage_rate': shrinkage_rate}
            ),
            outcomes=outcomes,
            metrics=metrics,
            timestamp=np.datetime64('now').astype(str)
        )
        
        return result


class SegmentationQualityAnalyzer:
    """Analyzer for segmentation quality across different scenarios."""
    
    def __init__(self, digital_twin: DigitalTwin):
        """
        Initialize segmentation quality analyzer.
        
        Args:
            digital_twin: Patient digital twin
        """
        self.digital_twin = digital_twin
    
    def analyze_segmentation_quality(self) -> Dict:
        """
        Analyze segmentation quality metrics.
        
        Returns:
            Dictionary of quality metrics
        """
        if not self.digital_twin.segmentation_results:
            return {'error': 'No segmentation results available'}
        
        seg_results = self.digital_twin.segmentation_results
        
        return {
            'dice_score': seg_results.get('dice', 0.0),
            'iou_score': seg_results.get('iou', 0.0),
            'lesion_area': seg_results.get('lesion_area', 0),
            'quality_rating': self._get_quality_rating(seg_results.get('dice', 0.0))
        }
    
    def _get_quality_rating(self, dice_score: float) -> str:
        """Get quality rating based on Dice score."""
        if dice_score >= 0.8:
            return 'Excellent'
        elif dice_score >= 0.7:
            return 'Good'
        elif dice_score >= 0.6:
            return 'Fair'
        else:
            return 'Poor'


class WhatIfAnalyzer:
    """What-if analysis for lesion segmentation scenarios."""
    
    def __init__(self, digital_twin: DigitalTwin):
        """
        Initialize what-if analyzer.
        
        Args:
            digital_twin: Patient digital twin
        """
        self.digital_twin = digital_twin
    
    def simulate_what_if_scenarios(self, scenarios: List[Dict]) -> List[SimulationResult]:
        """
        Run multiple what-if scenarios for lesion evolution.
        
        Args:
            scenarios: List of scenario configurations
            
        Returns:
            List of simulation results
        """
        results = []
        lesion_simulator = LesionEvolutionSimulator()
        
        # Get initial lesion area from digital twin
        if self.digital_twin.lesion_features:
            initial_area = self.digital_twin.lesion_features.get('lesion_area_pixels', 1000)
        elif self.digital_twin.imaging_features:
            initial_area = self.digital_twin.imaging_features.lesion_area_pixels
        else:
            initial_area = 1000  # Default
        
        for scenario in scenarios:
            scenario_type = scenario.get('type')
            
            if scenario_type == 'lesion_growth':
                # Simulate different growth rates
                for growth_rate in np.linspace(0.02, 0.1, 5):
                    result = lesion_simulator.simulate_lesion_growth(
                        initial_area=initial_area,
                        growth_rate=growth_rate,
                        time_steps=scenario.get('time_steps', 24)
                    )
                    results.append(result)
            
            elif scenario_type == 'lesion_shrinkage':
                # Simulate different treatment responses
                for shrinkage_rate in np.linspace(0.05, 0.15, 5):
                    result = lesion_simulator.simulate_lesion_shrinkage(
                        initial_area=initial_area,
                        shrinkage_rate=shrinkage_rate,
                        time_steps=scenario.get('time_steps', 24)
                    )
                    results.append(result)
        
        return results


class SimulationVisualizer:
    """Visualizer for lesion segmentation simulation results."""
    
    @staticmethod
    def plot_lesion_evolution(result: SimulationResult, save_path: Optional[str] = None):
        """
        Plot lesion evolution simulation.
        
        Args:
            result: Simulation result
            save_path: Optional path to save plot
        """
        outcomes = result.outcomes
        
        times = [o['time_hours'] for o in outcomes]
        areas = [o['lesion_area'] for o in outcomes]
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, areas, 'r-', linewidth=2, label='Lesion Area')
        plt.fill_between(times, areas, alpha=0.3)
        plt.xlabel('Time (hours)')
        plt.ylabel('Lesion Area (pixels)')
        plt.title('Lesion Evolution Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Lesion evolution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_lesion_shrinkage(result: SimulationResult, save_path: Optional[str] = None):
        """
        Plot lesion shrinkage (treatment response) simulation.
        
        Args:
            result: Simulation result
            save_path: Optional path to save plot
        """
        outcomes = result.outcomes
        
        times = [o['time_hours'] for o in outcomes]
        areas = [o['lesion_area'] for o in outcomes]
        shrinkage_percentages = [o['shrinkage_percentage'] for o in outcomes]
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Lesion area over time
        axes[0].plot(times, areas, 'g-', linewidth=2, label='Lesion Area')
        axes[0].fill_between(times, areas, alpha=0.3)
        axes[0].set_xlabel('Time (hours)')
        axes[0].set_ylabel('Lesion Area (pixels)')
        axes[0].set_title('Lesion Area Reduction Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Shrinkage percentage
        axes[1].plot(times, shrinkage_percentages, 'b-', linewidth=2, label='Shrinkage %')
        axes[1].fill_between(times, shrinkage_percentages, alpha=0.3)
        axes[1].set_xlabel('Time (hours)')
        axes[1].set_ylabel('Shrinkage Percentage (%)')
        axes[1].set_title('Treatment Response: Lesion Shrinkage')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Lesion shrinkage plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_segmentation_quality(quality_metrics: Dict, save_path: Optional[str] = None):
        """
        Plot segmentation quality metrics.
        
        Args:
            quality_metrics: Quality metrics dictionary
            save_path: Optional path to save plot
        """
        metrics_names = ['Dice Score', 'IoU Score']
        metrics_values = [quality_metrics.get('dice_score', 0), quality_metrics.get('iou_score', 0)]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(metrics_names, metrics_values, color=['steelblue', 'coral'], alpha=0.7)
        
        # Add quality threshold line
        plt.axhline(y=0.7, color='green', linestyle='--', label='Good Quality Threshold')
        
        plt.ylabel('Score')
        plt.title('Segmentation Quality Metrics')
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend()
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Segmentation quality plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main function to test segmentation-based simulation capabilities."""
    
    print("Testing Lesion Segmentation Simulation Module")
    print("=" * 60)
    
    # Test lesion evolution simulator
    print("\n1. Testing Lesion Evolution Simulator")
    lesion_simulator = LesionEvolutionSimulator()
    
    # Simulate lesion growth
    growth_result = lesion_simulator.simulate_lesion_growth(
        initial_area=1000,
        growth_rate=0.05,
        time_steps=24
    )
    print(f"  Lesion Growth Simulation:")
    print(f"    Initial Area: {growth_result.metrics['initial_area']:.0f} pixels")
    print(f"    Final Area: {growth_result.metrics['final_area']:.0f} pixels")
    print(f"    Growth: {growth_result.metrics['growth_percentage']:.1f}%")
    
    # Simulate lesion shrinkage
    shrinkage_result = lesion_simulator.simulate_lesion_shrinkage(
        initial_area=1000,
        shrinkage_rate=0.08,
        time_steps=24
    )
    print(f"  Lesion Shrinkage Simulation:")
    print(f"    Initial Area: {shrinkage_result.metrics['initial_area']:.0f} pixels")
    print(f"    Final Area: {shrinkage_result.metrics['final_area']:.0f} pixels")
    print(f"    Shrinkage: {shrinkage_result.metrics['shrinkage_percentage']:.1f}%")
    
    # Test segmentation quality analyzer
    print("\n2. Testing Segmentation Quality Analyzer")
    
    # Create a mock digital twin with segmentation results
    try:
        from .twin import DigitalTwin, PatientProfile
    except ImportError:
        from digital_twin.twin import DigitalTwin, PatientProfile
    mock_twin = DigitalTwin(
        patient_id="test_patient",
        patient_profile=PatientProfile(patient_id="test_patient")
    )
    mock_twin.add_segmentation_result({
        'dice': 0.85,
        'iou': 0.75,
        'lesion_area': 1200
    })
    mock_twin.add_lesion_features({
        'lesion_area_pixels': 1200,
        'lesion_center_x': 112.0,
        'lesion_center_y': 112.0,
        'segmentation_quality': 0.85
    })
    
    quality_analyzer = SegmentationQualityAnalyzer(mock_twin)
    quality_metrics = quality_analyzer.analyze_segmentation_quality()
    print(f"  Segmentation Quality:")
    print(f"    Dice Score: {quality_metrics['dice_score']:.3f}")
    print(f"    IoU Score: {quality_metrics['iou_score']:.3f}")
    print(f"    Lesion Area: {quality_metrics['lesion_area']} pixels")
    print(f"    Quality Rating: {quality_metrics['quality_rating']}")
    
    # Test what-if analyzer
    print("\n3. Testing What-If Analyzer")
    what_if_analyzer = WhatIfAnalyzer(mock_twin)
    
    scenarios = [
        {'type': 'lesion_growth', 'time_steps': 24},
        {'type': 'lesion_shrinkage', 'time_steps': 24}
    ]
    
    what_if_results = what_if_analyzer.simulate_what_if_scenarios(scenarios)
    print(f"  Generated {len(what_if_results)} what-if scenarios")
    
    # Test visualization
    print("\n4. Testing Visualization")
    output_dir = Path(r"d:\Stroke-DigitalTwin\outputs\simulation_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    SimulationVisualizer.plot_lesion_evolution(
        growth_result,
        save_path=str(output_dir / "lesion_evolution.png")
    )
    
    SimulationVisualizer.plot_lesion_shrinkage(
        shrinkage_result,
        save_path=str(output_dir / "lesion_shrinkage.png")
    )
    
    SimulationVisualizer.plot_segmentation_quality(
        quality_metrics,
        save_path=str(output_dir / "segmentation_quality.png")
    )
    
    print(f"\nSimulation plots saved to {output_dir}")
    
    print("\n" + "=" * 60)
    print("Segmentation Simulation Module Testing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

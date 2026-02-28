"""
AEGIS Auto-Optimizer
Automatically optimizes strategy parameters based on performance
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json

import pandas as pd
import numpy as np
import yaml

from ..core.signal_generator import SignalGenerator
from ..utils.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """
    Optimizes strategy parameters based on historical performance
    """
    
    def __init__(self):
        self.tracker = PerformanceTracker()
        self.optimization_history = []
        self.param_bounds = {
            'min_confidence': (0.5, 0.8),
            'min_risk_reward': (1.0, 3.0),
            'kelly_fraction': (0.25, 0.75),
            'max_position_pct': (0.05, 0.20)
        }
    
    def analyze_performance(self, lookback_days: int = 30) -> Dict:
        """
        Analyze recent performance to identify issues
        """
        metrics = self.tracker.calculate_metrics()
        
        if not metrics or metrics['total_trades'] < 20:
            return {'status': 'insufficient_data'}
        
        issues = []
        suggestions = []
        
        # Check win rate
        if metrics['win_rate'] < 0.45:
            issues.append('low_win_rate')
            suggestions.append({
                'param': 'min_confidence',
                'current': 0.6,
                'suggested': 0.7,
                'reason': 'Low win rate, increase confidence threshold'
            })
        
        # Check drawdown
        if metrics['max_drawdown'] < -0.10:
            issues.append('high_drawdown')
            suggestions.append({
                'param': 'max_position_pct',
                'current': 0.10,
                'suggested': 0.05,
                'reason': 'High drawdown, reduce position size'
            })
        
        # Check Sharpe
        if metrics['sharpe_ratio'] < 1.0:
            issues.append('low_sharpe')
            suggestions.append({
                'param': 'min_risk_reward',
                'current': 1.5,
                'suggested': 2.0,
                'reason': 'Low risk-adjusted returns, require better R:R'
            })
        
        # Check profit factor
        if metrics['profit_factor'] < 1.5:
            issues.append('low_profit_factor')
            suggestions.append({
                'param': 'kelly_fraction',
                'current': 0.5,
                'suggested': 0.25,
                'reason': 'Reduce Kelly fraction to preserve capital'
            })
        
        return {
            'status': 'analyzed',
            'issues': issues,
            'suggestions': suggestions,
            'current_metrics': metrics
        }
    
    def optimize_parameters(self) -> Optional[Dict]:
        """
        Generate optimized parameters
        """
        analysis = self.analyze_performance()
        
        if analysis['status'] == 'insufficient_data':
            return None
        
        # Load current config
        with open('config/settings.yaml') as f:
            config = yaml.safe_load(f)
        
        # Apply suggestions
        new_params = {}
        for suggestion in analysis.get('suggestions', []):
            param = suggestion['param']
            new_params[param] = suggestion['suggested']
        
        # Record optimization
        optimization_record = {
            'timestamp': datetime.now().isoformat(),
            'previous_metrics': analysis['current_metrics'],
            'issues_found': analysis['issues'],
            'parameter_changes': new_params
        }
        
        self.optimization_history.append(optimization_record)
        
        # Save to file
        with open('data/processed/optimization_history.json', 'w') as f:
            json.dump(self.optimization_history, f, indent=2, default=str)
        
        return {
            'new_params': new_params,
            'reasoning': analysis['suggestions']
        }
    
    def A_B_test(
        self,
        param_a: Dict,
        param_b: Dict,
        test_periods: int = 10
    ) -> Dict:
        """
        A/B test two parameter sets
        """
        # This would run two parallel simulations
        # For now, return structure
        
        return {
            'test_id': f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'variant_a': param_a,
            'variant_b': param_b,
            'periods': test_periods,
            'status': 'pending'
        }
    
    def get_optimal_parameters(self) -> Dict:
        """
        Get current optimal parameters
        """
        # Check if we have recent optimization
        opt_result = self.optimize_parameters()
        
        if opt_result:
            return opt_result['new_params']
        
        # Return defaults
        return {
            'min_confidence': 0.6,
            'min_risk_reward': 1.5,
            'kelly_fraction': 0.5,
            'max_position_pct': 0.10
        }


class ModelMonitor:
    """
    Monitors ML model performance and triggers retraining
    """
    
    def __init__(self):
        self.performance_threshold = 0.55  # Minimum accuracy
        self.drift_threshold = 0.1  # Maximum allowed drift
    
    def check_model_drift(
        self,
        recent_predictions: pd.DataFrame,
        actual_outcomes: pd.Series
    ) -> Dict:
        """
        Check for model performance degradation
        """
        if len(recent_predictions) < 50:
            return {'status': 'insufficient_data'}
        
        # Calculate recent accuracy
        accuracy = (recent_predictions['prediction'] == actual_outcomes).mean()
        
        # Check for drift in feature distributions
        # (Simplified - would use statistical tests in production)
        
        drift_detected = accuracy < self.performance_threshold
        
        return {
            'status': 'drift_detected' if drift_detected else 'healthy',
            'recent_accuracy': accuracy,
            'threshold': self.performance_threshold,
            'recommendation': 'retrain' if drift_detected else 'monitor'
        }
    
    def should_retrain(self) -> bool:
        """
        Determine if models should be retrained
        """
        # Check last training time
        try:
            with open('data/models/ensemble_latest_meta.json') as f:
                meta = json.load(f)
                last_train = datetime.fromisoformat(meta['timestamp'])
                
                days_since = (datetime.now() - last_train).days
                
                if days_since > 7:
                    return True
                
        except:
            return True
        
        return False


# Convenience function
def run_optimization() -> Dict:
    """Run full optimization cycle"""
    optimizer = StrategyOptimizer()
    return optimizer.optimize_parameters() or {'status': 'no_changes_needed'}

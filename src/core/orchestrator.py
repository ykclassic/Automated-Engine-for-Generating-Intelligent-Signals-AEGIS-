"""
AEGIS Main Orchestrator
Coordinates all components for end-to-end operation
"""

import logging
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Path injection to ensure 'src' is discoverable from the root
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

try:
    from src.core.data_fetcher import DataPipeline
    from src.core.signal_generator import SignalGenerator
    # Note: Check if your file is risk_management.py or risk_manager.py based on your log
    from src.core.risk_management import RiskManager, RiskLevel
    from src.utils.performance_tracker import PerformanceTracker
    from src.utils.system_monitor import SystemMonitor
    from src.utils.optimizer import StrategyOptimizer
    from src.notifications.formatter import SignalFormatter
except ImportError as e:
    logging.error(f"Orchestrator Import Error: {e}")
    raise

logger = logging.getLogger(__name__)

class AEGISOrchestrator:
    def __init__(self, risk_level: str = 'moderate', account_balance: float = 10000):
        # Convert string to RiskLevel logic if necessary
        self.account_balance = account_balance
        
        self.data_pipeline = DataPipeline()
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager(risk_level=risk_level)
        self.performance_tracker = PerformanceTracker()
        self.system_monitor = SystemMonitor()
        self.optimizer = StrategyOptimizer()
        self.formatter = SignalFormatter()
        
        logger.info("🛡️ AEGIS Orchestrator initialized")

    def calculate_all(self, df):
        """
        Proxy method used by FeatureEngineer to access indicator logic.
        (Matches the call signature in your feature_engineering.py)
        """
        # This assumes your orchestrator pulls from src.indicators
        from src.indicators.trend import TrendIndicators
        from src.indicators.momentum import MomentumIndicators
        
        # Example calculation flow
        df = TrendIndicators().add_ema(df)
        df = MomentumIndicators().add_rsi(df)
        return df

    def run_cycle(self) -> Dict:
        cycle_start = datetime.now()
        results = {'timestamp': cycle_start.isoformat(), 'status': 'running', 'signals_generated': 0, 'errors': []}
        
        try:
            all_data = self.data_pipeline.fetch_all_assets()
            if not all_data:
                results['errors'].append("No data fetched")
                return results
            
            # Additional cycle logic here...
            results['status'] = 'success'
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            results['status'] = 'error'
            results['errors'].append(str(e))
        
        return results

if __name__ == "__main__":
    orchestrator = AEGISOrchestrator()
    print(json.dumps(orchestrator.run_cycle(), indent=2))

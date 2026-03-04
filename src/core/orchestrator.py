import logging
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Setup Root Path
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

logger = logging.getLogger(__name__)

class AEGISOrchestrator:
    def __init__(self, risk_level: str = 'moderate', account_balance: float = 10000.0):
        self.risk_level = risk_level
        self.account_balance = account_balance
        logger.info("AEGIS Orchestrator initialized")

    def calculate_all(self, df):
        """
        Proxy for Indicator Orchestration.
        Imports indicator modules only when needed to decouple dependencies.
        """
        try:
            from src.indicators.trend import TrendIndicators
            from src.indicators.momentum import MomentumIndicators
            
            trend = TrendIndicators()
            mom = MomentumIndicators()
            
            df = trend.add_ema(df, periods=[50, 200])
            df = mom.add_rsi(df, period=14)
            return df
        except ImportError as e:
            logger.error(f"Failed to load indicator modules: {e}")
            return df

    def run_cycle(self) -> Dict:
        """
        End-to-end execution loop.
        Uses local imports to prevent ModuleNotFoundErrors in specialized jobs.
        """
        cycle_start = datetime.now()
        results = {
            'timestamp': cycle_start.isoformat(),
            'status': 'running',
            'signals_generated': 0,
            'errors': []
        }
        
        try:
            # Lazy Imports for components that require heavy external libraries
            from src.core.data_fetcher import DataPipeline
            from src.core.risk_management import RiskManager
            from src.core.signal_generator import SignalGenerator

            # 1. Fetch Data
            pipeline = DataPipeline()
            all_data = pipeline.fetch_all_assets()
            
            if not all_data:
                results['status'] = 'no_data'
                return results

            # 2. Generate Signals
            generator = SignalGenerator()
            signals = generator.generate_all_signals(all_data, self.account_balance)
            results['signals_generated'] = len(signals)
            
            results['status'] = 'success'
            results['duration'] = (datetime.now() - cycle_start).total_seconds()
            
        except Exception as e:
            logger.error(f"Cycle execution failed: {e}")
            results['status'] = 'error'
            results['errors'].append(str(e))
        
        return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orchestrator = AEGISOrchestrator()
    print(json.dumps(orchestrator.run_cycle(), indent=2, default=str))

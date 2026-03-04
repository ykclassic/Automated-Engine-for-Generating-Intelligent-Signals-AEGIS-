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
        The core engine called by FeatureEngineer.
        Coordinates technical indicator logic from the indicators package.
        """
        try:
            from src.indicators.trend import TrendIndicators
            from src.indicators.momentum import MomentumIndicators
            
            # Initialize indicator modules
            trend = TrendIndicators()
            mom = MomentumIndicators()
            
            # Chain the indicator calculations
            df = trend.add_ema(df, periods=[50, 200])
            df = trend.add_bollinger_bands(df, period=20, std_dev=2)
            df = mom.add_rsi(df, period=14)
            
            return df
        except Exception as e:
            logger.error(f"Error in Orchestrator indicator chain: {e}")
            raise  # Raise so FeatureEngineer knows to use fallback

    def run_cycle(self) -> Dict:
        """End-to-end execution loop for the main bot runner."""
        cycle_start = datetime.now()
        results = {'timestamp': cycle_start.isoformat(), 'status': 'running', 'signals_generated': 0, 'errors': []}
        
        try:
            from src.core.data_fetcher import DataPipeline
            from src.core.signal_generator import SignalGenerator

            pipeline = DataPipeline()
            all_data = pipeline.fetch_all_assets()
            
            if not all_data:
                results['status'] = 'no_data'
                return results

            generator = SignalGenerator()
            signals = generator.generate_all_signals(all_data, self.account_balance)
            results['signals_generated'] = len(signals)
            results['status'] = 'success'
            
        except Exception as e:
            logger.error(f"Cycle execution failed: {e}")
            results['status'] = 'error'
            results['errors'].append(str(e))
        
        return results

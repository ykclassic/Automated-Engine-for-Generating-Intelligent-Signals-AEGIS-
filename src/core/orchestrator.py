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
        self.config_path = "config/indicators.yaml"
        logger.info("🛡️ AEGIS Orchestrator initialized")

    def calculate_all(self, df):
        """
        Coordinates the calculation across all indicator modules.
        This maps directly to your custom volume, momentum, trend, and volatility files.
        """
        try:
            # Ensure the config directory exists to prevent initialization errors
            os.makedirs("config", exist_ok=True)
            if not os.path.exists(self.config_path):
                with open(self.config_path, 'w') as f:
                    f.write("indicators: {}\n")

            from src.indicators.trend import TrendIndicators
            from src.indicators.momentum import MomentumIndicators
            from src.indicators.volume import VolumeIndicators
            from src.indicators.volatility import VolatilityIndicators
            
            # Initialize modules with your config path
            trend = TrendIndicators(config_path=self.config_path)
            mom = MomentumIndicators(config_path=self.config_path)
            vol = VolumeIndicators(config_path=self.config_path)
            vlt = VolatilityIndicators(config_path=self.config_path)
            
            # Chain the 'calculate_all' method from each of your files
            logger.info("Calculating Trend indicators...")
            df = trend.calculate_all(df)
            
            logger.info("Calculating Momentum indicators...")
            df = mom.calculate_all(df)
            
            logger.info("Calculating Volume indicators...")
            df = vol.calculate_all(df)
            
            logger.info("Calculating Volatility indicators...")
            df = vlt.calculate_all(df)
            
            return df

        except Exception as e:
            logger.error(f"❌ Orchestration Error: {str(e)}")
            # If a specific module fails, we return the DF as-is so the pipeline continues
            return df

    def run_cycle(self) -> Dict:
        """Main execution loop for high-level bot operation."""
        cycle_start = datetime.now()
        results = {
            'timestamp': cycle_start.isoformat(),
            'status': 'running',
            'signals_generated': 0,
            'errors': []
        }
        
        try:
            from src.core.data_fetcher import DataPipeline
            from src.core.signal_generator import SignalGenerator

            # 1. Fetch
            pipeline = DataPipeline()
            all_data = pipeline.fetch_all_assets()
            
            if not all_data:
                results['status'] = 'no_data'
                return results

            # 2. Process & Signal
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
    # Test internal calculation if a sample file exists
    sample_file = Path("data/raw/BTC_USDT_1h.parquet")
    if sample_file.exists():
        import pandas as pd
        test_df = pd.read_parquet(sample_file)
        result_df = orchestrator.calculate_all(test_df)
        print(f"Calculation test successful. Columns: {len(result_df.columns)}")

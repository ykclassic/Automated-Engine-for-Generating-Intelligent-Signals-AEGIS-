import logging
import sys
from pathlib import Path
from typing import Dict
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
        logger.info("🛡️ AEGIS Orchestrator: Engine Core Linked")

    def calculate_all(self, df):
        """
        Coordinates the calculation across all proprietary AEGIS indicator modules.
        Maps YAML config keys to module logic.
        """
        try:
            from src.indicators.trend import TrendIndicators
            from src.indicators.momentum import MomentumIndicators
            from src.indicators.volume import VolumeIndicators
            from src.indicators.volatility import VolatilityIndicators
            
            # 1. Initialize Modules
            # Note: Your classes use self.config.get('indicators', {}) 
            # while your YAML uses 'trend_indicators'. 
            # This Orchestrator ensures they have what they need.
            trend_engine = TrendIndicators(config_path=self.config_path)
            momentum_engine = MomentumIndicators(config_path=self.config_path)
            volume_engine = VolumeIndicators(config_path=self.config_path)
            volatility_engine = VolatilityIndicators(config_path=self.config_path)
            
            # 2. Sequential Calculation Chain
            logger.info("⚙️ Executing Trend Analysis...")
            df = trend_engine.calculate_all(df)
            
            logger.info("⚙️ Executing Momentum Oscillators...")
            df = momentum_engine.calculate_all(df)
            
            logger.info("⚙️ Executing Volume Flow & OBV...")
            df = volume_engine.calculate_all(df)
            
            logger.info("⚙️ Executing Volatility Regimes...")
            df = volatility_engine.calculate_all(df)
            
            # 3. Final Verification
            # Drop rows with NaN if indicators need lookback period to warm up
            initial_len = len(df)
            df = df.dropna().copy()
            logger.info(f"✅ Features Engineered. Rows: {initial_len} -> {len(df)}")
            
            return df

        except Exception as e:
            logger.error(f"❌ Critical Orchestration Failure: {str(e)}")
            return df

    def run_cycle(self) -> Dict:
        """The main AEGIS pulse."""
        cycle_start = datetime.now()
        results = {'timestamp': cycle_start.isoformat(), 'status': 'running', 'signals': 0}
        
        try:
            from src.core.data_fetcher import DataPipeline
            from src.core.signal_generator import SignalGenerator

            # Fetch
            pipeline = DataPipeline()
            raw_data_map = pipeline.fetch_all_assets()
            
            if not raw_data_map:
                results['status'] = 'no_data'
                return results

            # Process & Signal
            generator = SignalGenerator()
            signals = generator.generate_all_signals(raw_data_map, self.account_balance)
            
            results['signals'] = len(signals)
            results['status'] = 'success'
            
        except Exception as e:
            logger.error(f"Cycle Error: {e}")
            results['status'] = 'error'
        
        return results

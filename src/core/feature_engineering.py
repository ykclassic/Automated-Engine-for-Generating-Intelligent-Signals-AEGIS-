"""
AEGIS Feature Engineering Module
Job 2 in the AEGIS Pipeline: Transforms raw OHLCV data into 
quantitative features for signal generation.
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add root to path for local execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    from src.indicators.orchestrator import IndicatorOrchestrator
except ImportError:
    # Fallback for runner contexts
    from indicators import IndicatorOrchestrator

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Handles technical indicator calculation, normalization, 
    and regime identification.
    """
    
    def __init__(self):
        self.orchestrator = IndicatorOrchestrator()

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point for feature calculation.
        """
        if df is None or df.empty:
            logger.error("Empty DataFrame passed to FeatureEngineer.")
            return pd.DataFrame()

        # 1. Base Technical Indicators
        df = self.orchestrator.calculate_all(df)

        # 2. Volatility Regimes (Normalized)
        if 'atr' in df.columns:
            df['volatility_ratio'] = df['atr'] / df['close']
            df['volatility_regime'] = np.where(
                df['volatility_ratio'] > df['volatility_ratio'].rolling(50).mean(), 
                'high', 'normal'
            )

        # 3. Momentum & Trend Confluence
        if all(col in df.columns for col in ['ema_50', 'ema_200']):
            df['trend_bullish_score'] = np.where(df['ema_50'] > df['ema_200'], 1.0, 0.0)
            df['ema_50_200_ratio'] = df['ema_50'] / df['ema_200']

        # 4. Cleanup: Handle NaNs from rolling windows
        # We drop the 'warmup' rows (usually 200 for EMA 200)
        initial_len = len(df)
        df = df.dropna().copy()
        logger.info(f"Feature calculation complete. Rows: {initial_len} -> {len(df)}")

        return df

if __name__ == "__main__":
    # Quick CLI test
    engineer = FeatureEngineer()
    print("FeatureEngineer Initialized Successfully.")

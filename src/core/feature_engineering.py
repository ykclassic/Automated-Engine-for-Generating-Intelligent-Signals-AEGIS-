"""
AEGIS Feature Engineering Module
Standardized to absolute imports for GitHub Actions compatibility.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path

# Add the project root to sys.path to ensure 'src' is discoverable
# This handles cases where PYTHONPATH might not propagate to sub-processes
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

try:
    # Absolute import is preferred for the AEGIS pipeline
    from src.indicators.orchestrator import IndicatorOrchestrator
except ImportError as e:
    logging.error(f"Failed to import IndicatorOrchestrator: {e}")
    # Final fallback for unusual runner environments
    sys.path.append(str(root_path / "src"))
    from indicators.orchestrator import IndicatorOrchestrator

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.orchestrator = IndicatorOrchestrator()

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        # 1. Base Technical Indicators
        df = self.orchestrator.calculate_all(df)

        # 2. Add custom AEGIS logic (Volatility & Trend)
        if 'atr' in df.columns:
            df['volatility_ratio'] = df['atr'] / df['close']
        
        if all(col in df.columns for col in ['ema_50', 'ema_200']):
            df['trend_bullish'] = (df['ema_50'] > df['ema_200']).astype(int)

        # Drop NaNs created by lagging indicators (e.g., EMA 200)
        return df.dropna().copy()

if __name__ == "__main__":
    print("✅ FeatureEngineer module loaded successfully.")

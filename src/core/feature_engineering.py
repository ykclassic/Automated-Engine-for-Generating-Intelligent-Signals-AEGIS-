"""
AEGIS Feature Engineering Module
Resilient import logic for GitHub Actions Runners.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Resilient Import Block ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent # Goes from core -> src -> root

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    # Attempt absolute import from root
    from src.indicators.orchestrator import IndicatorOrchestrator
except (ImportError, ModuleNotFoundError):
    try:
        # Attempt direct import from src
        sys.path.append(str(project_root / "src"))
        from indicators.orchestrator import IndicatorOrchestrator
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"CRITICAL: Could not find IndicatorOrchestrator. {e}")
        raise

# --- Class Definition ---
class FeatureEngineer:
    def __init__(self):
        self.orchestrator = IndicatorOrchestrator()

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all features. Ensure IndicatorOrchestrator 
        has a 'calculate_all' method.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # 1. Indicators from Orchestrator
        df = self.orchestrator.calculate_all(df)

        # 2. Supplemental AEGIS Features
        if 'close' in df.columns:
            # Simple Log Returns
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            
            # Trend Check (SMA 50/200)
            if len(df) > 200:
                df['sma_50'] = df['close'].rolling(window=50).mean()
                df['sma_200'] = df['close'].rolling(window=200).mean()
                df['bullish_regime'] = (df['sma_50'] > df['sma_200']).astype(int)

        return df.dropna().copy()

if __name__ == "__main__":
    print("✅ FeatureEngineer loaded.")

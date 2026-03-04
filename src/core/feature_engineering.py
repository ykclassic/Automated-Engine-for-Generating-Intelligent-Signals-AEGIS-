"""
AEGIS Feature Engineering Module
Job 2 in the AEGIS Pipeline: Transforms raw OHLCV data into 
quantitative features using the AEGIS Orchestrator.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path

# --- THE PATH GUARD ---
# Ensures 'src' is discoverable regardless of where the script is called from
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- RESILIENT IMPORTS ---
try:
    # Based on your logs, the orchestrator lives in src.core
    from src.core.orchestrator import AEGISOrchestrator
except ImportError as e:
    logger.error(f"CRITICAL: Could not find AEGISOrchestrator in src/core/orchestrator.py. Error: {e}")
    # Last ditch attempt for local environments
    try:
        from orchestrator import AEGISOrchestrator
    except ImportError:
        sys.exit(1)

class FeatureEngineer:
    """
    AEGIS Feature Engineering Engine
    Utilizes the Orchestrator to calculate technical indicators 
    and applies secondary alpha-generating filters.
    """
    
    def __init__(self):
        # We use the Orchestrator as the 'Indicator Factory'
        self.orchestrator = AEGISOrchestrator()
        logger.info("✅ FeatureEngineer initialized with AEGIS Orchestrator")

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing pipeline for transforming raw price data into 
        a feature set suitable for signal generation.
        """
        if df is None or df.empty:
            logger.warning("Received empty DataFrame for feature engineering.")
            return pd.DataFrame()

        # 1. Base Technical Indicators (Calculated via Orchestrator)
        # Note: This calls the internal indicator logic in src/indicators/
        try:
            df = self.orchestrator.calculate_all(df)
        except AttributeError:
            logger.error("Orchestrator is missing 'calculate_all' method.")
            # Fallback to manual column generation if orchestrator fails
            df = self._manual_fallback_features(df)

        # 2. Volatility Analysis (Alpha Feature)
        if 'close' in df.columns:
            # Normalized ATR or simple volatility
            df['returns'] = df['close'].pct_change()
            df['volatility_20'] = df['returns'].rolling(window=20).std()
            
            # Identify "Volatility Squeeze"
            df['vol_sma'] = df['volatility_20'].rolling(window=50).mean()
            df['is_squeezed'] = (df['volatility_20'] < df['vol_sma']).astype(int)

        # 3. Trend Confluence (Alpha Feature)
        if 'ema_50' in df.columns and 'ema_200' in df.columns:
            df['trend_strength'] = (df['ema_50'] / df['ema_200']) - 1
            df['bullish_regime'] = (df['ema_50'] > df['ema_200']).astype(int)

        # 4. Cleanup
        # We drop the first 200 rows to remove 'warm-up' noise from EMA/SMA 200
        initial_count = len(df)
        df = df.dropna().copy()
        
        logger.info(f"Processed features: {initial_count} -> {len(df)} rows.")
        return df

    def _manual_fallback_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Emergency feature set if orchestrator/indicators fail to load."""
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['rsi_14'] = self._calc_rsi(df['close'], 14)
        return df

    @staticmethod
    def _calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

if __name__ == "__main__":
    # Test initialization
    try:
        engineer = FeatureEngineer()
        print("Success: FeatureEngineer is ready for the AEGIS pipeline.")
    except Exception as e:
        print(f"Failure: {e}")

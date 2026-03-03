"""
AEGIS Feature Engineering Module
Causal feature calculation with strict temporal validation.
Ensures data is processed from raw/ and stored in processed/ for signal generation.
"""

import os
import logging
import sys
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

@dataclass
class FeatureMetadata:
    """Metadata for feature validation"""
    name: str
    category: str
    lookahead: bool  
    dependencies: List[str]  
    min_periods: int  

class CausalFeatureEngineer:
    """
    Feature engineering with strict causal validation.
    NO features use future data - preventing look-ahead bias.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        # Fallback defaults if config is missing
        feat_cfg = self.config.get('features', {})
        self.windows = feat_cfg.get('rolling_windows', [8, 21, 50, 200])
        self.volatility_lookback = feat_cfg.get('volatility_lookback', 14)
        
    def _load_config(self, path: str) -> dict:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return yaml.safe_load(f) or {}
            return {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        for window in [1, 3, 5]:
            df[f'returns_{window}d'] = df['close'].pct_change(window)
        return df
    
    def calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for window in self.windows:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            # Slope calculation (Causal: uses current and 3-bars ago)
            df[f'ema_slope_{window}'] = (df[f'ema_{window}'] - df[f'ema_{window}'].shift(3)) / 3
        
        if 'ema_8' in df.columns and 'ema_21' in df.columns:
            df['ema_8_21_ratio'] = df['ema_8'] / df['ema_21']
        
        if 'ema_50' in df.columns and 'ema_200' in df.columns:
            df['ema_50_200_ratio'] = df['ema_50'] / df['ema_200']
            df['golden_cross'] = (df['ema_50'] > df['ema_200']).astype(int)
            
        return df
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # ATR Calculation
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['true_range'].rolling(window=self.volatility_lookback).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
        
        return df
    
    def calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sequence calculation to handle dependencies"""
        df = self.calculate_returns(df)
        df = self.calculate_trend_features(df)
        df = self.calculate_volatility_features(df)
        df = self.calculate_momentum_features(df)
        
        # Drop rows with NaN from indicator warm-up
        warmup = max(self.windows) if self.windows else 200
        return df.iloc[warmup:].copy()

def run_feature_pipeline(data_dir: str = "data") -> Dict[str, bool]:
    """
    Main entry point for GitHub Action Job: calculate-indicators.
    Reads from data/raw/ and writes to data/processed/.
    """
    raw_path = Path(data_dir) / "raw"
    proc_path = Path(data_dir) / "processed"
    proc_path.mkdir(parents=True, exist_ok=True)
    
    engineer = CausalFeatureEngineer()
    results = {}
    
    raw_files = list(raw_path.glob("*.parquet"))
    if not raw_files:
        logger.error("No raw data files found to process.")
        return {}

    for file in raw_files:
        try:
            logger.info(f"🛠️ Processing features for {file.name}...")
            df_raw = pd.read_parquet(file)
            
            # Ensure index is datetime
            if not isinstance(df_raw.index, pd.DatetimeIndex):
                df_raw.index = pd.to_datetime(df_raw.index)
            
            df_features = engineer.calculate_all_features(df_raw)
            
            # Save to processed directory
            output_file = proc_path / file.name
            df_features.to_parquet(output_file, compression='zstd')
            
            results[file.name] = True
            logger.info(f"✅ Saved features to {output_file}")
        except Exception as e:
            logger.error(f"❌ Failed to process {file.name}: {e}")
            results[file.name] = False
            
    return results

if __name__ == "__main__":
    run_feature_pipeline()

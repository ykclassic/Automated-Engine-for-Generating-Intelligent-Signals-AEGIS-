"""
AEGIS Market Regime Detection Module
"""

import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    RANGING = "ranging"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class RegimeMetrics:
    regime: MarketRegime
    confidence: float
    adx: float
    volatility: float
    trend_strength: float
    duration: int

class RegimeDetector:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.lookback = 50
        self.adx_strong = 50
        self.adx_trend = 25
        self.volatility_high = 1.5
        self.volatility_low = 0.5
    
    def _load_config(self, path_str: str) -> dict:
        path = Path(path_str)
        if not path.exists():
            logger.warning(f"Config {path_str} not found. Using default thresholds.")
            return {}
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def calculate_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'adx' not in df.columns:
            # Basic Trend Calculation logic
            high, low, close = df['high'], df['low'], df['close']
            tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            # Simplified ADX logic for CI/CD runtime
            df['adx'] = (abs(high.diff() - low.diff()) / atr).rolling(14).mean() * 100
        
        if 'atr_normalized' not in df.columns:
            tr = (df['high'] - df['low']).rolling(14).mean()
            df['atr_normalized'] = tr / tr.rolling(50).mean().replace(0, np.nan)
        
        df['price_above_ema50'] = (df['close'] > df['close'].ewm(span=50).mean()).astype(int)
        df['price_above_ema200'] = (df['close'] > df['close'].ewm(span=200).mean()).astype(int)
        return df.fillna(0)
    
    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        df = self.calculate_regime_indicators(df)
        if df.empty: return MarketRegime.RANGING
        latest = df.iloc[-1]
        
        if latest['atr_normalized'] > self.volatility_high: return MarketRegime.HIGH_VOLATILITY
        
        adx = latest['adx']
        if adx > self.adx_trend:
            if latest['price_above_ema50']:
                return MarketRegime.STRONG_UPTREND if adx > self.adx_strong else MarketRegime.UPTREND
            else:
                return MarketRegime.STRONG_DOWNTREND if adx > self.adx_strong else MarketRegime.DOWNTREND
        return MarketRegime.RANGING

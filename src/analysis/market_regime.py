"""
AEGIS Market Regime Detection Module
Identifies trending, ranging, and volatile market conditions
"""

import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

import pandas as pd
import numpy as np
import yaml
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    RANGING = "ranging"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class RegimeMetrics:
    """Container for regime analysis metrics"""
    regime: MarketRegime
    confidence: float
    adx: float
    volatility: float
    trend_strength: float
    duration: int


class RegimeDetector:
    """
    Detects market regimes using multiple methods:
    1. ADX-based trend strength
    2. Volatility percentiles
    3. Price action structure
    4. Machine learning clustering (optional)
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.lookback = 50  # Periods for regime calculation
        
        # Thresholds
        self.adx_strong = 50
        self.adx_trend = 25
        self.volatility_high = 1.5
        self.volatility_low = 0.5
    
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def calculate_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate base indicators for regime detection
        """
        df = df.copy()
        
        # Trend strength (ADX)
        if 'adx' not in df.columns:
            # Calculate basic ADX if not present
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            atr = tr.rolling(window=14).mean()
            plus_di = 100 * plus_dm.rolling(window=14).mean() / atr
            minus_di = 100 * minus_dm.rolling(window=14).mean() / atr
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(window=14).mean()
        
        # Volatility regime
        if 'atr_normalized' not in df.columns:
            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            df['atr_normalized'] = atr / atr.rolling(50).mean()
        
        # Price structure
        df['higher_highs'] = (
            (df['high'] > df['high'].shift(1)) & 
            (df['high'].shift(1) > df['high'].shift(2))
        ).astype(int)
        
        df['lower_lows'] = (
            (df['low'] < df['low'].shift(1)) & 
            (df['low'].shift(1) < df['low'].shift(2))
        ).astype(int)
        
        # Trend consistency
        df['price_above_ema50'] = (df['close'] > df['close'].ewm(span=50).mean()).astype(int)
        df['price_above_ema200'] = (df['close'] > df['close'].ewm(span=200).mean()).astype(int)
        
        return df
    
    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime from latest data
        """
        df = self.calculate_regime_indicators(df)
        latest = df.iloc[-1]
        
        # Volatility check first
        if latest['atr_normalized'] > self.volatility_high:
            return MarketRegime.HIGH_VOLATILITY
        if latest['atr_normalized'] < self.volatility_low:
            return MarketRegime.LOW_VOLATILITY
        
        # Trend strength
        adx = latest['adx']
        
        if adx > self.adx_trend:
            # Trending market - determine direction
            if latest['price_above_ema50'] and latest['price_above_ema200']:
                if adx > self.adx_strong:
                    return MarketRegime.STRONG_UPTREND
                return MarketRegime.UPTREND
            elif not latest['price_above_ema50'] and not latest['price_above_ema200']:
                if adx > self.adx_strong:
                    return MarketRegime.STRONG_DOWNTREND
                return MarketRegime.DOWNTREND
        
        # Ranging market
        return MarketRegime.RANGING
    
    def detect_regime_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime for entire series
        """
        df = self.calculate_regime_indicators(df)
        regimes = pd.Series(index=df.index, dtype=object)
        
        for i in range(self.lookback, len(df)):
            window = df.iloc[i-self.lookback:i+1]
            regimes.iloc[i] = self.detect_regime(window).value
        
        return regimes
    
    def calculate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime features to dataframe
        """
        df = df.copy()
        
        # Regime detection
        df['market_regime'] = self.detect_regime_series(df)
        
        # One-hot encoding for regimes
        for regime in MarketRegime:
            df[f'is_{regime.value}'] = (df['market_regime'] == regime.value).astype(int)
        
        # Regime stability (how long current regime has persisted)
        df['regime_change'] = (df['market_regime'] != df['market_regime'].shift(1)).astype(int)
        df['regime_duration'] = df.groupby(df['regime_change'].cumsum()).cumcount() + 1
        
        # Trend quality score (0-1)
        df['trend_quality'] = (
            df['adx'] / 100 * 0.4 +
            (df['higher_highs'].rolling(10).mean() if 'higher_highs' in df.columns else 0.5) * 0.3 +
            (df['price_above_ema50'] if 'price_above_ema50' in df.columns else 0.5) * 0.3
        )
        
        return df
    
    def get_regime_metrics(self, df: pd.DataFrame) -> RegimeMetrics:
        """
        Get detailed metrics for current regime
        """
        regime = self.detect_regime(df)
        latest = df.iloc[-1]
        
        # Calculate confidence based on indicator clarity
        if regime in [MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND]:
            confidence = min(latest['adx'] / 100, 0.95)
        elif regime in [MarketRegime.UPTREND, MarketRegime.DOWNTREND]:
            confidence = min(latest['adx'] / 75, 0.85)
        elif regime == MarketRegime.RANGING:
            confidence = max(0, 1 - latest['adx'] / 50)
        else:
            confidence = 0.7
        
        # Duration in current regime
        regimes = self.detect_regime_series(df)
        current_regime = regimes.iloc[-1]
        duration = 0
        for i in range(len(regimes) - 1, -1, -1):
            if regimes.iloc[i] == current_regime:
                duration += 1
            else:
                break
        
        return RegimeMetrics(
            regime=regime,
            confidence=confidence,
            adx=latest.get('adx', 0),
            volatility=latest.get('atr_normalized', 1),
            trend_strength=latest.get('adx', 0) / 100,
            duration=duration
        )
    
    def regime_based_filter(
        self,
        df: pd.DataFrame,
        allowed_regimes: List[MarketRegime]
    ) -> pd.DataFrame:
        """
        Filter dataframe to only include allowed regimes
        """
        df = self.calculate_regime_features(df)
        regime_values = [r.value for r in allowed_regimes]
        return df[df['market_regime'].isin(regime_values)]


class MLRegimeDetector(RegimeDetector):
    """
    Machine learning-based regime detection using Gaussian Mixture Models
    """
    
    def __init__(self, config_path: str = "config/settings.yaml", n_regimes: int = 4):
        super().__init__(config_path)
        self.n_regimes = n_regimes
        self.model = GaussianMixture(n_components=n_regimes, random_state=42)
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame):
        """
        Fit GMM model on historical data
        """
        features = self._extract_features(df)
        self.model.fit(features)
        self.is_fitted = True
        logger.info(f"Fitted GMM with {self.n_regimes} regimes on {len(features)} samples")
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features for ML model
        """
        df = self.calculate_regime_indicators(df)
        
        features = pd.DataFrame({
            'adx': df['adx'],
            'atr_normalized': df['atr_normalized'],
            'returns_volatility': df['close'].pct_change().rolling(20).std(),
            'trend_slope': df['close'].ewm(span=20).mean().diff(5),
            'price_vs_ema50': (df['close'] - df['close'].ewm(span=50).mean()) / df['close']
        }).dropna()
        
        return features.values
    
    def predict_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict regime using fitted model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        features = self._extract_features(df)
        predictions = self.model.predict(features)
        
        # Map clusters to regime labels based on characteristics
        regime_series = pd.Series(index=df.index[-len(predictions):], dtype=object)
        
        for i, pred in enumerate(predictions):
            # Simple mapping based on feature values
            feat = features[i]
            adx, vol = feat[0], feat[1]
            
            if adx > 40 and vol < 1.2:
                regime_series.iloc[i] = MarketRegime.STRONG_UPTREND.value
            elif adx > 25:
                regime_series.iloc[i] = MarketRegime.UPTREND.value
            elif vol > 1.5:
                regime_series.iloc[i] = MarketRegime.HIGH_VOLATILITY.value
            else:
                regime_series.iloc[i] = MarketRegime.RANGING.value
        
        return regime_series


# Convenience functions
def detect_current_regime(df: pd.DataFrame) -> str:
    """Quick regime detection"""
    detector = RegimeDetector()
    return detector.detect_regime(df).value

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime features to dataframe"""
    detector = RegimeDetector()
    return detector.calculate_regime_features(df)

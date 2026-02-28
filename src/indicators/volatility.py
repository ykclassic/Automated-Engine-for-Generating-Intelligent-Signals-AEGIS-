"""
AEGIS Volatility Indicators Module
Volatility measurement and bands
"""

import logging
from typing import Dict, List

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class VolatilityIndicators:
    """
    Volatility-based indicators
    """
    
    def __init__(self, config_path: str = "config/indicators.yaml"):
        self.config = self._load_config(config_path)
        self.vol_config = self.config.get('indicators', {}).get('volatility', {})
    
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
        
        return atr
    
    def bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = 'close'
    ) -> Dict[str, pd.Series]:
        """
        Bollinger Bands
        """
        sma = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        # Bandwidth and %B
        bandwidth = (upper - lower) / sma
        percent_b = (df[column] - lower) / (upper - lower)
        
        return {
            'middle': sma,
            'upper': upper,
            'lower': lower,
            'bandwidth': bandwidth,
            'percent_b': percent_b
        }
    
    def keltner_channels(
        self,
        df: pd.DataFrame,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Keltner Channels
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        ema = typical_price.ewm(span=ema_period, adjust=False).mean()
        
        atr = self.atr(df, atr_period)
        
        upper = ema + (multiplier * atr)
        lower = ema - (multiplier * atr)
        
        return {
            'middle': ema,
            'upper': upper,
            'lower': lower
        }
    
    def donchian_channels(
        self,
        df: pd.DataFrame,
        period: int = 20
    ) -> Dict[str, pd.Series]:
        """
        Donchian Channels
        """
        upper = df['high'].rolling(window=period).max()
        lower = df['low'].rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all volatility indicators
        """
        df = df.copy()
        
        # ATR
        atr_period = self.vol_config.get('atr', {}).get('period', 14)
        df['atr'] = self.atr(df, atr_period)
        df['atr_pct'] = df['atr'] / df['close']
        df['atr_normalized'] = df['atr'] / df['atr'].rolling(50).mean()
        
        # ATR-based volatility regime
        df['volatility_low'] = (df['atr_normalized'] < 0.8).astype(int)
        df['volatility_high'] = (df['atr_normalized'] > 1.2).astype(int)
        
        # Bollinger Bands
        bb_config = self.vol_config.get('bollinger_bands', {})
        bb_result = self.bollinger_bands(
            df,
            period=bb_config.get('period', 20),
            std_dev=bb_config.get('std_dev', 2.0)
        )
        df['bb_middle'] = bb_result['middle']
        df['bb_upper'] = bb_result['upper']
        df['bb_lower'] = bb_result['lower']
        df['bb_bandwidth'] = bb_result['bandwidth']
        df['bb_percent_b'] = bb_result['percent_b']
        
        # Bollinger signals
        df['bb_squeeze'] = (df['bb_bandwidth'] < df['bb_bandwidth'].rolling(50).quantile(0.1)).astype(int)
        df['price_above_bb_upper'] = (df['close'] > df['bb_upper']).astype(int)
        df['price_below_bb_lower'] = (df['close'] < df['bb_lower']).astype(int)
        
        # Keltner Channels
        kc_config = self.vol_config.get('keltner_channels', {})
        kc_result = self.keltner_channels(
            df,
            ema_period=kc_config.get('ema_period', 20),
            atr_period=kc_config.get('atr_period', 10),
            multiplier=kc_config.get('multiplier', 2.0)
        )
        df['kc_middle'] = kc_result['middle']
        df['kc_upper'] = kc_result['upper']
        df['kc_lower'] = kc_result['lower']
        
        # Donchian Channels
        df['dc_upper'] = df['high'].rolling(window=20).max()
        df['dc_lower'] = df['low'].rolling(window=20).min()
        df['dc_middle'] = (df['dc_upper'] + df['dc_lower']) / 2
        
        # Historical volatility (annualized)
        for window in [10, 20, 50]:
            df[f'hist_vol_{window}'] = (
                df['close'].pct_change().rolling(window=window).std() * 
                np.sqrt(365 * 24)  # Annualized for hourly data
            )
        
        # Volatility regime detection
        df['volatility_regime'] = pd.cut(
            df['atr_normalized'],
            bins=[0, 0.5, 0.8, 1.2, 2.0, float('inf')],
            labels=['very_low', 'low', 'normal', 'high', 'extreme']
        )
        
        return df

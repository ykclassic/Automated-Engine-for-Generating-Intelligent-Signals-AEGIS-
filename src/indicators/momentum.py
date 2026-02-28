"""
AEGIS Momentum Indicators Module
Oscillators and momentum measurements
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class MomentumIndicators:
    """
    Momentum oscillators and indicators
    """
    
    def __init__(self, config_path: str = "config/indicators.yaml"):
        self.config = self._load_config(config_path)
        self.momentum_config = self.config.get('indicators', {}).get('momentum', {})
    
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def rsi(self, df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Relative Strength Index
        """
        delta = df[column].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use Wilder's smoothing (RMA)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        slowing: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Stochastic Oscillator
        """
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        # Fast %K
        k_fast = 100 * (df['close'] - low_min) / (high_max - low_min)
        
        # Slow %K (with slowing)
        k_slow = k_fast.rolling(window=slowing).mean()
        
        # %D (signal line)
        d_slow = k_slow.rolling(window=d_period).mean()
        
        return {
            'k_fast': k_fast,
            'k_slow': k_slow,
            'd_slow': d_slow
        }
    
    def cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index
        """
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_sma = tp.rolling(window=period).mean()
        mean_dev = tp.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        cci = (tp - tp_sma) / (0.015 * mean_dev)
        return cci
    
    def williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Williams %R
        """
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        
        williams_r = -100 * (high_max - df['close']) / (high_max - low_min)
        return williams_r
    
    def awesome_oscillator(self, df: pd.DataFrame) -> pd.Series:
        """
        Awesome Oscillator
        """
        median_price = (df['high'] + df['low']) / 2
        ao = (
            median_price.rolling(window=5).mean() - 
            median_price.rolling(window=34).mean()
        )
        return ao
    
    def ultimate_oscillator(self, df: pd.DataFrame) -> pd.Series:
        """
        Ultimate Oscillator (Larry Williams)
        Combines short, medium, and long-term momentum
        """
        close = df['close']
        low = df['low']
        high = df['high']
        
        # Buying pressure and true range
        bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        # Averages for 7, 14, and 28 periods
        avg7 = bp.rolling(window=7).sum() / tr.rolling(window=7).sum()
        avg14 = bp.rolling(window=14).sum() / tr.rolling(window=14).sum()
        avg28 = bp.rolling(window=28).sum() / tr.rolling(window=28).sum()
        
        uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
        return uo
    
    def rate_of_change(self, df: pd.DataFrame, period: int = 12, column: str = 'close') -> pd.Series:
        """
        Rate of Change (ROC)
        """
        roc = ((df[column] - df[column].shift(period)) / df[column].shift(period)) * 100
        return roc
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all momentum indicators
        """
        df = df.copy()
        
        # RSI
        rsi_config = self.momentum_config.get('rsi', {})
        rsi_period = rsi_config.get('period', 14)
        df['rsi'] = self.rsi(df, rsi_period)
        df['rsi_normalized'] = df['rsi'] / 100
        
        # RSI conditions
        rsi_levels = rsi_config.get('levels', [30, 50, 70])
        df['rsi_oversold'] = (df['rsi'] < rsi_levels[0]).astype(int)
        df['rsi_overbought'] = (df['rsi'] > rsi_levels[2]).astype(int)
        df['rsi_bullish'] = (df['rsi'] > rsi_levels[1]).astype(int)
        
        # RSI slope (momentum of momentum)
        df['rsi_slope'] = df['rsi'].diff(3)
        df['rsi_divergence'] = np.where(
            (df['close'] > df['close'].shift(5)) & (df['rsi'] < df['rsi'].shift(5)),
            -1,  # Bearish divergence
            np.where(
                (df['close'] < df['close'].shift(5)) & (df['rsi'] > df['rsi'].shift(5)),
                1,   # Bullish divergence
                0
            )
        )
        
        # Stochastic
        stoch_config = self.momentum_config.get('stochastic', {})
        stoch_result = self.stochastic(
            df,
            k_period=stoch_config.get('k_period', 14),
            d_period=stoch_config.get('d_period', 3),
            slowing=stoch_config.get('slowing', 3)
        )
        df['stoch_k'] = stoch_result['k_slow']
        df['stoch_d'] = stoch_result['d_slow']
        df['stoch_cross'] = ((df['stoch_k'] > df['stoch_d']) & 
                             (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))).astype(int)
        
        stoch_levels = stoch_config.get('overbought', 80), stoch_config.get('oversold', 20)
        df['stoch_overbought'] = (df['stoch_k'] > stoch_levels[0]).astype(int)
        df['stoch_oversold'] = (df['stoch_k'] < stoch_levels[1]).astype(int)
        
        # CCI
        cci_config = self.momentum_config.get('cci', {})
        df['cci'] = self.cci(df, cci_config.get('period', 20))
        cci_levels = cci_config.get('overbought', 100), cci_config.get('oversold', -100)
        df['cci_overbought'] = (df['cci'] > cci_levels[0]).astype(int)
        df['cci_oversold'] = (df['cci'] < cci_levels[1]).astype(int)
        
        # Williams %R
        wr_config = self.momentum_config.get('williams_r', {})
        df['williams_r'] = self.williams_r(df, wr_config.get('period', 14))
        wr_levels = wr_config.get('overbought', -20), wr_config.get('oversold', -80)
        df['wr_overbought'] = (df['williams_r'] > wr_levels[0]).astype(int)
        df['wr_oversold'] = (df['williams_r'] < wr_levels[1]).astype(int)
        
        # Awesome Oscillator
        df['awesome_oscillator'] = self.awesome_oscillator(df)
        df['ao_color'] = np.where(df['awesome_oscillator'] > df['awesome_oscillator'].shift(1), 1, -1)
        df['ao_saucer'] = (
            (df['awesome_oscillator'] > 0) & 
            (df['ao_color'] == 1) & 
            (df['ao_color'].shift(1) == -1) & 
            (df['ao_color'].shift(2) == -1)
        ).astype(int)
        
        # Ultimate Oscillator
        df['ultimate_oscillator'] = self.ultimate_oscillator(df)
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = self.rate_of_change(df, period)
        
        # Composite momentum score
        # Normalizes all oscillators to 0-1 scale and averages
        df['momentum_score'] = (
            df['rsi_normalized'] * 0.3 +
            (df['stoch_k'] / 100) * 0.25 +
            ((df['cci'] + 200) / 400).clip(0, 1) * 0.2 +
            ((df['williams_r'] + 100) / 100).clip(0, 1) * 0.15 +
            ((df['ultimate_oscillator'] / 100).clip(0, 1)) * 0.1
        )
        
        return df

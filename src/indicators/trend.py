"""
AEGIS Trend Indicators Module
Trend-following and directional indicators
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class TrendIndicators:
    """
    Collection of trend-following indicators
    All calculations are causal (no future data used)
    """
    
    def __init__(self, config_path: str = "config/indicators.yaml"):
        self.config = self._load_config(config_path)
        self.trend_config = self.config.get('indicators', {}).get('trend', {})
    
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def ema(self, df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        Exponential Moving Average
        """
        return df[column].ewm(span=period, adjust=False, min_periods=period).mean()
    
    def sma(self, df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        Simple Moving Average
        """
        return df[column].rolling(window=period, min_periods=period).mean()
    
    def macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = 'close'
    ) -> Dict[str, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)
        Returns: macd_line, signal_line, histogram
        """
        ema_fast = self.ema(df, fast, column)
        ema_slow = self.ema(df, slow, column)
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }
    
    def adx(
        self,
        df: pd.DataFrame,
        period: int = 14,
        smoothing: int = 14
    ) -> Dict[str, pd.Series]:
        """
        Average Directional Index (ADX)
        Measures trend strength (not direction)
        Returns: adx, plus_di, minus_di
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        plus_dm[plus_dm <= minus_dm] = 0
        minus_dm[minus_dm <= plus_dm] = 0
        
        # Smoothed averages
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * plus_dm.rolling(window=period).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=period).mean() / atr
        
        # Directional Index
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=smoothing).mean()
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    def ichimoku(
        self,
        df: pd.DataFrame,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52,
        displacement: int = 26
    ) -> Dict[str, pd.Series]:
        """
        Ichimoku Cloud indicator
        Comprehensive trend system with support/resistance
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Tenkan-sen (Conversion Line): (Highest High + Lowest Low) / 2 for 9 periods
        tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2
        
        # Kijun-sen (Base Line): (Highest High + Lowest Low) / 2 for 26 periods
        kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line) / 2, displaced 26 periods forward
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B): (Highest High + Lowest Low) / 2 for 52 periods, displaced 26 periods forward
        senkou_span_b = ((high.rolling(window=senkou_b).max() + low.rolling(window=senkou_b).min()) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span): Close price displaced 26 periods backward
        chikou_span = close.shift(-displacement)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    def parabolic_sar(
        self,
        df: pd.DataFrame,
        af_start: float = 0.02,
        af_increment: float = 0.02,
        af_max: float = 0.2
    ) -> pd.Series:
        """
        Parabolic Stop and Reverse
        Trend following indicator
        """
        high = df['high']
        low = df['low']
        
        # Initialize
        sar = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)  # 1 for uptrend, -1 for downtrend
        ep = pd.Series(index=df.index, dtype=float)   # Extreme point
        af = pd.Series(index=df.index, dtype=float)   # Acceleration factor
        
        # Starting values (assume uptrend)
        sar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1
        ep.iloc[0] = high.iloc[0]
        af.iloc[0] = af_start
        
        for i in range(1, len(df)):
            # Previous values
            prev_sar = sar.iloc[i-1]
            prev_trend = trend.iloc[i-1]
            prev_ep = ep.iloc[i-1]
            prev_af = af.iloc[i-1]
            
            # Calculate new SAR
            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
            
            # Uptrend logic
            if prev_trend == 1:
                # Ensure SAR is below recent lows
                new_sar = min(new_sar, low.iloc[i-1], low.iloc[max(0, i-2)])
                
                # Check for trend reversal
                if new_sar > low.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = prev_ep
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = af_start
                else:
                    trend.iloc[i] = 1
                    sar.iloc[i] = new_sar
                    
                    # Update extreme point and acceleration factor
                    if high.iloc[i] > prev_ep:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(prev_af + af_increment, af_max)
                    else:
                        ep.iloc[i] = prev_ep
                        af.iloc[i] = prev_af
            
            # Downtrend logic
            else:
                # Ensure SAR is above recent highs
                new_sar = max(new_sar, high.iloc[i-1], high.iloc[max(0, i-2)])
                
                # Check for trend reversal
                if new_sar < high.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = prev_ep
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = af_start
                else:
                    trend.iloc[i] = -1
                    sar.iloc[i] = new_sar
                    
                    # Update extreme point and acceleration factor
                    if low.iloc[i] < prev_ep:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(prev_af + af_increment, af_max)
                    else:
                        ep.iloc[i] = prev_ep
                        af.iloc[i] = prev_af
        
        return sar
    
    def supertrend(
        self,
        df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0
    ) -> Dict[str, pd.Series]:
        """
        SuperTrend indicator
        Trend following with dynamic support/resistance
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # ATR calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Basic Upper and Lower Bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Final Bands and SuperTrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)  # 1 for uptrend, -1 for downtrend
        
        for i in range(period, len(df)):
            if close.iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
                
                if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i-1]:
                    lower_band.iloc[i] = lower_band.iloc[i-1]
                if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i-1]:
                    upper_band.iloc[i] = upper_band.iloc[i-1]
            
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
        
        return {
            'supertrend': supertrend,
            'direction': direction,
            'upper_band': upper_band,
            'lower_band': lower_band
        }
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all trend indicators
        """
        df = df.copy()
        
        # EMAs
        ema_periods = self.trend_config.get('ema', {}).get('periods', [8, 21, 50, 200])
        for period in ema_periods:
            df[f'ema_{period}'] = self.ema(df, period)
            # EMA slope (rate of change)
            df[f'ema_{period}_slope'] = df[f'ema_{period}'].diff(3) / 3
        
        # SMAs
        sma_periods = self.trend_config.get('sma', {}).get('periods', [50, 200])
        for period in sma_periods:
            df[f'sma_{period}'] = self.sma(df, period)
        
        # MACD
        macd_config = self.trend_config.get('macd', {})
        macd_result = self.macd(
            df,
            fast=macd_config.get('fast', 12),
            slow=macd_config.get('slow', 26),
            signal=macd_config.get('signal', 9)
        )
        df['macd_line'] = macd_result['macd_line']
        df['macd_signal'] = macd_result['signal_line']
        df['macd_histogram'] = macd_result['histogram']
        df['macd_above_signal'] = (df['macd_line'] > df['macd_signal']).astype(int)
        
        # ADX
        adx_config = self.trend_config.get('adx', {})
        adx_result = self.adx(
            df,
            period=adx_config.get('period', 14),
            smoothing=adx_config.get('smoothing', 14)
        )
        df['adx'] = adx_result['adx']
        df['plus_di'] = adx_result['plus_di']
        df['minus_di'] = adx_result['minus_di']
        df['di_difference'] = df['plus_di'] - df['minus_di']
        
        # ADX trend strength classification
        adx_levels = adx_config.get('levels', {})
        df['adx_trend_weak'] = (df['adx'] < adx_levels.get('weak', 25)).astype(int)
        df['adx_trend_strong'] = (df['adx'] >= adx_levels.get('strong', 50)).astype(int)
        
        # Ichimoku
        ichimoku_config = self.trend_config.get('ichimoku', {})
        ichimoku_result = self.ichimoku(
            df,
            tenkan=ichimoku_config.get('tenkan', 9),
            kijun=ichimoku_config.get('kijun', 26),
            senkou_b=ichimoku_config.get('senkou_b', 52),
            displacement=ichimoku_config.get('displacement', 26)
        )
        df['tenkan_sen'] = ichimoku_result['tenkan_sen']
        df['kijun_sen'] = ichimoku_result['kijun_sen']
        df['senkou_span_a'] = ichimoku_result['senkou_span_a']
        df['senkou_span_b'] = ichimoku_result['senkou_span_b']
        df['chikou_span'] = ichimoku_result['chikou_span']
        
        # Ichimoku signals
        df['price_above_cloud'] = (df['close'] > df[['senkou_span_a', 'senkou_span_b']].max(axis=1)).astype(int)
        df['price_below_cloud'] = (df['close'] < df[['senkou_span_a', 'senkou_span_b']].min(axis=1)).astype(int)
        df['tk_cross_bullish'] = ((df['tenkan_sen'] > df['kijun_sen']) & 
                                   (df['tenkan_sen'].shift(1) <= df['kijun_sen'].shift(1))).astype(int)
        
        # SuperTrend
        supertrend_result = self.supertrend(df)
        df['supertrend'] = supertrend_result['supertrend']
        df['supertrend_direction'] = supertrend_result['direction']
        
        # Trend alignment score (how many indicators agree on trend)
        df['trend_bullish_score'] = (
            (df['ema_8'] > df['ema_21']).astype(int) +
            (df['ema_21'] > df['ema_50']).astype(int) +
            (df['macd_line'] > df['macd_signal']).astype(int) +
            (df['plus_di'] > df['minus_di']).astype(int) +
            df['price_above_cloud'] +
            (df['supertrend_direction'] == 1).astype(int)
        ) / 6.0
        
        return df

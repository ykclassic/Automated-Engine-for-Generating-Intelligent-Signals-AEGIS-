"""
AEGIS Volume Indicators Module
Volume-based analysis and confirmation
"""

import logging
from typing import Dict

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class VolumeIndicators:
    """
    Volume analysis indicators
    """
    
    def __init__(self, config_path: str = "config/indicators.yaml"):
        self.config = self._load_config(config_path)
        self.vol_config = self.config.get('indicators', {}).get('volume', {})
    
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def obv(self, df: pd.DataFrame) -> pd.Series:
        """
        On-Balance Volume
        """
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def vwap(self, df: pd.DataFrame, anchor: str = 'D') -> pd.Series:
        """
        Volume Weighted Average Price
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).groupby(pd.Grouper(freq=anchor)).cumsum() / \
               df['volume'].groupby(pd.Grouper(freq=anchor)).cumsum()
        return vwap
    
    def mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Money Flow Index
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        
        money_flow_sign = np.where(typical_price > typical_price.shift(1), 1, -1)
        money_flow = raw_money_flow * money_flow_sign
        
        positive_flow = pd.Series(np.where(money_flow > 0, money_flow, 0), index=df.index)
        negative_flow = pd.Series(np.where(money_flow < 0, -money_flow, 0), index=df.index)
        
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()
        
        money_ratio = positive_sum / negative_sum
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    def volume_profile(self, df: pd.DataFrame, lookback: int = 100, bins: int = 10) -> Dict[str, pd.Series]:
        """
        Volume Profile - Point of Control and Value Areas
        """
        poc = pd.Series(index=df.index, dtype=float)
        value_area_high = pd.Series(index=df.index, dtype=float)
        value_area_low = pd.Series(index=df.index, dtype=float)
        
        for i in range(lookback, len(df)):
            window = df.iloc[i-lookback:i]
            
            # Create price bins
            price_min = window['low'].min()
            price_max = window['high'].max()
            price_bins = np.linspace(price_min, price_max, bins)
            
            # Calculate volume per bin
            bin_volumes = np.zeros(bins - 1)
            for j in range(len(price_bins) - 1):
                mask = (window['close'] >= price_bins[j]) & (window['close'] < price_bins[j+1])
                bin_volumes[j] = window.loc[mask, 'volume'].sum()
            
            # Point of Control (price with highest volume)
            poc_idx = np.argmax(bin_volumes)
            poc.iloc[i] = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
            
            # Value Area (70% of volume)
            total_volume = bin_volumes.sum()
            target_volume = total_volume * 0.7
            
            sorted_indices = np.argsort(bin_volumes)[::-1]
            cumulative_volume = 0
            value_bins = []
            
            for idx in sorted_indices:
                cumulative_volume += bin_volumes[idx]
                value_bins.append(idx)
                if cumulative_volume >= target_volume:
                    break
            
            value_area_high.iloc[i] = max([price_bins[i+1] for i in value_bins])
            value_area_low.iloc[i] = min([price_bins[i] for i in value_bins])
        
        return {
            'poc': poc,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low
        }
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all volume indicators
        """
        df = df.copy()
        
        # Volume moving averages
        for window in [5, 20, 50]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
        
        # Relative volume (compared to average)
        df['relative_volume'] = df['volume'] / df['volume_sma_20']
        df['high_volume'] = (df['relative_volume'] > 2.0).astype(int)
        
        # On-Balance Volume
        df['obv'] = self.obv(df)
        df['obv_sma'] = df['obv'].rolling(window=20).mean()
        df['obv_trend'] = (df['obv'] > df['obv_sma']).astype(int)
        
        # OBV divergence
        df['obv_divergence'] = np.where(
            (df['close'] > df['close'].shift(5)) & (df['obv'] < df['obv'].shift(5)),
            -1,
            np.where(
                (df['close'] < df['close'].shift(5)) & (df['obv'] > df['obv'].shift(5)),
                1,
                0
            )
        )
        
        # VWAP
        df['vwap'] = self.vwap(df)
        df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        df['above_vwap'] = (df['close'] > df['vwap']).astype(int)
        
        # MFI
        mfi_config = self.vol_config.get('mfi', {})
        df['mfi'] = self.mfi(df, mfi_config.get('period', 14))
        mfi_levels = mfi_config.get('overbought', 80), mfi_config.get('oversold', 20)
        df['mfi_overbought'] = (df['mfi'] > mfi_levels[0]).astype(int)
        df['mfi_oversold'] = (df['mfi'] < mfi_levels[1]).astype(int)
        
        # Volume trend
        df['volume_trend'] = np.where(
            df['volume'] > df['volume'].shift(1) * 1.5,
            2,  # Strong increase
            np.where(
                df['volume'] > df['volume'].shift(1),
                1,  # Moderate increase
                np.where(
                    df['volume'] < df['volume'].shift(1) * 0.5,
                    -2,  # Strong decrease
                    np.where(
                        df['volume'] < df['volume'].shift(1),
                        -1,  # Moderate decrease
                        0
                    )
                )
            )
        )
        
        # Volume-Price confirmation
        df['volume_price_confirmation'] = np.where(
            (df['close'] > df['close'].shift(1)) & (df['volume'] > df['volume'].shift(1)),
            1,  # Bullish confirmation
            np.where(
                (df['close'] < df['close'].shift(1)) & (df['volume'] > df['volume'].shift(1)),
                -1,  # Bearish confirmation
                0
            )
        )
        
        return df

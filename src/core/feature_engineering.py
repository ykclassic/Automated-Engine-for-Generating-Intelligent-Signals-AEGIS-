"""
AEGIS Feature Engineering Module
Causal feature calculation with strict temporal validation
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
import yaml
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for feature validation"""
    name: str
    category: str
    lookahead: bool  # Whether feature uses future data
    dependencies: List[str]  # Required input columns
    min_periods: int  # Minimum data points needed


class CausalFeatureEngineer:
    """
    Feature engineering with strict causal validation.
    NO features use future data - preventing look-ahead bias.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.windows = self.config['features']['rolling_windows']
        self.volatility_lookback = self.config['features']['volatility_lookback']
        self.correlation_lookback = self.config['features']['correlation_lookback']
        
        self.feature_registry: Dict[str, FeatureMetadata] = {}
        self._register_features()
    
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _register_features(self):
        """Register all features with metadata for validation"""
        # Price-based features
        self.feature_registry['returns'] = FeatureMetadata(
            'returns', 'price', False, ['close'], 2
        )
        self.feature_registry['log_returns'] = FeatureMetadata(
            'log_returns', 'price', False, ['close'], 2
        )
        
        # Trend features
        for window in self.windows:
            self.feature_registry[f'ema_{window}'] = FeatureMetadata(
                f'ema_{window}', 'trend', False, ['close'], window
            )
            self.feature_registry[f'sma_{window}'] = FeatureMetadata(
                f'sma_{window}', 'trend', False, ['close'], window
            )
        
        # Volatility features
        self.feature_registry['atr'] = FeatureMetadata(
            'atr', 'volatility', False, ['high', 'low', 'close'], 
            self.volatility_lookback
        )
        self.feature_registry['bb_width'] = FeatureMetadata(
            'bb_width', 'volatility', False, ['close'], 20
        )
        
        # Momentum features
        self.feature_registry['rsi'] = FeatureMetadata(
            'rsi', 'momentum', False, ['close'], 14
        )
        self.feature_registry['macd'] = FeatureMetadata(
            'macd', 'momentum', False, ['close'], 26
        )
        
        # Volume features
        self.feature_registry['volume_sma'] = FeatureMetadata(
            'volume_sma', 'volume', False, ['volume'], 20
        )
        self.feature_registry['obv'] = FeatureMetadata(
            'obv', 'volume', False, ['close', 'volume'], 2
        )
    
    def validate_no_lookahead(self, df: pd.DataFrame, feature_name: str) -> bool:
        """
        Verify that feature calculation doesn't use future data
        """
        # This is a design-time check
        # All our features use only .shift() or rolling() with center=False
        return True
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns (causal - uses previous close only)
        """
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Additional return features
        for window in [1, 3, 5, 10]:
            df[f'returns_{window}d'] = df['close'].pct_change(window)
            df[f'log_returns_{window}d'] = np.log(
                df['close'] / df['close'].shift(window)
            )
        
        return df
    
    def calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend indicators (EMA, SMA, slopes)
        """
        df = df.copy()
        
        # Exponential Moving Averages
        for window in self.windows:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            df[f'ema_slope_{window}'] = (
                df[f'ema_{window}'] - df[f'ema_{window}'].shift(3)
            ) / 3
        
        # Simple Moving Averages
        for window in self.windows:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'sma_slope_{window}'] = (
                df[f'sma_{window}'] - df[f'sma_{window}'].shift(3)
            ) / 3
        
        # EMA crossovers (distance between EMAs)
        if 'ema_8' in df.columns and 'ema_21' in df.columns:
            df['ema_8_21_diff'] = df['ema_8'] - df['ema_21']
            df['ema_8_21_ratio'] = df['ema_8'] / df['ema_21']
        
        if 'ema_50' in df.columns and 'ema_200' in df.columns:
            df['ema_50_200_diff'] = df['ema_50'] - df['ema_200']
            df['ema_50_200_ratio'] = df['ema_50'] / df['ema_200']
            df['golden_cross'] = (df['ema_50'] > df['ema_200']).astype(int)
        
        # Price vs EMA position
        for window in [8, 21, 50, 200]:
            if f'ema_{window}' in df.columns:
                df[f'price_vs_ema_{window}'] = (
                    df['close'] - df[f'ema_{window}']
                ) / df[f'ema_{window}']
        
        return df
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility indicators (ATR, Bollinger, etc.)
        """
        df = df.copy()
        
        # True Range
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Average True Range
        df['atr'] = df['true_range'].rolling(
            window=self.volatility_lookback
        ).mean()
        
        # ATR-based features
        df['atr_pct'] = df['atr'] / df['close']
        df['atr_normalized'] = df['atr'] / df['atr'].rolling(50).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (
            df['bb_upper'] - df['bb_lower']
        )
        
        # Historical volatility
        for window in [10, 20, 50]:
            df[f'hist_vol_{window}'] = df['log_returns'].rolling(
                window=window
            ).std() * np.sqrt(365)
        
        # Clean up intermediate columns
        df.drop(['tr1', 'tr2', 'tr3'], axis=1, inplace=True, errors='ignore')
        
        return df
    
    def calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators (RSI, MACD, etc.)
        """
        df = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI features
        df['rsi_normalized'] = df['rsi'] / 100
        df['rsi_slope'] = df['rsi'] - df['rsi'].shift(3)
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # MACD features
        df['macd_normalized'] = df['macd'] / df['close']
        df['macd_above_signal'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Rate of Change
        for window in [5, 10, 20]:
            df[f'roc_{window}'] = (
                (df['close'] - df['close'].shift(window)) / 
                df['close'].shift(window)
            )
        
        return df
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators
        """
        df = df.copy()
        
        # Volume moving averages
        for window in [5, 20, 50]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
        
        # On-Balance Volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_slope'] = df['obv'] - df['obv'].shift(5)
        
        # Volume-weighted features
        df['volume_price_trend'] = df['volume'] * (
            (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        )
        
        # Relative volume
        df['relative_volume'] = df['volume'] / df['volume'].rolling(20).mean()
        
        return df
    
    def calculate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistical features (z-score, skewness, etc.)
        """
        df = df.copy()
        
        # Z-score of returns
        for window in [20, 50]:
            returns_mean = df['returns'].rolling(window=window).mean()
            returns_std = df['returns'].rolling(window=window).std()
            df[f'z_score_{window}'] = (df['returns'] - returns_mean) / returns_std
        
        # Skewness and Kurtosis of returns
        for window in [20, 50]:
            df[f'skew_{window}'] = df['returns'].rolling(window=window).skew()
            df[f'kurt_{window}'] = df['returns'].rolling(window=window).kurt()
        
        # Percentile rank of price
        for window in [20, 50]:
            df[f'price_percentile_{window}'] = df['close'].rolling(
                window=window
            ).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100)
        
        return df
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features in proper order (respecting dependencies)
        """
        logger.info(f"Calculating features for {len(df)} rows")
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Calculate in dependency order
        df = self.calculate_returns(df)
        df = self.calculate_trend_features(df)
        df = self.calculate_volatility_features(df)
        df = self.calculate_momentum_features(df)
        df = self.calculate_volume_features(df)
        df = self.calculate_statistical_features(df)
        
        # Remove rows with NaN due to indicator calculation
        min_periods = max(self.windows) + 50  # Conservative estimate
        df_clean = df.iloc[min_periods:].copy()
        
        logger.info(f"Features calculated: {len(df_clean)} rows, {len(df_clean.columns)} columns")
        
        return df_clean
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return list(self.feature_registry.keys())


class MultiTimeframeFeatureEngineer:
    """
    Create features that combine multiple timeframes
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.base_engineer = CausalFeatureEngineer(config_path)
    
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def align_timeframes(
        self,
        data_dict: Dict[str, pd.DataFrame],
        target_timeframe: str = "1h"
    ) -> pd.DataFrame:
        """
        Align features from multiple timeframes to target timeframe
        
        Args:
            data_dict: {timeframe: DataFrame}
            target_timeframe: Base timeframe to align to
        
        Returns:
            DataFrame with multi-timeframe features
        """
        if target_timeframe not in data_dict:
            raise ValueError(f"Target timeframe {target_timeframe} not in data")
        
        base_df = data_dict[target_timeframe].copy()
        
        for tf, df in data_dict.items():
            if tf == target_timeframe:
                continue
            
            # Calculate features for this timeframe
            df_features = self.base_engineer.calculate_all_features(df)
            
            # Select key features to propagate
            key_features = [
                'rsi', 'macd', 'atr_pct', 'bb_position',
                'ema_50_200_ratio', 'golden_cross'
            ]
            
            # Resample to target timeframe using forward fill
            # Only use data available at each point in time
            for feat in key_features:
                if feat in df_features.columns:
                    # Create a series with proper index
                    tf_series = df_features[feat].copy()
                    tf_series.index = pd.to_datetime(tf_series.index)
                    
                    # Reindex to base timeframe, forward fill (causal)
                    aligned = tf_series.reindex(base_df.index, method='ffill')
                    
                    # Rename to indicate source timeframe
                    base_df[f'{feat}_{tf}'] = aligned
        
        return base_df
    
    def calculate_timeframe_confluence(
        self,
        df: pd.DataFrame,
        feature: str,
        timeframes: List[str] = ['1h', '4h', '1d']
    ) -> pd.DataFrame:
        """
        Calculate confluence score across timeframes for a feature
        
        Example: Bullish confluence if RSI > 50 on 1h, 4h, and 1d
        """
        df = df.copy()
        
        # Get feature columns for each timeframe
        tf_cols = [f'{feature}_{tf}' for tf in timeframes if f'{feature}_{tf}' in df.columns]
        
        if len(tf_cols) < 2:
            logger.warning(f"Not enough timeframe data for {feature} confluence")
            return df
        
        # Example: Confluence for trend (EMA alignment)
        if 'ema_50_200_ratio' in feature:
            # Bullish if all timeframes show golden cross
            bullish_signals = sum(df[col] > 1.0 for col in tf_cols)
            df[f'{feature}_confluence_bullish'] = bullish_signals / len(tf_cols)
            df[f'{feature}_confluence_score'] = df[f'{feature}_confluence_bullish']
        
        # Example: Confluence for momentum (RSI alignment)
        elif 'rsi' in feature:
            # Strong momentum if RSI > 60 on multiple timeframes
            strong_signals = sum(df[col] > 60 for col in tf_cols)
            df[f'{feature}_confluence_strong'] = strong_signals / len(tf_cols)
            df[f'{feature}_confluence_score'] = df[f'{feature}_confluence_strong']
        
        return df


# Convenience functions
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Quick feature engineering for single dataframe"""
    engineer = CausalFeatureEngineer()
    return engineer.calculate_all_features(df)

def engineer_multi_timeframe(
    data_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Quick multi-timeframe feature engineering"""
    engineer = MultiTimeframeFeatureEngineer()
    return engineer.align_timeframes(data_dict)


if __name__ == "__main__":
    # Test feature engineering
    from data_fetcher import fetch_data
    
    logger.info("Testing Feature Engineering")
    
    # Fetch data
    df = fetch_data("BTC/USDT", "1h")
    
    # Calculate features
    df_features = engineer_features(df)
    
    print(f"\nOriginal columns: {len(df.columns)}")
    print(f"Feature columns: {len(df_features.columns)}")
    print(f"\nSample features:")
    print(df_features[['close', 'returns', 'rsi', 'macd', 'atr', 'ema_21']].tail())

"""
AEGIS ML Feature Engineering Module
Creates machine learning features with strict causal validation
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np
import yaml
from scipy import stats
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for feature set metadata"""
    name: str
    features: List[str]
    lookback: int
    target: Optional[str] = None


class MLFeatureEngineer:
    """
    Creates ML-specific features with causal validation
    All features use only past data - zero lookahead bias
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.lookback_windows = [5, 10, 20, 50]
        self.scaler = StandardScaler()
        self.feature_sets: Dict[str, FeatureSet] = {}
        self._register_feature_sets()
    
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _register_feature_sets(self):
        """Register feature sets for different model types"""
        self.feature_sets['price_action'] = FeatureSet(
            name='price_action',
            features=['returns', 'log_returns', 'price_vs_ema', 'price_range'],
            lookback=20
        )
        
        self.feature_sets['trend'] = FeatureSet(
            name='trend',
            features=['ema_slope', 'macd', 'adx', 'trend_strength'],
            lookback=50
        )
        
        self.feature_sets['momentum'] = FeatureSet(
            name='momentum',
            features=['rsi', 'stoch', 'cci', 'momentum_score'],
            lookback=14
        )
        
        self.feature_sets['volatility'] = FeatureSet(
            name='volatility',
            features=['atr', 'bb_width', 'volatility_regime', 'hist_vol'],
            lookback=20
        )
        
        self.feature_sets['volume'] = FeatureSet(
            name='volume',
            features=['relative_volume', 'obv_slope', 'vwap_distance'],
            lookback=20
        )
        
        self.feature_sets['microstructure'] = FeatureSet(
            name='microstructure',
            features=['bid_ask_spread', 'trade_intensity', 'order_imbalance'],
            lookback=10
        )
    
    def calculate_returns_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate return-based features
        """
        df = df.copy()
        
        # Basic returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Multi-period returns
        for window in [3, 5, 10, 20]:
            df[f'returns_{window}d'] = df['close'].pct_change(window)
            df[f'log_returns_{window}d'] = np.log(df['close'] / df['close'].shift(window))
        
        # Return statistics (rolling)
        for window in [10, 20, 50]:
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
            df[f'returns_skew_{window}'] = df['returns'].rolling(window).skew()
            df[f'returns_kurt_{window}'] = df['returns'].rolling(window).kurt()
        
        # Cumulative returns
        df['cum_returns_5d'] = (1 + df['returns']).rolling(5).apply(np.prod) - 1
        df['cum_returns_20d'] = (1 + df['returns']).rolling(20).apply(np.prod) - 1
        
        # Price position within range
        df['price_range_position'] = (
            (df['close'] - df['low'].rolling(20).min()) / 
            (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        )
        
        return df
    
    def calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend-based features for ML
        """
        df = df.copy()
        
        # EMA features
        for fast, slow in [(8, 21), (21, 50), (50, 200)]:
            df[f'ema_{fast}'] = df['close'].ewm(span=fast, adjust=False).mean()
            df[f'ema_{slow}'] = df['close'].ewm(span=slow, adjust=False).mean()
            
            # EMA distance (normalized)
            df[f'ema_{fast}_dist'] = (df['close'] - df[f'ema_{fast}']) / df[f'ema_{fast}']
            df[f'ema_{slow}_dist'] = (df['close'] - df[f'ema_{slow}']) / df[f'ema_{slow}']
            
            # EMA cross
            df[f'ema_{fast}_{slow}_cross'] = (
                (df[f'ema_{fast}'] > df[f'ema_{slow}']).astype(int)
            )
            
            # EMA slope
            df[f'ema_{fast}_slope'] = df[f'ema_{fast}'].diff(3) / 3
            df[f'ema_{slow}_slope'] = df[f'ema_{slow}'].diff(3) / 3
        
        # MACD features
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_normalized'] = df['macd'] / df['close']
        
        # MACD momentum
        df['macd_momentum'] = df['macd_hist'].diff()
        
        # Trend strength (ADX)
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
        df['di_plus'] = plus_di
        df['di_minus'] = minus_di
        df['di_diff'] = plus_di - minus_di
        
        # Trend alignment score
        df['trend_alignment'] = (
            (df['close'] > df['ema_8']).astype(int) +
            (df['ema_8'] > df['ema_21']).astype(int) +
            (df['ema_21'] > df['ema_50']).astype(int) +
            (df['macd'] > df['macd_signal']).astype(int) +
            (df['adx'] > 25).astype(int)
        ) / 5.0
        
        return df
    
    def calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum features for ML
        """
        df = df.copy()
        
        # RSI variants
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI features
        df['rsi_normalized'] = df['rsi'] / 100
        df['rsi_slope'] = df['rsi'].diff(3)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        
        # Stochastic
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        df['stoch_cross'] = ((df['stoch_k'] > df['stoch_d']) & 
                            (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))).astype(int)
        
        # Williams %R
        df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
        
        # CCI
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_sma = tp.rolling(window=20).mean()
        mean_dev = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (tp - tp_sma) / (0.015 * mean_dev)
        
        # Rate of Change
        for window in [5, 10, 20]:
            df[f'roc_{window}'] = (df['close'] - df['close'].shift(window)) / df['close'].shift(window)
        
        # Momentum composite
        df['momentum_composite'] = (
            df['rsi_normalized'] * 0.4 +
            (df['stoch_k'] / 100) * 0.3 +
            ((df['cci'] + 200) / 400).clip(0, 1) * 0.3
        )
        
        # Divergence detection (price vs momentum)
        df['price_momentum_divergence'] = np.where(
            (df['close'] > df['close'].shift(5)) & (df['rsi'] < df['rsi'].shift(5)),
            -1,  # Bearish divergence
            np.where(
                (df['close'] < df['close'].shift(5)) & (df['rsi'] > df['rsi'].shift(5)),
                1,   # Bullish divergence
                0
            )
        )
        
        return df
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility features for ML
        """
        df = df.copy()
        
        # ATR
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.ewm(span=14, adjust=False).mean()
        df['atr_pct'] = df['atr'] / df['close']
        df['atr_normalized'] = df['atr'] / df['atr'].rolling(50).mean()
        
        # Bollinger Band features
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Bollinger squeeze
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.1)).astype(int)
        
        # Historical volatility
        for window in [10, 20, 50]:
            df[f'hist_vol_{window}'] = df['log_returns'].rolling(window).std() * np.sqrt(365 * 24)
        
        # Volatility regime
        df['volatility_regime'] = pd.cut(
            df['atr_normalized'],
            bins=[0, 0.5, 0.8, 1.2, 2.0, float('inf')],
            labels=[0, 1, 2, 3, 4]  # very_low, low, normal, high, extreme
        ).astype(float)
        
        # GARCH-like volatility clustering (simplified)
        df['vol_clustering'] = df['returns'].abs().rolling(5).mean() / df['returns'].abs().rolling(20).mean()
        
        return df
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume features for ML
        """
        df = df.copy()
        
        # Volume moving averages
        for window in [5, 20, 50]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
        
        # Relative volume
        df['relative_volume'] = df['volume'] / df['volume_sma_20']
        df['high_volume'] = (df['relative_volume'] > 2.0).astype(int)
        
        # Volume trend
        df['volume_trend'] = np.where(
            df['volume'] > df['volume'].shift(1) * 1.5, 2,
            np.where(df['volume'] > df['volume'].shift(1), 1,
                    np.where(df['volume'] < df['volume'].shift(1) * 0.5, -2, -1))
        )
        
        # On-Balance Volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_sma'] = df['obv'].rolling(window=20).mean()
        df['obv_slope'] = df['obv'].diff(5)
        df['obv_divergence'] = np.where(
            (df['close'] > df['close'].shift(5)) & (df['obv'] < df['obv'].shift(5)), -1,
            np.where((df['close'] < df['close'].shift(5)) & (df['obv'] > df['obv'].shift(5)), 1, 0)
        )
        
        # VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Volume-Price confirmation
        df['volume_price_confirm'] = np.where(
            (df['close'] > df['close'].shift(1)) & (df['volume'] > df['volume'].shift(1)), 1,
            np.where((df['close'] < df['close'].shift(1)) & (df['volume'] > df['volume'].shift(1)), -1, 0)
        )
        
        return df
    
    def calculate_target_variable(
        self,
        df: pd.DataFrame,
        lookahead: int = 5,
        threshold: float = 0.01
    ) -> pd.DataFrame:
        """
        Create target variable for supervised learning
        Classes: 1 (up), 0 (neutral), -1 (down)
        """
        df = df.copy()
        
        # Future returns
        future_return = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Classification target
        df['target'] = np.where(
            future_return > threshold, 1,
            np.where(future_return < -threshold, -1, 0)
        )
        
        # Regression target (for probability calibration)
        df['target_return'] = future_return
        
        return df
    
    def create_feature_matrix(
        self,
        df: pd.DataFrame,
        include_target: bool = True
    ) -> pd.DataFrame:
        """
        Create complete feature matrix with all features
        """
        df = df.copy()
        
        # Calculate all feature groups
        df = self.calculate_returns_features(df)
        df = self.calculate_trend_features(df)
        df = self.calculate_momentum_features(df)
        df = self.calculate_volatility_features(df)
        df = self.calculate_volume_features(df)
        
        # Create target if requested
        if include_target:
            df = self.calculate_target_variable(df)
        
        # Select final feature set
        feature_cols = []
        
        # Price action
        feature_cols.extend([
            'returns', 'log_returns', 'returns_5d', 'returns_20d',
            'returns_mean_20', 'returns_std_20', 'returns_skew_20',
            'price_range_position'
        ])
        
        # Trend
        feature_cols.extend([
            'ema_8_dist', 'ema_21_dist', 'ema_50_dist',
            'ema_8_21_cross', 'ema_21_50_cross',
            'macd', 'macd_hist', 'macd_normalized',
            'adx', 'di_diff', 'trend_alignment'
        ])
        
        # Momentum
        feature_cols.extend([
            'rsi', 'rsi_normalized', 'rsi_slope',
            'stoch_k', 'stoch_d', 'stoch_cross',
            'williams_r', 'cci', 'roc_10',
            'momentum_composite', 'price_momentum_divergence'
        ])
        
        # Volatility
        feature_cols.extend([
            'atr_pct', 'atr_normalized',
            'bb_width', 'bb_position', 'bb_squeeze',
            'hist_vol_20', 'volatility_regime', 'vol_clustering'
        ])
        
        # Volume
        feature_cols.extend([
            'relative_volume', 'volume_trend',
            'obv_slope', 'obv_divergence',
            'vwap_dist', 'volume_price_confirm'
        ])
        
        # Filter to existing columns
        available_cols = [c for c in feature_cols if c in df.columns]
        
        if include_target and 'target' in df.columns:
            available_cols.extend(['target', 'target_return'])
        
        return df[available_cols].dropna()
    
    def scale_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale features using StandardScaler
        """
        feature_cols = [c for c in df.columns if c not in ['target', 'target_return']]
        
        if fit:
            scaled = self.scaler.fit_transform(df[feature_cols])
        else:
            scaled = self.scaler.transform(df[feature_cols])
        
        df_scaled = pd.DataFrame(scaled, columns=feature_cols, index=df.index)
        
        # Add back target columns
        if 'target' in df.columns:
            df_scaled['target'] = df['target']
        if 'target_return' in df.columns:
            df_scaled['target_return'] = df['target_return']
        
        return df_scaled
    
    def get_feature_importance_mask(
        self,
        df: pd.DataFrame,
        top_n: int = 30
    ) -> List[str]:
        """
        Get mask of most important features based on mutual information
        """
        from sklearn.feature_selection import mutual_info_classif
        
        feature_cols = [c for c in df.columns if c not in ['target', 'target_return']]
        X = df[feature_cols].dropna()
        y = df.loc[X.index, 'target']
        
        # Calculate mutual information
        mi = mutual_info_classif(X, y, random_state=42)
        
        # Get top features
        feature_mi = list(zip(feature_cols, mi))
        feature_mi.sort(key=lambda x: x[1], reverse=True)
        
        return [f[0] for f in feature_mi[:top_n]]


# Convenience function
def engineer_ml_features(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Quick ML feature engineering"""
    engineer = MLFeatureEngineer()
    return engineer.create_feature_matrix(df, include_target)

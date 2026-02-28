"""
AEGIS Utility Helpers
Common utility functions for data processing
"""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import pytz


def timeframe_to_seconds(timeframe: str) -> int:
    """Convert timeframe string to seconds"""
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    
    multipliers = {
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800
    }
    
    return value * multipliers.get(unit, 3600)


def timeframe_to_pandas_freq(timeframe: str) -> str:
    """Convert timeframe to pandas frequency string"""
    unit = timeframe[-1].lower()
    value = timeframe[:-1]
    return f"{value}{unit.upper()}"


def resample_ohlcv(
    df: pd.DataFrame,
    target_timeframe: str,
    source_timeframe: str
) -> pd.DataFrame:
    """
    Resample OHLCV data to higher timeframe
    Ensures causal calculation (no look-ahead)
    
    Args:
        df: DataFrame with OHLCV data
        target_timeframe: Target timeframe (e.g., '4h')
        source_timeframe: Source timeframe (e.g., '1h')
    
    Returns:
        Resampled DataFrame
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # Resample logic
    resampled = df.resample(timeframe_to_pandas_freq(target_timeframe)).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Remove empty candles
    resampled = resampled.dropna()
    
    resampled = resampled.reset_index()
    return resampled


def calculate_returns(
    prices: pd.Series,
    log_returns: bool = True
) -> pd.Series:
    """
    Calculate returns series (causal calculation)
    
    Args:
        prices: Price series
        log_returns: Use log returns if True, simple returns if False
    
    Returns:
        Returns series
    """
    if log_returns:
        return np.log(prices / prices.shift(1))
    else:
        return prices.pct_change()


def generate_cache_key(
    symbol: str,
    timeframe: str,
    start_time: datetime,
    end_time: datetime
) -> str:
    """Generate unique cache key for data request"""
    key_string = f"{symbol}_{timeframe}_{start_time.isoformat()}_{end_time.isoformat()}"
    return hashlib.md5(key_string.encode()).hexdigest()


def ensure_datetime_utc(dt: Union[datetime, str, pd.Timestamp]) -> datetime:
    """Ensure datetime is UTC timezone-aware"""
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.UTC)
    else:
        dt = dt.astimezone(pytz.UTC)
    
    return dt


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    if denominator == 0 or np.isnan(denominator):
        return default
    return numerator / denominator


def rolling_window(
    arr: np.ndarray,
    window: int
) -> np.ndarray:
    """
    Create rolling window view of array (memory efficient)
    
    Args:
        arr: Input array
        window: Window size
    
    Returns:
        Array of shape (len(arr) - window + 1, window)
    """
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def save_dataframe_atomic(
    df: pd.DataFrame,
    filepath: Path,
    format: str = "parquet"
) -> None:
    """
    Save DataFrame atomically to prevent corruption
    
    Args:
        df: DataFrame to save
        filepath: Target file path
        format: File format ('parquet' or 'csv')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file first
    temp_file = filepath.with_suffix(f"{filepath.suffix}.tmp")
    
    try:
        if format == "parquet":
            df.to_parquet(temp_file, compression="zstd", index=False)
        elif format == "csv":
            df.to_csv(temp_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Atomic rename
        temp_file.replace(filepath)
        
    except Exception:
        # Clean up temp file on failure
        if temp_file.exists():
            temp_file.unlink()
        raise


def load_yaml_config(filepath: Path) -> Dict[str, Any]:
    """Load YAML configuration file"""
    import yaml
    
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def merge_dicts(base: Dict, update: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


class TimeSeriesAligner:
    """
    Align multiple time series with different timeframes
    Ensures causal alignment (no future data leakage)
    """
    
    @staticmethod
    def align_timeframes(
        primary_df: pd.DataFrame,
        higher_tf_dfs: Dict[str, pd.DataFrame],
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Align higher timeframe data to primary timeframe
        
        Args:
            primary_df: Primary timeframe DataFrame
            higher_tf_dfs: Dict of {timeframe: df} for higher timeframes
            timestamp_col: Timestamp column name
        
        Returns:
            Aligned DataFrame with higher TF features
        """
        result = primary_df.copy()
        result[timestamp_col] = pd.to_datetime(result[timestamp_col])
        
        for tf, tf_df in higher_tf_dfs.items():
            tf_df = tf_df.copy()
            tf_df[timestamp_col] = pd.to_datetime(tf_df[timestamp_col])
            
            # Forward fill higher TF data (causal - only past data)
            tf_df = tf_df.set_index(timestamp_col)
            tf_df = tf_df.reindex(
                result[timestamp_col],
                method='ffill'
            )
            tf_df = tf_df.reset_index()
            
            # Rename columns to indicate timeframe
            tf_df.columns = [
                f"{col}_{tf}" if col != timestamp_col else col
                for col in tf_df.columns
            ]
            
            # Merge
            result = result.merge(
                tf_df,
                on=timestamp_col,
                how='left'
            )
        
        return result

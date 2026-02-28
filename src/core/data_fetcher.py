"""
AEGIS Data Fetcher Module
Institutional-grade data acquisition with robust error handling
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import ccxt
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataValidationResult:
    """Data validation result container"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    missing_pct: float
    gap_count: int


class ExchangeManager:
    """
    Manages exchange connections with failover and rate limiting
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.rate_limiters: Dict[str, float] = {}
        self.last_request_time: Dict[str, float] = {}
        
        self._initialize_exchanges()
    
    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        exchange_ids = [
            self.config['api']['exchanges']['primary'],
            self.config['api']['exchanges']['backup']
        ]
        
        for exchange_id in exchange_ids:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                    }
                })
                
                # Test connection
                exchange.load_markets()
                
                self.exchanges[exchange_id] = exchange
                self.rate_limiters[exchange_id] = 1.0 / self.config['api']['rate_limits'][exchange_id]
                self.last_request_time[exchange_id] = 0
                
                logger.info(f"Initialized exchange: {exchange_id}")
                
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_id}: {e}")
                continue
        
        if not self.exchanges:
            raise ConnectionError("No exchanges could be initialized")
    
    def _rate_limit(self, exchange_id: str):
        """Apply rate limiting"""
        min_interval = self.rate_limiters[exchange_id]
        elapsed = time.time() - self.last_request_time[exchange_id]
        
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time[exchange_id] = time.time()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError))
    )
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        exchange_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with automatic failover
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '4h')
            since: Timestamp in milliseconds
            limit: Number of candles
            exchange_id: Specific exchange to use
        
        Returns:
            DataFrame with OHLCV data
        """
        exchanges_to_try = [exchange_id] if exchange_id else list(self.exchanges.keys())
        
        for ex_id in exchanges_to_try:
            if ex_id not in self.exchanges:
                continue
            
            try:
                self._rate_limit(ex_id)
                exchange = self.exchanges[ex_id]
                
                logger.info(f"Fetching {symbol} {timeframe} from {ex_id}")
                
                ohlcv = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )
                
                if not ohlcv:
                    logger.warning(f"No data returned for {symbol} from {ex_id}")
                    continue
                
                df = self._ohlcv_to_dataframe(ohlcv, symbol, timeframe)
                df['exchange'] = ex_id
                
                logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
                return df
                
            except ccxt.BadSymbol:
                logger.error(f"Symbol {symbol} not available on {ex_id}")
                continue
            except Exception as e:
                logger.error(f"Error fetching from {ex_id}: {e}")
                continue
        
        raise Exception(f"Failed to fetch data for {symbol} from all exchanges")
    
    def _ohlcv_to_dataframe(
        self,
        ohlcv: List[List[Union[int, float]]],
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
        """Convert CCXT OHLCV to DataFrame"""
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        
        # Ensure numeric types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        return df
    
    def get_exchange_status(self) -> Dict[str, bool]:
        """Get status of all exchanges"""
        return {ex_id: True for ex_id in self.exchanges.keys()}


class DataValidator:
    """
    Validates OHLCV data quality
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.validation_params = self.config['data']['validation']
    
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        """
        Comprehensive data validation
        
        Checks:
        1. Missing values
        2. Price gaps
        3. Zero volume candles
        4. OHLC logic (high >= low, etc.)
        5. Timestamp continuity
        """
        errors = []
        warnings = []
        
        # Check 1: Missing values
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        if missing_pct > self.validation_params['max_missing_pct']:
            errors.append(f"Missing data: {missing_pct:.2f}% exceeds {self.validation_params['max_missing_pct']}%")
        
        # Check 2: Price gaps
        gap_count = self._detect_price_gaps(df)
        if gap_count > 0:
            warnings.append(f"Detected {gap_count} price gaps > {self.validation_params['max_price_gap_pct']}%")
        
        # Check 3: Zero volume
        zero_volume = (df['volume'] == 0).sum()
        if zero_volume > len(df) * 0.1:  # More than 10% zero volume
            warnings.append(f"{zero_volume} candles with zero volume")
        
        # Check 4: OHLC logic
        ohlc_errors = self._check_ohlc_logic(df)
        if ohlc_errors > 0:
            errors.append(f"{ohlc_errors} OHLC logic violations")
        
        # Check 5: Timestamp continuity
        continuity_errors = self._check_timestamp_continuity(df)
        if continuity_errors > 0:
            warnings.append(f"{continuity_errors} timestamp discontinuities")
        
        is_valid = len(errors) == 0
        
        return DataValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            missing_pct=missing_pct,
            gap_count=gap_count
        )
    
    def _detect_price_gaps(self, df: pd.DataFrame) -> int:
        """Detect significant price gaps between candles"""
        if len(df) < 2:
            return 0
        
        prev_close = df['close'].shift(1)
        curr_open = df['open']
        
        gap_pct = abs((curr_open - prev_close) / prev_close * 100)
        gaps = (gap_pct > self.validation_params['max_price_gap_pct']).sum()
        
        return int(gaps)
    
    def _check_ohlc_logic(self, df: pd.DataFrame) -> int:
        """Verify OHLC relationships"""
        errors = 0
        errors += (df['high'] < df['low']).sum()
        errors += (df['high'] < df['open']).sum()
        errors += (df['high'] < df['close']).sum()
        errors += (df['low'] > df['open']).sum()
        errors += (df['low'] > df['close']).sum()
        return int(errors)
    
    def _check_timestamp_continuity(self, df: pd.DataFrame) -> int:
        """Check for missing timestamps"""
        if len(df) < 2:
            return 0
        
        # Infer frequency
        freq = pd.infer_freq(df.index)
        if freq is None:
            return 0
        
        expected = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
        missing = len(expected) - len(df)
        
        return max(0, missing)


class DataStore:
    """
    Manages data persistence with Parquet format
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.raw_path = Path(self.config['storage']['raw_data_path'])
        self.processed_path = Path(self.config['storage']['processed_data_path'])
        self.cache_path = Path(self.config['storage']['cache_path'])
        
        self._ensure_directories()
    
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _ensure_directories(self):
        """Create necessary directories"""
        for path in [self.raw_path, self.processed_path, self.cache_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def save_raw_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        exchange: str
    ) -> str:
        """
        Save raw OHLCV data to Parquet
        
        Returns:
            Path to saved file
        """
        # Create filename: BTC_USDT_1h_binance_20240101_20240131.parquet
        start_date = df.index[0].strftime('%Y%m%d')
        end_date = df.index[-1].strftime('%Y%m%d')
        symbol_clean = symbol.replace('/', '_')
        
        filename = f"{symbol_clean}_{timeframe}_{exchange}_{start_date}_{end_date}.parquet"
        filepath = self.raw_path / filename
        
        # Save with compression
        df.to_parquet(filepath, compression='zstd', index=True)
        logger.info(f"Saved raw data: {filepath}")
        
        return str(filepath)
    
    def load_raw_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load raw data for symbol/timeframe
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            DataFrame or None if not found
        """
        symbol_clean = symbol.replace('/', '_')
        pattern = f"{symbol_clean}_{timeframe}_*.parquet"
        
        files = list(self.raw_path.glob(pattern))
        if not files:
            return None
        
        # Load and concatenate all matching files
        dfs = []
        for file in sorted(files):
            df = pd.read_parquet(file)
            dfs.append(df)
        
        if not dfs:
            return None
        
        combined = pd.concat(dfs)
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.sort_index(inplace=True)
        
        # Apply date filters
        if start_date:
            combined = combined[combined.index >= start_date]
        if end_date:
            combined = combined[combined.index <= end_date]
        
        return combined
    
    def get_latest_timestamp(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[datetime]:
        """Get the latest available timestamp for symbol/timeframe"""
        df = self.load_raw_data(symbol, timeframe)
        if df is not None and not df.empty:
            return df.index[-1]
        return None
    
    def clean_old_data(self, max_age_days: int = 30):
        """Remove raw data files older than max_age_days"""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        
        for file in self.raw_path.glob("*.parquet"):
            # Extract dates from filename
            try:
                # Parse filename to get end date
                parts = file.stem.split('_')
                if len(parts) >= 5:
                    end_date_str = parts[-1]
                    end_date = datetime.strptime(end_date_str, '%Y%m%d')
                    
                    if end_date < cutoff:
                        file.unlink()
                        logger.info(f"Removed old data: {file}")
            except Exception as e:
                logger.warning(f"Could not parse filename {file}: {e}")


class DataPipeline:
    """
    Main data pipeline orchestrator
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.exchange_manager = ExchangeManager(config_path)
        self.validator = DataValidator(config_path)
        self.store = DataStore(config_path)
        
        self.timeframes = self.config['data']['timeframes']
        self.lookback = self.config['data']['lookback']
    
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def fetch_complete_data(
        self,
        symbol: str,
        timeframe: str,
        update_only: bool = True
    ) -> pd.DataFrame:
        """
        Fetch complete or incremental data for symbol/timeframe
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            update_only: If True, only fetch new data since last save
        
        Returns:
            DataFrame with complete data
        """
        if update_only:
            latest = self.store.get_latest_timestamp(symbol, timeframe)
            if latest:
                # Fetch from latest candle
                since = int(latest.timestamp() * 1000)
                logger.info(f"Updating {symbol} {timeframe} from {latest}")
            else:
                since = None
        else:
            since = None
        
        # Calculate limit if starting fresh
        limit = None if since else self.lookback.get(timeframe, 500)
        
        # Fetch from exchange
        df_new = self.exchange_manager.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=limit
        )
        
        # Validate new data
        validation = self.validator.validate(df_new)
        if not validation.is_valid:
            logger.error(f"Validation failed for {symbol} {timeframe}: {validation.errors}")
            raise ValueError(f"Data validation failed: {validation.errors}")
        
        if validation.warnings:
            logger.warning(f"Validation warnings: {validation.warnings}")
        
        # Merge with existing data if updating
        if update_only and since:
            df_existing = self.store.load_raw_data(symbol, timeframe)
            if df_existing is not None:
                df_combined = pd.concat([df_existing, df_new])
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                df_combined.sort_index(inplace=True)
                df_new = df_combined
        
        # Save to storage
        exchange = df_new['exchange'].iloc[0]
        self.store.save_raw_data(df_new, symbol, timeframe, exchange)
        
        return df_new
    
    def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes
        
        Args:
            symbol: Trading pair
            timeframes: List of timeframes, defaults to config
        
        Returns:
            Dictionary of {timeframe: DataFrame}
        """
        if timeframes is None:
            timeframes = list(self.timeframes.values())
        
        data = {}
        for tf in timeframes:
            try:
                df = self.fetch_complete_data(symbol, tf, update_only=True)
                data[tf] = df
                logger.info(f"Fetched {tf}: {len(df)} candles")
            except Exception as e:
                logger.error(f"Failed to fetch {tf}: {e}")
                continue
        
        return data
    
    def fetch_all_assets(
        self,
        assets_config_path: str = "config/assets.yaml"
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch data for all configured assets
        
        Returns:
            Dictionary of {symbol: {timeframe: DataFrame}}
        """
        with open(assets_config_path, 'r') as f:
            assets_config = yaml.safe_load(f)
        
        all_data = {}
        selected_tiers = assets_config['selection']['default_tiers']
        max_assets = assets_config['selection']['max_assets']
        
        assets_to_fetch = []
        for tier in selected_tiers:
            if tier in assets_config['assets']:
                assets_to_fetch.extend(assets_config['assets'][tier])
        
        assets_to_fetch = assets_to_fetch[:max_assets]
        
        for asset in assets_to_fetch:
            symbol = asset['symbol']
            try:
                logger.info(f"Fetching data for {symbol}")
                data = self.fetch_multi_timeframe(symbol)
                all_data[symbol] = data
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue
        
        return all_data
    
    def get_data_summary(self) -> pd.DataFrame:
        """Generate summary of stored data"""
        records = []
        
        for file in self.store.raw_path.glob("*.parquet"):
            try:
                df = pd.read_parquet(file)
                symbol = df['symbol'].iloc[0]
                timeframe = df['timeframe'].iloc[0]
                exchange = df['exchange'].iloc[0]
                
                records.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'exchange': exchange,
                    'start_date': df.index[0],
                    'end_date': df.index[-1],
                    'candles': len(df),
                    'file_size_mb': file.stat().st_size / (1024 * 1024)
                })
            except Exception as e:
                logger.warning(f"Could not read {file}: {e}")
        
        return pd.DataFrame(records)


# Convenience functions for quick access
def fetch_data(symbol: str, timeframe: str = "1h") -> pd.DataFrame:
    """Quick fetch for single symbol/timeframe"""
    pipeline = DataPipeline()
    return pipeline.fetch_complete_data(symbol, timeframe, update_only=True)

def fetch_multi_tf(symbol: str) -> Dict[str, pd.DataFrame]:
    """Quick fetch for all timeframes"""
    pipeline = DataPipeline()
    return pipeline.fetch_multi_timeframe(symbol)

def update_all_data() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Update all configured assets"""
    pipeline = DataPipeline()
    return pipeline.fetch_all_assets()


if __name__ == "__main__":
    # Test the pipeline
    logger.info("Testing AEGIS Data Pipeline")
    
    # Test single fetch
    df = fetch_data("BTC/USDT", "1h")
    print(f"\nFetched BTC/USDT 1h: {len(df)} candles")
    print(df.tail())
    
    # Test multi-timeframe
    data = fetch_multi_tf("BTC/USDT")
    print(f"\nFetched {len(data)} timeframes")
    for tf, df in data.items():
        print(f"  {tf}: {len(df)} candles")

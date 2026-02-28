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
    Includes proxy support for restricted locations
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.rate_limiters: Dict[str, float] = {}
        self.last_request_time: Dict[str, float] = {}
        
        # Proxy configuration from environment
        self.proxies = self._get_proxies()
        
        self._initialize_exchanges()
    
    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML"""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            # Return default config
            return {
                'api': {
                    'exchanges': {
                        'primary': 'binance',
                        'backup': 'kraken'
                    },
                    'rate_limits': {
                        'binance': 10,
                        'kraken': 3
                    },
                    'retry': {
                        'max_attempts': 3,
                        'backoff_factor': 2,
                        'timeout_seconds': 30
                    }
                }
            }
    
    def _get_proxies(self) -> Dict[str, str]:
        """Get proxy configuration from environment variables"""
        proxies = {}
        
        # Check for proxy settings
        http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        
        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https'] = https_proxy
        
        if proxies:
            logger.info(f"Using proxies: {proxies}")
        
        return proxies
    
    def _initialize_exchanges(self):
        """Initialize exchange connections with fallback"""
        # List of exchanges to try (in order)
        exchange_list = [
            self.config.get('api', {}).get('exchanges', {}).get('primary', 'binance'),
            self.config.get('api', {}).get('exchanges', {}).get('backup', 'kraken'),
            'kraken',      # US-friendly
            'coinbase',    # US-friendly
            'okx',         # Global
            'gateio',      # Global
            'kucoin'       # Global
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        exchange_list = [x for x in exchange_list if not (x in seen or seen.add(x))]
        
        for exchange_id in exchange_list:
            if exchange_id in self.exchanges:
                continue
            
            try:
                exchange_class = getattr(ccxt, exchange_id)
                
                # Exchange-specific options
                options = {
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                    }
                }
                
                # Add proxy if configured
                if self.proxies:
                    options['proxies'] = self.proxies
                
                # Special handling for specific exchanges
                if exchange_id == 'binance':
                    # Try both .com and .us
                    options['options']['adjustForTimeDifference'] = True
                
                exchange = exchange_class(options)
                
                # Test connection with timeout
                import socket
                original_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(10)
                
                try:
                    exchange.load_markets()
                    
                    self.exchanges[exchange_id] = exchange
                    
                    # Set rate limit
                    rate_limits = self.config.get('api', {}).get('rate_limits', {})
                    default_rate = 5  # Safe default
                    self.rate_limiters[exchange_id] = 1.0 / rate_limits.get(exchange_id, default_rate)
                    self.last_request_time[exchange_id] = 0
                    
                    logger.info(f"✅ Initialized exchange: {exchange_id}")
                    
                    # If we have at least 2 exchanges, we're good
                    if len(self.exchanges) >= 2:
                        break
                        
                except ccxt.NetworkError as e:
                    logger.warning(f"Network error with {exchange_id}: {e}")
                    continue
                except ccxt.ExchangeError as e:
                    if "restricted location" in str(e).lower() or "403" in str(e):
                        logger.warning(f"🚫 {exchange_id} blocked in this region")
                    else:
                        logger.warning(f"Exchange error with {exchange_id}: {e}")
                    continue
                finally:
                    socket.setdefaulttimeout(original_timeout)
                
            except Exception as e:
                logger.warning(f"Failed to initialize {exchange_id}: {e}")
                continue
        
        if not self.exchanges:
            logger.error("⚠️ No exchanges could be initialized - will use cached data only")
            # Don't raise error - allow system to work with cached data
    
    def _rate_limit(self, exchange_id: str):
        """Apply rate limiting"""
        if exchange_id not in self.rate_limiters:
            return
        
        min_interval = self.rate_limiters[exchange_id]
        elapsed = time.time() - self.last_request_time.get(exchange_id, 0)
        
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
        # If specific exchange requested, try it first
        if exchange_id and exchange_id in self.exchanges:
            exchanges_to_try = [exchange_id]
        else:
            exchanges_to_try = list(self.exchanges.keys())
        
        last_error = None
        
        for ex_id in exchanges_to_try:
            try:
                self._rate_limit(ex_id)
                exchange = self.exchanges[ex_id]
                
                logger.info(f"Fetching {symbol} {timeframe} from {ex_id}")
                
                # Handle symbol format differences
                normalized_symbol = self._normalize_symbol(symbol, ex_id)
                
                ohlcv = exchange.fetch_ohlcv(
                    symbol=normalized_symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )
                
                if not ohlcv:
                    logger.warning(f"No data returned for {symbol} from {ex_id}")
                    continue
                
                df = self._ohlcv_to_dataframe(ohlcv, symbol, timeframe)
                df['exchange'] = ex_id
                
                logger.info(f"✅ Fetched {len(df)} candles for {symbol} {timeframe}")
                return df
                
            except ccxt.BadSymbol:
                logger.warning(f"Symbol {symbol} not available on {ex_id}")
                continue
            except ccxt.NetworkError as e:
                logger.warning(f"Network error with {ex_id}: {e}")
                last_error = e
                continue
            except ccxt.ExchangeError as e:
                if "restricted" in str(e).lower() or "403" in str(e):
                    logger.warning(f"🚫 {ex_id} blocked for {symbol}")
                else:
                    logger.warning(f"Exchange error with {ex_id}: {e}")
                last_error = e
                continue
            except Exception as e:
                logger.error(f"Unexpected error with {ex_id}: {e}")
                last_error = e
                continue
        
        # If we get here, all exchanges failed
        raise Exception(f"Failed to fetch data for {symbol} from all exchanges: {last_error}")
    
    def _normalize_symbol(self, symbol: str, exchange_id: str) -> str:
        """Normalize symbol format for specific exchange"""
        # Most exchanges use BTC/USDT format
        # Some might use BTC-USDT or BTCUSDT
        return symbol  # Default, can be extended per exchange
    
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
    
    def has_working_exchange(self) -> bool:
        """Check if at least one exchange is working"""
        return len(self.exchanges) > 0


class DataValidator:
    """
    Validates OHLCV data quality
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.validation_params = self.config.get('data', {}).get('validation', {
            'max_missing_pct': 2.0,
            'min_volume_threshold': 1000,
            'max_price_gap_pct': 5.0
        })
    
    def _load_config(self, path: str) -> dict:
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {}
    
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
        if missing_pct > self.validation_params.get('max_missing_pct', 2.0):
            errors.append(f"Missing data: {missing_pct:.2f}% exceeds {self.validation_params.get('max_missing_pct', 2.0)}%")
        
        # Check 2: Price gaps
        gap_count = self._detect_price_gaps(df)
        if gap_count > 0:
            warnings.append(f"Detected {gap_count} price gaps > {self.validation_params.get('max_price_gap_pct', 5.0)}%")
        
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
        gaps = (gap_pct > self.validation_params.get('max_price_gap_pct', 5.0)).sum()
        
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
        self.raw_path = Path(self.config.get('storage', {}).get('raw_data_path', 'data/raw'))
        self.processed_path = Path(self.config.get('storage', {}).get('processed_data_path', 'data/processed'))
        self.cache_path = Path(self.config.get('storage', {}).get('cache_path', 'data/cache'))
        
        self._ensure_directories()
    
    def _load_config(self, path: str) -> dict:
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {}
    
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
        logger.info(f"💾 Saved raw data: {filepath}")
        
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
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Could not read {file}: {e}")
                continue
        
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
                        logger.info(f"🗑️ Removed old data: {file}")
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
        
        self.timeframes = self.config.get('data', {}).get('timeframes', {
            'primary': '1h',
            'secondary': '4h',
            'tertiary': '1d',
            'quaternary': '1w'
        })
        self.lookback = self.config.get('data', {}).get('lookback', {
            '1h': 500,
            '4h': 300,
            '1d': 200,
            '1w': 100
        })
    
    def _load_config(self, path: str) -> dict:
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def fetch_complete_data(
        self,
        symbol: str,
        timeframe: str,
        update_only: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch complete or incremental data for symbol/timeframe
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            update_only: If True, only fetch new data since last save
        
        Returns:
            DataFrame with complete data or None if failed
        """
        # Check if we have any working exchanges
        if not self.exchange_manager.has_working_exchange():
            logger.warning("⚠️ No working exchanges - attempting to use cached data")
            # Try to load from cache only
            cached = self.store.load_raw_data(symbol, timeframe)
            if cached is not None:
                logger.info(f"📂 Using cached data for {symbol} {timeframe}: {len(cached)} candles")
                return cached
            else:
                logger.error(f"❌ No cached data available for {symbol} {timeframe}")
                return None
        
        if update_only:
            latest = self.store.get_latest_timestamp(symbol, timeframe)
            if latest:
                # Fetch from latest candle
                since = int(latest.timestamp() * 1000)
                logger.info(f"🔄 Updating {symbol} {timeframe} from {latest}")
            else:
                since = None
        else:
            since = None
        
        # Calculate limit if starting fresh
        limit = None if since else self.lookback.get(timeframe, 500)
        
        try:
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
                logger.error(f"❌ Validation failed for {symbol} {timeframe}: {validation.errors}")
                # Still save if partial data is useful
                if validation.missing_pct < 10:  # Allow some missing data
                    logger.warning("Using partially valid data")
                else:
                    raise ValueError(f"Data validation failed: {validation.errors}")
            
            if validation.warnings:
                logger.warning(f"⚠️ Validation warnings: {validation.warnings}")
            
            # Merge with existing data if updating
            if update_only and since:
                df_existing = self.store.load_raw_data(symbol, timeframe)
                if df_existing is not None:
                    df_combined = pd.concat([df_existing, df_new])
                    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                    df_combined.sort_index(inplace=True)
                    df_new = df_combined
            
            # Save to storage
            exchange = df_new['exchange'].iloc[0] if 'exchange' in df_new.columns else 'unknown'
            self.store.save_raw_data(df_new, symbol, timeframe, exchange)
            
            return df_new
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch from exchanges: {e}")
            # Fallback to cached data
            cached = self.store.load_raw_data(symbol, timeframe)
            if cached is not None:
                logger.info(f"📂 Fallback to cached data: {len(cached)} candles")
                return cached
            return None
    
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
                if df is not None and not df.empty:
                    data[tf] = df
                    logger.info(f"✅ Fetched {tf}: {len(df)} candles")
                else:
                    logger.warning(f"⚠️ No data for {tf}")
            except Exception as e:
                logger.error(f"❌ Failed to fetch {tf}: {e}")
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
        try:
            with open(assets_config_path, 'r') as f:
                assets_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load assets config: {e}")
            return {}
        
        all_data = {}
        selected_tiers = assets_config.get('selection', {}).get('default_tiers', ['tier_1', 'tier_2'])
        max_assets = assets_config.get('selection', {}).get('max_assets', 10)
        
        assets_to_fetch = []
        for tier in selected_tiers:
            if tier in assets_config.get('assets', {}):
                assets_to_fetch.extend(assets_config['assets'][tier])
        
        assets_to_fetch = assets_to_fetch[:max_assets]
        
        for asset in assets_to_fetch:
            symbol = asset['symbol']
            try:
                logger.info(f"📊 Fetching data for {symbol}")
                data = self.fetch_multi_timeframe(symbol)
                if data:
                    all_data[symbol] = data
            except Exception as e:
                logger.error(f"❌ Failed to fetch {symbol}: {e}")
                continue
        
        return all_data
    
    def get_data_summary(self) -> pd.DataFrame:
        """Generate summary of stored data"""
        records = []
        
        for file in self.store.raw_path.glob("*.parquet"):
            try:
                df = pd.read_parquet(file)
                symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'unknown'
                timeframe = df['timeframe'].iloc[0] if 'timeframe' in df.columns else 'unknown'
                exchange = df['exchange'].iloc[0] if 'exchange' in df.columns else 'unknown'
                
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
def fetch_data(symbol: str, timeframe: str = "1h") -> Optional[pd.DataFrame]:
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
    logger.info("🧪 Testing AEGIS Data Pipeline")
    
    # Test single fetch
    try:
        df = fetch_data("BTC/USDT", "1h")
        if df is not None:
            print(f"\n✅ Fetched BTC/USDT 1h: {len(df)} candles")
            print(df.tail())
        else:
            print("\n⚠️ No data fetched - using cache or check exchanges")
    except Exception as e:
        print(f"\n❌ Error: {e}")

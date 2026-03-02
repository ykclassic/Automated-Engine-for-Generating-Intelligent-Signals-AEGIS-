"""
AEGIS Data Fetcher Module
Smart exchange selection based on geographic availability
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import socket

import pandas as pd
import numpy as np
import ccxt
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExchangeManager:
    GLOBAL_EXCHANGES = ['kraken', 'coinbase', 'okx', 'gateio', 'kucoin', 'mexc']
    RESTRICTED_EXCHANGES = ['binance', 'bybit', 'bitfinex', 'huobi']
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.rate_limiters: Dict[str, float] = {}
        self.last_request_time: Dict[str, float] = {}
        self.proxies = self._get_proxies()
        self._initialize_exchanges_smart()
    
    def _load_config(self, path: str) -> dict:
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _get_proxies(self) -> Dict[str, str]:
        proxies = {}
        http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https'] = https_proxy
        if proxies:
            logger.info(f"Using proxies: {proxies}")
        return proxies
    
    def _test_exchange(self, exchange_id: str, timeout: int = 10) -> bool:
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({'enableRateLimit': True, 'timeout': timeout * 1000})
            if self.proxies:
                exchange.proxies = self.proxies
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(timeout)
            try:
                exchange.load_markets()
                socket.setdefaulttimeout(original_timeout)
                logger.info(f"✅ {exchange_id} is accessible")
                return True
            except ccxt.NetworkError as e:
                if "restricted" in str(e).lower() or "403" in str(e) or "451" in str(e):
                    logger.warning(f"🚫 {exchange_id} blocked in this region")
                else:
                    logger.warning(f"⚠️ {exchange_id} network error: {e}")
                return False
            except Exception as e:
                logger.warning(f"❌ {exchange_id} error: {e}")
                return False
            finally:
                socket.setdefaulttimeout(original_timeout)
        except Exception as e:
            logger.warning(f"Failed to test {exchange_id}: {e}")
            return False
    
    def _initialize_exchanges_smart(self):
        logger.info("🔍 Testing exchange accessibility...")
        working_exchanges = []
        for exchange_id in self.GLOBAL_EXCHANGES:
            if self._test_exchange(exchange_id):
                working_exchanges.append(exchange_id)
                if len(working_exchanges) >= 3:
                    break
        if len(working_exchanges) < 2:
            logger.info("⚠️ Limited global exchanges working, trying alternatives...")
            for exchange_id in self.RESTRICTED_EXCHANGES:
                if exchange_id not in working_exchanges and self._test_exchange(exchange_id):
                    working_exchanges.append(exchange_id)
                    if len(working_exchanges) >= 3:
                        break
        for exchange_id in working_exchanges:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                config = {'enableRateLimit': True, 'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}}
                exchange_config = self.config.get('exchanges', {}).get(exchange_id, {})
                if exchange_config.get('api_key'):
                    config['apiKey'] = exchange_config['api_key']
                if exchange_config.get('api_secret'):
                    config['secret'] = exchange_config['api_secret']
                if self.proxies:
                    config['proxies'] = self.proxies
                exchange = exchange_class(config)
                self.exchanges[exchange_id] = exchange
                self.rate_limiters[exchange_id] = exchange_config.get('rate_limit', 1.0)
                self.last_request_time[exchange_id] = 0
                logger.info(f"✅ Initialized {exchange_id}")
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_id}: {e}")
        if not self.exchanges:
            raise RuntimeError("❌ No exchanges available! Check network/proxy settings.")
        logger.info(f"🚀 Active exchanges: {list(self.exchanges.keys())}")
    
    def _rate_limit(self, exchange_id: str):
        min_interval = self.rate_limiters.get(exchange_id, 1.0)
        last_time = self.last_request_time.get(exchange_id, 0)
        elapsed = time.time() - last_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time[exchange_id] = time.time()
    
    def get_exchange(self, preferred: Optional[str] = None) -> ccxt.Exchange:
        if preferred and preferred in self.exchanges:
            return self.exchanges[preferred]
        return next(iter(self.exchanges.values()))
    
    def get_all_exchanges(self) -> List[ccxt.Exchange]:
        return list(self.exchanges.values())


@dataclass
class DataFetchResult:
    success: bool
    data: Optional[pd.DataFrame]
    exchange: str
    symbol: str
    timeframe: str
    error: Optional[str] = None
    latency_ms: float = 0.0
    rows_fetched: int = 0


class DataFetcher:
    def __init__(self, exchange_manager: ExchangeManager, cache_dir: str = "data/cache"):
        self.exchange_manager = exchange_manager
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_hours = 1
        logger.info(f"📁 Cache directory: {self.cache_dir.absolute()}")
    
    def _get_cache_path(self, exchange: str, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace('/', '_').replace(':', '_')
        return self.cache_dir / f"{exchange}_{safe_symbol}_{timeframe}.parquet"
    
    def _load_from_cache(self, cache_path: Path, max_age_hours: float) -> Optional[pd.DataFrame]:
        if not cache_path.exists():
            return None
        file_age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if file_age_hours > max_age_hours:
            return None
        try:
            df = pd.read_parquet(cache_path)
            logger.debug(f"📂 Cache hit: {cache_path.name}")
            return df
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path):
        try:
            df.to_parquet(cache_path, compression='zstd')
            logger.debug(f"💾 Cached: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError)), reraise=True)
    def _fetch_from_exchange(self, exchange: ccxt.Exchange, symbol: str, timeframe: str,
                             since: Optional[int] = None, limit: int = 500) -> pd.DataFrame:
        exchange_id = exchange.id
        self.exchange_manager._rate_limit(exchange_id)
        start_time = time.time()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        latency = (time.time() - start_time) * 1000
        if not ohlcv:
            raise ValueError(f"No data returned for {symbol} on {exchange_id}")
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df['exchange'] = exchange_id
        df['symbol'] = symbol
        logger.info(f"📊 Fetched {len(df)} rows from {exchange_id} for {symbol} ({timeframe}) - {latency:.0f}ms")
        return df
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', since: Optional[datetime] = None,
                    limit: int = 500, use_cache: bool = True,
                    preferred_exchange: Optional[str] = None) -> DataFetchResult:
        start_time = time.time()
        if use_cache:
            for exchange_id in self.exchange_manager.exchanges.keys():
                cache_path = self._get_cache_path(exchange_id, symbol, timeframe)
                cached_df = self._load_from_cache(cache_path, self.cache_ttl_hours)
                if cached_df is not None and len(cached_df) >= limit * 0.8:
                    return DataFetchResult(success=True, data=cached_df, exchange=exchange_id,
                                           symbol=symbol, timeframe=timeframe,
                                           latency_ms=(time.time() - start_time) * 1000,
                                           rows_fetched=len(cached_df))
        since_ms = int(since.timestamp() * 1000) if since else None
        exchanges_to_try = []
        if preferred_exchange and preferred_exchange in self.exchange_manager.exchanges:
            exchanges_to_try.append(self.exchange_manager.exchanges[preferred_exchange])
        for ex_id, ex in self.exchange_manager.exchanges.items():
            if ex not in exchanges_to_try:
                exchanges_to_try.append(ex)
        last_error = None
        for exchange in exchanges_to_try:
            exchange_id = exchange.id
            try:
                if symbol not in exchange.symbols:
                    continue
                df = self._fetch_from_exchange(exchange, symbol, timeframe, since=since_ms, limit=limit)
                if len(df) < 10:
                    logger.warning(f"Insufficient data from {exchange_id}: {len(df)} rows")
                    continue
                if use_cache:
                    self._save_to_cache(df, self._get_cache_path(exchange_id, symbol, timeframe))
                return DataFetchResult(success=True, data=df, exchange=exchange_id, symbol=symbol,
                                       timeframe=timeframe, latency_ms=(time.time() - start_time) * 1000,
                                       rows_fetched=len(df))
            except ccxt.BadSymbol:
                continue
            except Exception as e:
                last_error = f"{exchange_id}: {str(e)}"
                logger.warning(f"Failed to fetch from {exchange_id}: {e}")
                continue
        return DataFetchResult(success=False, data=None, exchange="", symbol=symbol, timeframe=timeframe,
                               error=f"All exchanges failed. Last error: {last_error}",
                               latency_ms=(time.time() - start_time) * 1000)
    
    def fetch_multiple(self, symbols: List[str], timeframe: str = '1h', limit: int = 500) -> Dict[str, DataFetchResult]:
        results = {}
        for symbol in symbols:
            results[symbol] = self.fetch_ohlcv(symbol, timeframe, limit=limit)
            time.sleep(0.5)
        successful = sum(1 for r in results.values() if r.success)
        logger.info(f"📈 Batch fetch complete: {successful}/{len(symbols)} successful")
        return results
    
    def get_latest_price(self, symbol: str, preferred_exchange: Optional[str] = None) -> Optional[float]:
        exchange = self.exchange_manager.get_exchange(preferred_exchange)
        try:
            self.exchange_manager._rate_limit(exchange.id)
            return exchange.fetch_ticker(symbol).get('last')
        except Exception as e:
            logger.warning(f"Failed to get ticker for {symbol}: {e}")
            return None


class DataPipeline:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.exchange_manager = ExchangeManager(config_path)
        self.fetcher = DataFetcher(self.exchange_manager)
        self.assets_config = self._load_assets_config()
    
    def _load_config(self, path: str) -> dict:
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _load_assets_config(self) -> dict:
        try:
            with open("config/assets.yaml", 'r') as f:
                return yaml.safe_load(f)
        except Exception:
            return {'assets': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'], 'timeframes': ['1h', '4h', '1d']}
    
    def run_pipeline(self, timeframe: Optional[str] = None) -> Dict[str, DataFetchResult]:
        assets = self.assets_config.get('assets', [])
        timeframes = [timeframe] if timeframe else self.assets_config.get('timeframes', ['1h'])
        all_results = {}
        for tf in timeframes:
            logger.info(f"🔄 Processing timeframe: {tf}")
            results = self.fetcher.fetch_multiple(assets, tf)
            all_results.update({f"{k}_{tf}": v for k, v in results.items()})
        return all_results
    
    def validate_data_quality(self, result: DataFetchResult) -> Tuple[bool, List[str]]:
        if not result.success or result.data is None:
            return False, ["Fetch failed"]
        issues = []
        df = result.data
        if df.isnull().sum().any():
            issues.append("Missing values detected")
        if (df[['open', 'high', 'low', 'close']] == 0).any().any():
            issues.append("Zero prices detected")
        if (df['low'] > df['high']).any():
            issues.append("Invalid high/low relationship")
        if (df['volume'] < 0).any():
            issues.append("Negative volume detected")
        return len(issues) == 0, issues


def create_data_fetcher(config_path: str = "config/settings.yaml") -> DataFetcher:
    return DataFetcher(ExchangeManager(config_path))


def quick_fetch(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 100) -> Optional[pd.DataFrame]:
    try:
        result = create_data_fetcher().fetch_ohlcv(symbol, timeframe, limit=limit)
        return result.data if result.success else None
    except Exception as e:
        logger.error(f"Quick fetch failed: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🚀 AEGIS Data Fetcher - Testing Mode")
    try:
        pipeline = DataPipeline()
        result = pipeline.fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=50)
        if result.success:
            print(f"✅ Success from {result.exchange}: {result.rows_fetched} rows, {result.latency_ms:.0f}ms")
            print(result.data.tail(3))
        else:
            print(f"❌ Failed: {result.error}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

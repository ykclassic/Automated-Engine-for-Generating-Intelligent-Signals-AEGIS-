"""
AEGIS Data Fetcher Module
Smart exchange selection based on geographic availability.
Updated for the AEGIS Multi-Job Pipeline.
"""

import os
import time
import logging
import sys
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import socket
import json

import pandas as pd
import numpy as np
import ccxt
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging to show up clearly in GitHub Action logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class ExchangeManager:
    """Manages exchange connections with smart geographic detection"""
    
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
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return yaml.safe_load(f) or {}
            return {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _get_proxies(self) -> Dict[str, str]:
        proxies = {}
        http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        if http_proxy: proxies['http'] = http_proxy
        if https_proxy: proxies['https'] = https_proxy
        return proxies
    
    def _test_exchange(self, exchange_id: str, timeout: int = 10) -> bool:
        """Checks if an exchange is blocked/accessible in the current runner region"""
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({'enableRateLimit': True, 'timeout': timeout * 1000})
            if self.proxies:
                exchange.proxies = self.proxies
            
            # Use a fast low-level socket check for connectivity
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(timeout)
            try:
                exchange.load_markets()
                return True
            except (ccxt.NetworkError, ccxt.AuthenticationError) as e:
                err_str = str(e).lower()
                if any(x in err_str for x in ["restricted", "403", "451", "denied"]):
                    logger.warning(f"🚫 {exchange_id} is region-blocked.")
                return False
            finally:
                socket.setdefaulttimeout(original_timeout)
        except Exception:
            return False
    
    def _initialize_exchanges_smart(self):
        logger.info("🔍 Detecting available exchanges...")
        working_count = 0
        
        # Priority 1: Global Friendly Exchanges
        for ex_id in self.GLOBAL_EXCHANGES:
            if self._test_exchange(ex_id):
                self._add_exchange(ex_id)
                working_count += 1
                if working_count >= 3: break
                
        # Priority 2: Alternative/Restricted (only if needed)
        if working_count < 2:
            for ex_id in self.RESTRICTED_EXCHANGES:
                if ex_id not in self.exchanges and self._test_exchange(ex_id):
                    self._add_exchange(ex_id)
                    working_count += 1
                    if working_count >= 3: break
                    
        if not self.exchanges:
            logger.error("❌ Critical: No exchanges reachable.")

    def _add_exchange(self, exchange_id: str):
        try:
            exchange_class = getattr(ccxt, exchange_id)
            config = {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}
            if self.proxies: config['proxies'] = self.proxies
            
            self.exchanges[exchange_id] = exchange_class(config)
            self.rate_limiters[exchange_id] = 1.0 
            self.last_request_time[exchange_id] = 0
            logger.info(f"✅ Initialized {exchange_id}")
        except Exception as e:
            logger.error(f"Failed to add {exchange_id}: {e}")

    def _rate_limit(self, exchange_id: str):
        min_interval = self.rate_limiters.get(exchange_id, 1.0)
        elapsed = time.time() - self.last_request_time.get(exchange_id, 0)
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time[exchange_id] = time.time()

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
    """Handles OHLCV fetching with persistence for AEGIS Pipeline"""
    
    def __init__(self, exchange_manager: ExchangeManager, data_dir: str = "data"):
        self.exchange_manager = exchange_manager
        self.raw_dir = Path(data_dir) / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError)), reraise=True)
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> DataFetchResult:
        start_time = time.time()
        last_error = None
        
        for ex_id, exchange in self.exchange_manager.exchanges.items():
            try:
                self.exchange_manager._rate_limit(ex_id)
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if not ohlcv: continue
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)
                
                # Save to raw directory for the next job in workflow
                safe_symbol = symbol.replace('/', '_').replace(':', '_')
                file_path = self.raw_dir / f"{safe_symbol}_{timeframe}.parquet"
                df.to_parquet(file_path, compression='zstd')
                
                return DataFetchResult(
                    success=True, data=df, exchange=ex_id, symbol=symbol,
                    timeframe=timeframe, rows_fetched=len(df),
                    latency_ms=(time.time() - start_time) * 1000
                )
            except Exception as e:
                last_error = str(e)
                continue
                
        return DataFetchResult(success=False, data=None, exchange="", symbol=symbol, 
                               timeframe=timeframe, error=last_error)

def update_all_data(config_path: str = "config/settings.yaml") -> Dict[str, DataFetchResult]:
    """
    Main entry point for GitHub Actions Workflow 'fetch-data' job.
    """
    manager = ExchangeManager(config_path)
    fetcher = DataFetcher(manager)
    
    # Load assets from config/assets.yaml
    assets_path = Path("config/assets.yaml")
    if assets_path.exists():
        with open(assets_path, 'r') as f:
            assets_cfg = yaml.safe_load(f)
            assets = assets_cfg.get('assets', ['BTC/USDT'])
            timeframes = assets_cfg.get('timeframes', ['1h'])
    else:
        assets, timeframes = ['BTC/USDT'], ['1h']

    results = {}
    for tf in timeframes:
        for symbol in assets:
            key = f"{symbol}_{tf}"
            logger.info(f"⏳ Fetching {symbol} ({tf})...")
            results[key] = fetcher.fetch_ohlcv(symbol, tf)
            
    return results

if __name__ == "__main__":
    # Test execution
    update_all_data()

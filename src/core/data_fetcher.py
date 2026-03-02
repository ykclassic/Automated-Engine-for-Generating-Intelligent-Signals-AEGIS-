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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExchangeManager:
    """
    Manages exchange connections with smart geographic detection
    Prioritizes exchanges known to work in restricted regions
    """
    
    # Exchanges known to work in most regions (including US, EU, restricted areas)
    GLOBAL_EXCHANGES = ['kraken', 'coinbase', 'okx', 'gateio', 'kucoin', 'mexc']
    
    # Exchanges often blocked (China, US sanctions, etc.)
    RESTRICTED_EXCHANGES = ['binance', 'bybit', 'bitfinex', 'huobi']
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.rate_limiters: Dict[str, float] = {}
        self.last_request_time: Dict[str, float] = {}
        self.proxies = self._get_proxies()
        
        # Smart initialization - try global exchanges first
        self._initialize_exchanges_smart()
    
    def _load_config(self, path: str) -> dict:
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _get_proxies(self) -> Dict[str, str]:
        """Get proxy configuration from environment"""
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
        """
        Quick test if exchange is accessible without full initialization
        """
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': timeout * 1000,  # Convert to ms
            })
            
            if self.proxies:
                exchange.proxies = self.proxies
            
            # Quick test - just load markets with timeout
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
        """
        Initialize exchanges with smart geographic detection.
        Tests global exchanges first, falls back to restricted ones if needed.
        """
        logger.info("🔍 Testing exchange accessibility...")
        
        working_exchanges = []
        
        # First pass: Test global exchanges (prioritized)
        for exchange_id in self.GLOBAL_EXCHANGES:
            if self._test_exchange(exchange_id):
                working_exchanges.append(exchange_id)
                if len(working_exchanges) >= 3:  # Stop once we have 3 working
                    break
        
        # Second pass: If we have < 2 working, try restricted exchanges
        if len(working_exchanges) < 2:
            logger.info("⚠️ Limited global exchanges working, trying alternatives...")
            for exchange_id in self.RESTRICTED_EXCHANGES:
                if exchange_id not in working_exchanges and self._test_exchange(exchange_id):
                    working_exchanges.append(exchange_id)
                    if len(working_exchanges) >= 3:
                        break
        
        # Initialize working exchanges with full config
        for exchange_id in working_exchanges:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                config = {
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                        'adjustForTimeDifference': True,
                    }
                }
                
                # Add API keys if available in config
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
        """Apply rate limiting for exchange requests"""
        min_interval = self.rate_limiters.get(exchange_id, 1.0)
        last_time = self.last_request_time.get(exchange_id, 0)
        elapsed = time.time() - last_time
        
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time[exchange_id] = time.time()
    
    def get_exchange(self, preferred: Optional[str] = None) -> ccxt.Exchange:
        """
        Get best available exchange.
        
        Args:
            preferred: Preferred exchange ID if available
            
        Returns:
            ccxt.Exchange instance
        """
        if preferred and preferred in self.exchanges:
            return self.exchanges[preferred]
        
        # Return first available (prioritized by initialization order)
        return next(iter(self.exchanges.values()))
    
    def get_all_exchanges(self) -> List[ccxt.Exchange]:
        """Get all working exchanges for redundancy"""
        return list(self.exchanges.values())


@dataclass
class DataFetchResult:
    """Standardized result from data fetching operations"""
    success: bool
    data: Optional[pd.DataFrame]
    exchange: str
    symbol: str
    timeframe: str
    error: Optional[str] = None
    latency_ms: float = 0.0
    rows_fetched: int = 0


class DataFetcher:
    """
    High-level data fetching with multi-exchange fallback,
    caching, and intelligent retry logic.
    """
    
    def __init__(self, exchange_manager: ExchangeManager, cache_dir: str = "data/cache"):
        self.exchange_manager = exchange_manager
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
        self.cache_ttl_hours = 1  # 1 hour for recent data
        self.max_cache_age_days = 7
        
        logger.info(f"📁 Cache directory: {self.cache_dir.absolute()}")
    
    def _get_cache_path(self, exchange: str, symbol: str, timeframe: str) -> Path:
        """Generate cache file path"""
        safe_symbol = symbol.replace('/', '_').replace(':', '_')
        filename = f"{exchange}_{safe_symbol}_{timeframe}.parquet"
        return self.cache_dir / filename
    
    def _load_from_cache(self, cache_path: Path, max_age_hours: float) -> Optional[pd.DataFrame]:
        """Load data from cache if valid"""
        if not cache_path.exists():
            return None
        
        file_age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        
        if file_age_hours > max_age_hours:
            return None
        
        try:
            df = pd.read_parquet(cache_path)
            logger.debug(f"📂 Cache hit: {cache_path.name} ({file_age_hours:.1f}h old)")
            return df
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def _save_to_cache(self, df: pd.Data

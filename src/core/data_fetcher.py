"""
AEGIS Data Fetcher Module
Multi-exchange OHLCV data fetching with failover and caching
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import ccxt

from src.utils.logger import LoggerMixin
from src.utils.validators import DataValidator, ValidationResult
from src.utils.helpers import (
    ensure_datetime_utc,
    generate_cache_key,
    save_dataframe_atomic,
    timeframe_to_seconds
)


class ExchangeRateLimiter:
    """
    Token bucket rate limiter for exchange API compliance
    """
    
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.tokens = max_requests
        self.last_update = time.time()
        self.lock_time = 0
    
    def acquire(self) -> bool:
        """
        Acquire a rate limit token
        
        Returns:
            True if token acquired, False if rate limited
        """
        now = time.time()
        
        # Check if in lockout period
        if now < self.lock_time:
            return False
        
        # Add tokens based on time passed
        time_passed = now - self.last_update
        self.tokens = min(
            self.max_requests,
            self.tokens + (time_passed * self.max_requests / self.window_seconds)
        )
        self.last_update = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        
        return False
    
    def lockout(self, seconds: int):
        """Temporarily lockout requests"""
        self.lock_time = time.time() + seconds


class DataFetcher(LoggerMixin):
    """
    Institutional-grade data fetcher with:
    - Multi-exchange failover
    - Intelligent caching
    - Rate limiting
    - Data validation
    - Gap detection and filling
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        cache_dir: Path = Path("data/raw"),
        use_cache: bool = True
    ):
        super().__init__()
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize exchanges
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.rate_limiters: Dict[str, ExchangeRateLimiter] = {}
        self._init_exchanges()
        
        # Data validator
        self.validator = DataValidator(
            max_missing_pct=config['data']['quality']['max_missing_pct'],
            max_gap_minutes=config['data']['quality']['max_gap_minutes'],
            outlier_std_threshold=config['data']['quality']['outlier_std_threshold']
        )
        
        self.logger.info("DataFetcher initialized", 
                        primary_exchange=config['exchanges']['primary'])
    
    def _init_exchanges(self):
        """Initialize exchange connections with rate limiters"""
        exchange_configs = {
            'binance': {
                'rate_limit': 1200,
                'options': {'defaultType': 'spot'}
            },
            'bybit': {
                'rate_limit': 600,
                'options': {'defaultType': 'spot'}
            },
            'okx': {
                'rate_limit': 600,
                'options': {'defaultType': 'spot'}
            }
        }
        
        for exchange_id in [self.config['exchanges']['primary']] + \
                          self.config['exchanges'].get('fallbacks', []):
            try:
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'options': exchange_configs.get(exchange_id, {}).get('options', {})
                })
                
                self.exchanges[exchange_id] = exchange
                self.rate_limiters[exchange_id] = ExchangeRateLimiter(
                    max_requests=exchange_configs.get(exchange_id, {}).get('rate_limit', 600)
                )
                
                self.logger.info(f"Initialized exchange: {exchange_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to

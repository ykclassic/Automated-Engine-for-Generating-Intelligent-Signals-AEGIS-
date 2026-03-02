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
    

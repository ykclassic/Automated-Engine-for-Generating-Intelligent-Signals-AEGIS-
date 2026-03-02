import os
import time
import logging
import yaml
import ccxt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AEGIS-Fetcher")

@dataclass
class DataFetchResult:
    success: bool
    data: Optional[pd.DataFrame]
    exchange: str
    symbol: str
    timeframe: str
    error: Optional[str] = None
    rows_fetched: int = 0

class ExchangeManager:
    def __init__(self, config_path: Path):
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self._load_and_init(config_path)

    def _load_and_init(self, path: Path):
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Using Bitget as a reliable default for 2026/Runner environments
            exchange_id = 'bitget'
            ex_class = getattr(ccxt, exchange_id)
            self.exchanges[exchange_id] = ex_class({'enableRateLimit': True})
            logger.info(f"✅ Initialized {exchange_id} (Global Access)")
        except Exception as e:
            logger.error(f"Failed to init exchanges: {e}")

class DataPipeline:
    def __init__(self):
        # Resolve paths relative to the project root
        self.root = Path(__file__).parent.parent.parent
        self.config_path = self.root / "config" / "settings.yaml"
        self.assets_path = self.root / "config" / "assets.yaml"
        
        self.manager = ExchangeManager(self.config_path)
        self.assets_config = self._load_assets()

    def _load_assets(self) -> dict:
        if self.assets_path.exists():
            with open(self.assets_path, 'r') as f:
                return yaml.safe_load(f)
        return {'assets': ['BTC/USDT', 'ETH/USDT'], 'timeframes': ['1h']}

    def run_pipeline(self) -> Dict[str, DataFetchResult]:
        results = {}
        exchange = next(iter(self.manager.exchanges.values()))
        
        for symbol in self.assets_config['assets']:
            for tf in self.assets_config['timeframes']:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=100)
                    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
                    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                    results[f"{symbol}_{tf}"] = DataFetchResult(
                        success=True, data=df, exchange=exchange.id, 
                        symbol=symbol, timeframe=tf, rows_fetched=len(df)
                    )
                    logger.info(f"📊 Fetched {symbol} ({tf})")
                except Exception as e:
                    logger.warning(f"Failed {symbol}: {e}")
        return results

# --- THE MISSING LINK (Fixes ImportError) ---
def update_all_data():
    """
    Global entry point for the AEGIS system.
    Initializes the pipeline and returns the full dataset.
    """
    logger.info("🚀 AEGIS: Starting global data update...")
    pipeline = DataPipeline()
    return pipeline.run_pipeline()

if __name__ == "__main__":
    # Test local run
    update_all_data()

import os
import time
import logging
import yaml
import ccxt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AEGIS-Fetcher")

class DataPipeline:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent
        self.assets_path = self.root / "config" / "assets.yaml"
        self.cache_dir = self.root / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.exchange = ccxt.bitget({'enableRateLimit': True})
        self.assets, self.timeframes = self._parse_nested_assets()

    def _parse_nested_assets(self) -> Tuple[List[str], List[str]]:
        pairs = []
        default_tfs = ['1h']
        if not self.assets_path.exists():
            return ['BTC/USDT'], default_tfs

        try:
            with open(self.assets_path, 'r') as f:
                content = yaml.safe_load(f)
            categories = content.get('categories', {})
            for cat_key in categories:
                cat_pairs = categories[cat_key].get('pairs', {})
                for p_key, p_val in cat_pairs.items():
                    pairs.append(p_val.get('display_name', p_key))
            
            tfs = content.get('global', {}).get('timeframes', default_tfs)
            return pairs, tfs
        except Exception as e:
            logger.error(f"YAML Parse Error: {e}")
            return ['BTC/USDT'], default_tfs

    def run_pipeline(self) -> dict:
        results = {}
        for symbol in self.assets:
            for tf in self.timeframes:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=100)
                    if not ohlcv: continue
                    
                    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
                    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                    
                    # --- SAVE TO DISK ---
                    safe_name = symbol.replace('/', '_')
                    file_path = self.cache_dir / f"{safe_name}_{tf}.parquet"
                    df.to_parquet(file_path, index=False)
                    
                    results[f"{symbol}_{tf}"] = str(file_path)
                    logger.info(f"💾 Saved {symbol} to {file_path.name}")
                    time.sleep(0.1) 
                except Exception as e:
                    logger.warning(f"⚠️ Skip {symbol}: {e}")
        return results

def update_all_data():
    logger.info("🚀 AEGIS: Starting Data Fetch & Persistence...")
    pipeline = DataPipeline()
    return pipeline.run_pipeline()

if __name__ == "__main__":
    update_all_data()

import os
import time
import logging
import yaml
import ccxt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AEGIS-Fetcher")

class DataPipeline:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent
        self.assets_path = self.root / "config" / "assets.yaml"
        self.config_path = self.root / "config" / "settings.yaml"
        
        # Initialize Exchange
        self.exchange = ccxt.bitget({'enableRateLimit': True})
        self.assets, self.timeframes = self._parse_nested_assets()

    def _parse_nested_assets(self) -> Tuple[List[str], List[str]]:
        """Extracts trading pairs from highly nested/categorized YAML."""
        pairs = []
        default_tfs = ['1h']
        
        if not self.assets_path.exists():
            return ['BTC/USDT'], default_tfs

        try:
            with open(self.assets_path, 'r') as f:
                content = yaml.safe_load(f)
            
            # Navigate to categories -> sub-categories -> pairs
            categories = content.get('categories', {})
            for cat_key in categories:
                cat_data = categories[cat_key]
                cat_pairs = cat_data.get('pairs', {})
                
                for p_key, p_val in cat_pairs.items():
                    # Prioritize the 'display_name' if it exists (e.g. BTC/USDT)
                    # Fallback to the key (e.g. BTCUSDT) if not
                    display_name = p_val.get('display_name', p_key)
                    pairs.append(display_name)
            
            # Grab global update frequency or timeframe
            tfs = content.get('global', {}).get('timeframes', default_tfs)
            
            logger.info(f"✅ Extracted {len(pairs)} pairs from {len(categories)} categories.")
            return pairs, tfs
        except Exception as e:
            logger.error(f"Failed to parse categorized YAML: {e}")
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
                    results[f"{symbol}_{tf}"] = {"success": True, "data": df}
                    logger.info(f"📊 Fetched {symbol}")
                    time.sleep(0.1) # Respect rate limits
                except Exception as e:
                    logger.warning(f"⚠️ Skip {symbol}: {e}")
        return results

def update_all_data():
    """Main entry point for AEGIS Runner."""
    logger.info("🚀 AEGIS: Starting Category-Aware Update...")
    pipeline = DataPipeline()
    return pipeline.run_pipeline()

if __name__ == "__main__":
    update_all_data()

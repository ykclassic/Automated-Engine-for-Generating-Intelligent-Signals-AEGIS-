import os
import time
import logging
import yaml
import ccxt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AEGIS-Fetcher")

class DataPipeline:
    def __init__(self):
        # Explicit path resolution targeting data/raw for GitHub Actions compatibility
        self.root = Path(__file__).resolve().parent.parent.parent
        self.assets_path = self.root / "config" / "assets.yaml"
        self.raw_dir = self.root / "data" / "raw"
        
        # Ensure the raw directory exists before we start saving files
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize exchange
        self.exchange = ccxt.bitget({'enableRateLimit': True})
        self.assets, self.timeframes = self._parse_nested_assets()

    def _parse_nested_assets(self) -> Tuple[List[str], List[str]]:
        """Extracts trading pairs safely from the categorized YAML structure."""
        pairs = []
        default_tfs = ['1h']
        
        if not self.assets_path.exists():
            logger.error(f"❌ Assets config missing at {self.assets_path}. Using defaults.")
            return ['BTC/USDT'], default_tfs

        try:
            with open(self.assets_path, 'r') as f:
                content = yaml.safe_load(f)
            
            categories = content.get('categories', {})
            for cat_key, cat_data in categories.items():
                cat_pairs = cat_data.get('pairs', {})
                for p_key, p_val in cat_pairs.items():
                    # Prioritize the 'display_name' (e.g., BTC/USDT)
                    pairs.append(p_val.get('display_name', p_key))
            
            # Extract global timeframe or default to 1h
            tfs = content.get('global', {}).get('timeframes', default_tfs)
            logger.info(f"✅ Loaded {len(pairs)} symbols from {self.assets_path.name}")
            return pairs, tfs
        except Exception as e:
            logger.error(f"YAML Parse Error: {e}")
            return ['BTC/USDT'], default_tfs

    def run_pipeline(self) -> dict:
        """Fetches data from exchange and saves to disk (Used by Job 1)."""
        summary = {"count": 0, "files": []}
        
        for symbol in self.assets:
            for tf in self.timeframes:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=100)
                    if not ohlcv: 
                        continue
                    
                    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
                    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                    
                    # File naming: BTC/USDT -> BTC_USDT_1h.parquet
                    safe_name = symbol.replace('/', '_').replace(':', '_')
                    file_path = self.raw_dir / f"{safe_name}_{tf}.parquet"
                    
                    # Save to data/raw as required by the GitHub Workflow
                    df.to_parquet(file_path, index=False)
                    summary["count"] += 1
                    summary["files"].append(str(file_path))
                    
                    logger.info(f"💾 Saved: {file_path.name} to data/raw/")
                    time.sleep(0.1) # Respect exchange rate limits
                except Exception as e:
                    logger.warning(f"⚠️ Failed {symbol}: {e}")
        
        return summary

    def fetch_all_assets(self) -> Dict[str, pd.DataFrame]:
        """Reads saved Parquet files from disk into memory (Used by Job 2)."""
        data_dict = {}
        logger.info(f"📥 Loading assets from {self.raw_dir}")
        
        if not self.raw_dir.exists():
            logger.warning("Raw directory does not exist.")
            return data_dict

        for file_path in self.raw_dir.glob("*.parquet"):
            try:
                # Extract symbol name from filename (e.g., 'BTC_USDT_1h')
                asset_key = file_path.stem 
                df = pd.read_parquet(file_path)
                data_dict[asset_key] = df
                logger.info(f"✅ Loaded {file_path.name} into memory.")
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")
                
        return data_dict

def update_all_data():
    """Global entry point for Job 1 in AEGIS workflow."""
    logger.info("🚀 AEGIS: Starting Data Fetch & Persistence to data/raw...")
    pipeline = DataPipeline()
    summary = pipeline.run_pipeline()
    
    data_ok = summary["count"] > 0
    logger.info(f"🏁 Final status: data_available={data_ok}, count={summary['count']}")
    
    # Returning the dictionary ensures len(result) > 0 evaluates to True in your runner
    return summary

if __name__ == "__main__":
    update_all_data()

"""
AEGIS Signal Generation Module
Job 3 in the AEGIS Pipeline: Generates actionable trading signals 
using processed features and institutional risk management.
"""

import os
import logging
import sys
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Robust Import Logic for Pipeline Environment
try:
    from src.core.risk_management import RiskManager, RiskLevel
except ImportError:
    # Fallback for local or specific runner environments
    sys.path.append(str(Path(__file__).resolve().parent))
    try:
        from risk_management import RiskManager, RiskLevel
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.error("Could not import RiskManager. Ensure risk_management.py exists.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    AEGIS Signal Engine with Integrated Risk Management
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        
        # Risk Configuration
        risk_cfg = self.config.get('risk', {})
        risk_level_str = risk_cfg.get('level', 'moderate').lower()
        
        risk_map = {
            'conservative': RiskLevel.CONSERVATIVE,
            'moderate': RiskLevel.MODERATE,
            'aggressive': RiskLevel.AGGRESSIVE
        }
        
        self.risk_manager = RiskManager(risk_level=risk_map.get(risk_level_str, RiskLevel.MODERATE))
        self.min_confidence = self.config.get('signals', {}).get('min_confidence', 0.65)

    def _load_config(self, path: str) -> dict:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return yaml.safe_load(f) or {}
            return {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _calculate_confluence(self, row: pd.Series) -> float:
        """
        Calculates a confidence score (0.0 - 1.0) based on processed features.
        Weights are biased toward Trend and Momentum confluence.
        """
        # Feature-based scoring
        trend_score = 0.7 if row.get('ema_50_200_ratio', 1) > 1.0 else 0.3
        momentum_score = 0.8 if 40 < row.get('rsi', 50) < 70 else 0.4
        vol_score = 1.0 if row.get('bb_position', 0.5) > 0.2 and row.get('bb_position', 0.5) < 0.8 else 0.5
        
        score = (trend_score * 0.4) + (momentum_score * 0.4) + (vol_score * 0.2)
        return round(float(score), 2)

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Analyzes the latest bar and returns a signal if confidence meets threshold."""
        if df is None or df.empty or len(df) < 5:
            return None

        latest = df.iloc[-1]
        
        # 1. Global Risk Check
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.warning(f"Risk Block for {symbol}: {reason}")
            return None

        # 2. Confluence Analysis
        confidence = self._calculate_confluence(latest)
        direction = "LONG" if confidence >= self.min_confidence else "SHORT" if confidence <= (1 - self.min_confidence) else None
        
        if not direction:
            return None

        # 3. SL/TP Logic (Based on ATR from feature_engineering)
        entry_price = float(latest['close'])
        atr = latest.get('atr', entry_price * 0.02)
        
        sl = entry_price - (atr * 2) if direction == "LONG" else entry_price + (atr * 2)
        tp = entry_price + (atr * 4) if direction == "LONG" else entry_price - (atr * 4)

        # 4. Position Sizing
        # Note: account_balance would ideally come from an exchange API; here we use a default.
        sizing = self.risk_manager.calculate_position_size(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            account_balance=self.config.get('trading', {}).get('nominal_balance', 10000.0)
        )

        return {
            'signal_id': f"{symbol.replace('/', '')}_{int(datetime.now().timestamp())}",
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'confidence': confidence,
            'risk_metrics': {
                'position_size_units': round(sizing.size_units, 8),
                'position_size_pct': f"{sizing.size_pct:.2%}",
                'leverage': sizing.leverage,
                'stop_loss': round(sizing.stop_loss_price, 2),
                'take_profit': round(sizing.take_profit_price, 2),
                'risk_reward': round(sizing.risk_reward_ratio, 2)
            }
        }

def process_all_signals(data_dir: str = "data"):
    """
    Pipeline entry point: Reads data/processed/ and writes to signals/
    """
    proc_path = Path(data_dir) / "processed"
    sig_path = Path("signals")
    sig_path.mkdir(parents=True, exist_ok=True)
    
    generator = SignalGenerator()
    all_signals = []

    processed_files = list(proc_path.glob("*.parquet"))
    if not processed_files:
        logger.error("No processed feature files found.")
        return

    for file in processed_files:
        try:
            symbol = file.stem.replace('_', '/').split('/')[0] + '/' + file.stem.split('_')[1] # Simple parsing
            df = pd.read_parquet(file)
            signal = generator.generate_signal(df, symbol)
            
            if signal:
                all_signals.append(signal)
                logger.info(f"🚀 SIGNAL GENERATED: {symbol} {signal['direction']} at {signal['entry_price']}")
        except Exception as e:
            logger.error(f"Error processing signals for {file.name}: {e}")

    # Save summary signal file for deployment/notification job
    if all_signals:
        output_file = sig_path / f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(output_file, 'w') as f:
            json.dump(all_signals, f, indent=4)
        logger.info(f"✅ Saved {len(all_signals)} signals to {output_file}")
    else:
        logger.info("ℹ️ No high-confidence signals identified in this cycle.")

if __name__ == "__main__":
    process_all_signals()

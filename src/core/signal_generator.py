"""
AEGIS Signal Generation Engine
Combines indicators and market regime to produce trade signals
"""

import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

# Add project root to sys.path for absolute imports
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

try:
    from src.indicators import IndicatorOrchestrator
except ImportError:
    from indicators import IndicatorOrchestrator

logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Analyzes processed data to generate buy/sell signals
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.orchestrator = IndicatorOrchestrator()
        self.config = config or {}
        self.min_confluence = self.config.get('min_confluence', 0.7)
        
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """
        Processes a single asset's dataframe and returns signals
        """
        if df is None or len(df) < 50:
            return []
            
        # 1. Calculate all indicators
        df = self.orchestrator.calculate_all(df)
        
        # 2. Get the most recent candle
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        
        # Logic: Trend + Momentum + Volatility Confluence
        # Note: Using .get() to avoid KeyErrors if an indicator fails
        is_bullish_trend = latest.get('trend_bullish_score', 0) > 0.6
        is_oversold = latest.get('rsi', 50) < 35
        is_volume_confirm = latest.get('relative_volume', 1.0) > 1.2
        
        # LONG SIGNAL
        if is_bullish_trend and (is_oversold or latest.get('stoch_k', 50) < 20):
            entry_price = latest['close']
            atr = latest.get('atr', entry_price * 0.02)
            
            signals.append({
                'signal_id': f"SIG_{symbol}_{latest.name if isinstance(latest.name, str) else datetime.now().strftime('%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'direction': 'long',
                'entry_price': float(entry_price),
                'stop_loss': float(entry_price - (atr * 2)),
                'take_profit': float(entry_price + (atr * 4)),
                'confidence': float(latest.get('trend_bullish_score', 0.5)),
                'indicators': self.orchestrator.get_indicator_summary(df)
            })
            
        # SHORT SIGNAL
        elif latest.get('trend_bullish_score', 1.0) < 0.4 and latest.get('rsi', 50) > 65:
            entry_price = latest['close']
            atr = latest.get('atr', entry_price * 0.02)
            
            signals.append({
                'signal_id': f"SIG_{symbol}_{datetime.now().strftime('%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'direction': 'short',
                'entry_price': float(entry_price),
                'stop_loss': float(entry_price + (atr * 2)),
                'take_profit': float(entry_price - (atr * 4)),
                'confidence': float(1.0 - latest.get('trend_bullish_score', 0.5)),
                'indicators': self.orchestrator.get_indicator_summary(df)
            })
            
        return signals

if __name__ == "__main__":
    # Test block for pipeline
    logging.basicConfig(level=logging.INFO)
    print("Signal Generator initialized and ready.")

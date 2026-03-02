"""
AEGIS Performance Tracking Module
Tracks signal accuracy and strategy performance
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import json

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """Complete trade record"""
    signal_id: str
    timestamp: datetime
    symbol: str
    direction: str
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    size: float
    pnl: Optional[float]
    exit_reason: Optional[str]
    holding_periods: Optional[int]

class PerformanceTracker:
    """Tracks and analyzes trading performance"""
    
    def __init__(self, storage_path: str = "data/processed/performance.json"):
        # Explicit path resolution for CI/CD environments
        self.root = Path(__file__).resolve().parent.parent.parent
        self.storage_path = self.root / storage_path
        self.trades: List[TradeRecord] = []
        self.signals: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.initial_capital = 10000
        self.current_equity = self.initial_capital
        
        self.load_history()

    def _json_serial(self, obj: Any) -> Any:
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")

    def load_history(self):
        """Load historical performance data with safety checks"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    # Handle empty or malformed files
                    trade_data = data.get('trades', [])
                    self.trades = []
                    for t in trade_data:
                        # Convert ISO strings back to datetime
                        if isinstance(t.get('timestamp'), str):
                            t['timestamp'] = datetime.fromisoformat(t['timestamp'])
                        self.trades.append(TradeRecord(**t))
                    
                    self.equity_curve = data.get('equity_curve', [])
                    if self.equity_curve:
                        self.current_equity = self.equity_curve[-1]['equity']
                logger.info(f"✅ Loaded {len(self.trades)} historical trades.")
            except Exception as e:
                logger.error(f"Error loading performance history: {e}")

    def save_history(self):
        """Save performance data using custom serializer"""
        try:
            data = {
                'trades': [asdict(t) for t in self.trades],
                'equity_curve': self.equity_curve
            }
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=self_json_serial)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def record_signal(self, signal: Dict):
        """Record generated signal from SignalGenerator"""
        # Ensure consistent timestamping
        if 'timestamp' not in signal:
            signal['timestamp'] = datetime.now()
        self.signals.append(signal)

    def update_signal_outcome(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Check open signals against current market prices.
        current_prices: {'BTC/USDT': 65000.0, ...}
        """
        completed = []
        now = datetime.now()
        
        for signal in self.signals:
            if signal.get('completed'): continue
            
            symbol = signal.get('symbol')
            # Handle key mapping if data_fetcher uses underscores
            price = current_prices.get(symbol) or current_prices.get(symbol.replace('/', '_'))
            
            if price is None: continue

            # Logic for Exit
            entry = signal['entry_price']
            stop = signal['stop_loss']
            target = signal['take_profit']
            direction = signal['direction']
            size = signal.get('size', 100) # Default size for calculation

            # Expiry Check (4 hours)
            sig_time = signal['timestamp']
            if isinstance(sig_time, str): sig_time = datetime.fromisoformat(sig_time)
            
            outcome = None
            exit_price = None

            if now - sig_time > timedelta(hours=4):
                outcome, exit_price = 'expiry', price
            elif direction == 'long':
                if price <= stop: outcome, exit_price = 'stop_loss', stop
                elif price >= target: outcome, exit_price = 'take_profit', target
            elif direction == 'short':
                if price >= stop: outcome, exit_price = 'stop_loss', stop
                elif price <= target: outcome, exit_price = 'take_profit', target

            if outcome:
                pnl_pct = (exit_price - entry) / entry if direction == 'long' else (entry - exit_price) / entry
                pnl_usd = pnl_pct * size
                
                signal.update({'completed': True, 'outcome': outcome, 'exit_price': exit_price, 'pnl': pnl_usd})
                
                new_trade = TradeRecord(
                    signal_id=str(hash(sig_time)),
                    timestamp=now,
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry,
                    exit_price=exit_price,
                    stop_loss=stop,
                    take_profit=target,
                    size=size,
                    pnl=pnl_usd,
                    exit_reason=outcome,
                    holding_periods=1
                )
                self.record_trade(new_trade)
                completed.append(signal)
        
        return completed

    def record_trade(self, trade: TradeRecord):
        self.trades.append(trade)
        self.current_equity += (trade.pnl or 0)
        self.equity_curve.append({
            'timestamp': datetime.now().isoformat(),
            'equity': self.current_equity
        })
        self.save_history()

    def calculate_metrics(self) -> Dict:
        if not self.trades: return {"status": "No trades"}
        
        df = pd.DataFrame([asdict(t) for t in self.trades])
        win_rate = len(df[df['pnl'] > 0]) / len(df)
        total_pnl = df['pnl'].sum()
        
        return {
            'total_trades': len(df),
            'win_rate': f"{win_rate:.1%}",
            'total_pnl': f"${total_pnl:.2f}",
            'current_equity': f"${self.current_equity:.2f}",
            'profit_factor': self._calc_profit_factor(df)
        }

    def _calc_profit_factor(self, df):
        wins = df[df['pnl'] > 0]['pnl'].sum()
        losses = abs(df[df['pnl'] < 0]['pnl'].sum())
        return round(wins / losses, 2) if losses > 0 else "∞"

    def generate_report(self) -> str:
        m = self.calculate_metrics()
        if "status" in m: return m["status"]
        return (f"🛡️ AEGIS REPORT | Trades: {m['total_trades']} | "
                f"WinRate: {m['win_rate']} | PnL: {m['total_pnl']} | "
                f"Equity: {m['current_equity']}")

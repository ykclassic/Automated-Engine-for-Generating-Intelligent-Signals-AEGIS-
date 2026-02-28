"""
AEGIS Performance Tracking Module
Tracks signal accuracy and strategy performance
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
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
    exit_reason: Optional[str]  # 'stop', 'target', 'expiry', 'manual'
    holding_periods: Optional[int]


class PerformanceTracker:
    """
    Tracks and analyzes trading performance
    """
    
    def __init__(self, storage_path: str = "data/processed/performance.json"):
        self.storage_path = Path(storage_path)
        self.trades: List[TradeRecord] = []
        self.signals: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.initial_capital = 10000
        self.current_equity = self.initial_capital
        
        self.load_history()
    
    def load_history(self):
        """Load historical performance data"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    self.trades = [TradeRecord(**t) for t in data.get('trades', [])]
                    self.equity_curve = data.get('equity_curve', [])
            except Exception as e:
                logger.error(f"Error loading performance history: {e}")
    
    def save_history(self):
        """Save performance data"""
        data = {
            'trades': [
                {
                    'signal_id': t.signal_id,
                    'timestamp': t.timestamp.isoformat(),
                    'symbol': t.symbol,
                    'direction': t.direction,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'stop_loss': t.stop_loss,
                    'take_profit': t.take_profit,
                    'size': t.size,
                    'pnl': t.pnl,
                    'exit_reason': t.exit_reason,
                    'holding_periods': t.holding_periods
                }
                for t in self.trades
            ],
            'equity_curve': self.equity_curve
        }
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def record_signal(self, signal: Dict):
        """Record generated signal"""
        self.signals.append({
            'timestamp': datetime.now().isoformat(),
            **signal
        })
    
    def update_signal_outcome(
        self,
        symbol: str,
        timestamp: datetime,
        current_price: float
    ) -> List[Dict]:
        """
        Check open signals against current price
        Returns completed trades
        """
        completed = []
        
        for signal in self.signals:
            # Skip if already processed
            if signal.get('completed'):
                continue
            
            # Check if signal expired
            signal_time = datetime.fromisoformat(signal['timestamp'])
            if datetime.now() - signal_time > timedelta(hours=4):
                signal['completed'] = True
                signal['outcome'] = 'expired'
                continue
            
            entry = signal.get('entry_price', 0)
            stop = signal.get('stop_loss', 0)
            target = signal.get('take_profit', 0)
            direction = signal.get('direction', '')
            
            # Check stop loss
            if direction == 'long' and current_price <= stop:
                pnl = (stop - entry) / entry * signal.get('size', 0)
                signal['completed'] = True
                signal['outcome'] = 'stop_loss'
                signal['exit_price'] = stop
                signal['pnl'] = pnl
                completed.append(signal)
            
            elif direction == 'short' and current_price >= stop:
                pnl = (entry - stop) / entry * signal.get('size', 0)
                signal['completed'] = True
                signal['outcome'] = 'stop_loss'
                signal['exit_price'] = stop
                signal['pnl'] = pnl
                completed.append(signal)
            
            # Check take profit
            elif direction == 'long' and current_price >= target:
                pnl = (target - entry) / entry * signal.get('size', 0)
                signal['completed'] = True
                signal['outcome'] = 'take_profit'
                signal['exit_price'] = target
                signal['pnl'] = pnl
                completed.append(signal)
            
            elif direction == 'short' and current_price <= target:
                pnl = (entry - target) / entry * signal.get('size', 0)
                signal['completed'] = True
                signal['outcome'] = 'take_profit'
                signal['exit_price'] = target
                signal['pnl'] = pnl
                completed.append(signal)
        
        return completed
    
    def record_trade(self, trade: TradeRecord):
        """Record completed trade"""
        self.trades.append(trade)
        
        # Update equity
        if trade.pnl:
            self.current_equity += trade.pnl
        
        self.equity_curve.append({
            'timestamp': datetime.now().isoformat(),
            'equity': self.current_equity
        })
        
        self.save_history()
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if not self.trades:
            return {}
        
        df = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'pnl': t.pnl or 0,
                'exit_reason': t.exit_reason,
                'direction': t.direction,
                'holding_periods': t.holding_periods or 0
            }
            for t in self.trades
        ])
        
        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Returns
        total_return = (self.current_equity - self.initial_capital) / self.initial_capital
        
        # Calculate daily returns for Sharpe
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_returns = df.groupby('date')['pnl'].sum() / self.initial_capital
        
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(365) if daily_returns.std() > 0 else 0
        
        # Sortino (downside deviation only)
        downside_returns = daily_returns[daily_returns < 0]
        sortino_ratio = daily_returns.mean() / downside_returns.std() * np.sqrt(365) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            equity_df = equity_df.set_index('timestamp')['equity']
            running_max = equity_df.cummax()
            drawdown = (equity_df - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
        # Expectancy
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # By exit reason
        exit_reasons = df.groupby('exit_reason').agg({
            'pnl': ['count', 'sum', 'mean']
        }).to_dict()
        
        # Monthly aggregation
        df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M')
        monthly = df.groupby('month')['pnl'].sum().to_dict()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'current_equity': self.current_equity,
            'exit_reasons': exit_reasons,
            'monthly_returns': monthly,
            'equity_curve': self.equity_curve[-100:]  # Last 100 points
        }
    
    def generate_report(self) -> str:
        """
        Generate human-readable performance report
        """
        metrics = self.calculate_metrics()
        
        if not metrics:
            return "No trades recorded yet."
        
        report = f"""
        🛡️ AEGIS Performance Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        📊 Trade Statistics
        -------------------
        Total Trades: {metrics['total_trades']}
        Win Rate: {metrics['win_rate']:.1%}
        Profit Factor: {metrics['profit_factor']:.2f}
        
        💰 Financial Metrics
        -------------------
        Total Return: {metrics['total_return']:.2%}
        Current Equity: ${metrics['current_equity']:,.2f}
        Expectancy per Trade: ${metrics['expectancy']:.2f}
        
        📈 Risk Metrics
        --------------
        Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
        Sortino Ratio: {metrics['sortino_ratio']:.2f}
        Max Drawdown: {metrics['max_drawdown']:.2%}
        
        🎯 Trade Quality
        ---------------
        Average Win: ${metrics['avg_win']:.2f}
        Average Loss: ${metrics['avg_loss']:.2f}
        """
        
        return report


# Convenience function
def track_performance(signals: List[Dict], prices: Dict[str, float]) -> Dict:
    """Quick performance tracking"""
    tracker = PerformanceTracker()
    
    for signal in signals:
        tracker.record_signal(signal)
    
    for symbol, price in prices.items():
        completed = tracker.update_signal_outcome(symbol, datetime.now(), price)
        for comp in completed:
            logger.info(f"Signal completed: {comp['symbol']} - {comp['outcome']}")
    
    return tracker.calculate_metrics()

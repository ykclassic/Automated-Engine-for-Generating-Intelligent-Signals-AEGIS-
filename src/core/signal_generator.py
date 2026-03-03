import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

# AEGIS Internal Imports
try:
    from src.indicators import IndicatorOrchestrator
    from src.core.risk_management import RiskManager, RiskLevel
except ImportError:
    from indicators import IndicatorOrchestrator
    from risk_management import RiskManager, RiskLevel

logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    AEGIS Signal Engine with Integrated Risk Management
    """
    
    def __init__(self, min_confidence: float = 0.65, risk_level: str = 'moderate'):
        self.orchestrator = IndicatorOrchestrator()
        
        # Map string from workflow to the Enum in your Risk Module
        risk_map = {
            'conservative': RiskLevel.CONSERVATIVE,
            'moderate': RiskLevel.MODERATE,
            'aggressive': RiskLevel.AGGRESSIVE
        }
        selected_level = risk_map.get(risk_level.lower(), RiskLevel.MODERATE)
        
        # Initialize your Risk Manager
        self.risk_manager = RiskManager(risk_level=selected_level)
        self.min_confidence = min_confidence

    def _calculate_confluence(self, row: pd.Series) -> float:
        """Weights indicators into a 0.0 - 1.0 score"""
        score = (
            row.get('trend_bullish_score', 0.5) * 0.4 +
            row.get('momentum_score', 0.5) * 0.3 +
            (1.0 if row.get('volatility_regime') == 'normal' else 0.5) * 0.2 +
            (1.0 if row.get('relative_volume', 1.0) > 1.5 else 0.5) * 0.1
        )
        return round(float(score), 2)

    def generate_signals(self, df: pd.DataFrame, symbol: str, account_balance: float = 10000.0) -> List[Dict]:
        if df is None or df.empty or len(df) < 50:
            return []

        # 1. Calculate Indicators
        df = self.orchestrator.calculate_all(df)
        latest = df.iloc[-1]
        
        # 2. Check for Circuit Breakers
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.warning(f"Risk Manager Halt: {reason}")
            return []

        confidence = self._calculate_confluence(latest)
        direction = "LONG" if confidence >= self.min_confidence else "SHORT" if confidence <= (1 - self.min_confidence) else None
        
        if not direction:
            return []

        # 3. Calculate Position Sizing using your Risk Module logic
        entry_price = float(latest['close'])
        atr = latest.get('atr', entry_price * 0.02)
        
        # Define SL/TP based on ATR
        sl = entry_price - (atr * 2) if direction == "LONG" else entry_price + (atr * 2)
        tp = entry_price + (atr * 4) if direction == "LONG" else entry_price - (atr * 4)

        # Use your RiskManager's Kelly Criterion logic
        sizing = self.risk_manager.calculate_position_size(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            account_balance=account_balance
        )

        return [{
            'signal_id': f"{symbol.replace('/', '')}_{int(datetime.now().timestamp())}",
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'confidence': confidence,
            'risk_metrics': {
                'position_size_units': round(sizing.size_units, 4),
                'position_size_pct': f"{sizing.size_pct:.2%}",
                'leverage': sizing.leverage,
                'stop_loss': round(sizing.stop_loss_price, 2),
                'take_profit': round(sizing.take_profit_price, 2),
                'risk_reward': round(sizing.risk_reward_ratio, 2)
            },
            'indicator_snapshot': self.orchestrator.get_indicator_summary(df)
        }]

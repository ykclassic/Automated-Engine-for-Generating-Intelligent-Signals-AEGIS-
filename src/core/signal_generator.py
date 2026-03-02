"""
AEGIS Signal Generator
Intelligent signal filtering and quality control
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yaml

# FIXED: Absolute imports for GitHub Actions environment
try:
    from indicators.indicator_orchestrator import IndicatorOrchestrator
    from analysis.market_regime import RegimeDetector, MarketRegime
    from analysis.confluence import ConfluenceEngine, ConfluenceScore
    from analysis.correlation import MultiTimeframeAnalyzer
    from core.risk_manager import RiskManager, RiskLevel
except ImportError:
    # Fallback for environments where 'src' is the direct root
    from src.indicators.indicator_orchestrator import IndicatorOrchestrator
    from src.analysis.market_regime import RegimeDetector, MarketRegime
    from src.analysis.confluence import ConfluenceEngine, ConfluenceScore
    from src.analysis.correlation import MultiTimeframeAnalyzer
    from src.core.risk_manager import RiskManager, RiskLevel

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Complete trading signal"""
    timestamp: datetime
    symbol: str
    direction: str  # long, short, neutral
    confidence: str  # very_high, high, moderate, low
    confidence_score: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: Dict
    timeframe_confluence: Dict
    indicators: Dict
    risk_metrics: Dict
    ml_prediction: Optional[Dict] = None
    expiration: Optional[datetime] = None


class SignalFilter:
    """Filters signals based on quality criteria"""
    
    def __init__(self):
        self.min_confidence = 0.6
        self.min_risk_reward = 1.5
        self.max_daily_signals = 10
        self.signal_cooldown = timedelta(hours=4)
        
        self.recent_signals: List[Dict] = []
        self.daily_signal_count = 0
        self.last_reset = datetime.now()
    
    def check_signal_quality(self, signal: ConfluenceScore) -> Tuple[bool, str]:
        if signal.confidence < self.min_confidence:
            return False, f"Confidence {signal.confidence:.2f} below threshold {self.min_confidence}"
        if signal.overall_direction.value == 0:
            return False, "Signal direction is neutral"
        if signal.agreement_ratio < 0.6:
            return False, f"Low agreement ratio: {signal.agreement_ratio:.2f}"
        return True, "Passed quality checks"
    
    def check_time_filters(self, symbol: str, direction: str, timestamp: datetime) -> Tuple[bool, str]:
        if timestamp.date() != self.last_reset.date():
            self.daily_signal_count = 0
            self.last_reset = timestamp
        if self.daily_signal_count >= self.max_daily_signals:
            return False, "Daily signal limit reached"
        for recent in self.recent_signals:
            if (recent['symbol'] == symbol and 
                recent['direction'] == direction and
                timestamp - recent['timestamp'] < self.signal_cooldown):
                return False, f"Signal in cooldown period for {symbol}"
        return True, "Passed time filters"
    
    def check_regime_compatibility(self, regime: MarketRegime, direction: str) -> Tuple[bool, str]:
        if regime == MarketRegime.STRONG_UPTREND and direction == 'short':
            return False, "Short signal in strong uptrend"
        if regime == MarketRegime.STRONG_DOWNTREND and direction == 'long':
            return False, "Long signal in strong downtrend"
        return True, "Regime compatible"
    
    def record_signal(self, signal: Dict):
        self.recent_signals.append({
            'symbol': signal['symbol'],
            'direction': signal['direction'],
            'timestamp': signal['timestamp']
        })
        cutoff = datetime.now() - timedelta(days=7)
        self.recent_signals = [s for s in self.recent_signals if s['timestamp'] > cutoff]
        self.daily_signal_count += 1


class SignalGenerator:
    """Main signal generation orchestrator"""
    
    def __init__(self, risk_level: RiskLevel = RiskLevel.MODERATE, use_ml: bool = True):
        self.indicator_orchestrator = IndicatorOrchestrator()
        self.regime_detector = RegimeDetector()
        self.confluence_engine = ConfluenceEngine()
        self.tf_analyzer = MultiTimeframeAnalyzer()
        self.risk_manager = RiskManager(risk_level=risk_level)
        self.signal_filter = SignalFilter()
        self.use_ml = use_ml
        
        if use_ml:
            try:
                from ml.predict import PredictionEngine
                self.ml_engine = PredictionEngine()
            except Exception as e:
                logger.warning(f"ML engine not available: {e}")
                self.ml_engine = None
    
    def generate_signal(self, symbol: str, timeframe_data: Dict[str, pd.DataFrame], account_balance: float = 10000) -> Optional[TradingSignal]:
        timestamp = datetime.now()
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            return None
        
        # Primary timeframe logic
        primary_key = '1h'
        if primary_key not in timeframe_data:
            # Flexible matching for data_fetcher keys (e.g., BTC_USDT_1h)
            found_key = next((k for k in timeframe_data.keys() if '1h' in k), None)
            if not found_key:
                logger.error(f"1h timeframe not found for {symbol}")
                return None
            primary_key = found_key

        primary_df = timeframe_data[primary_key]
        primary_df = self.indicator_orchestrator.calculate_all(primary_df)
        regime = self.regime_detector.detect_regime(primary_df)
        confluence = self.confluence_engine.calculate_confluence(primary_df)
        
        passes_quality, quality_msg = self.signal_filter.check_signal_quality(confluence)
        if not passes_quality:
            return None
        
        direction_map = {-1: 'short', 0: 'neutral', 1: 'long'}
        direction = direction_map[confluence.overall_direction.value]
        if direction == 'neutral':
            return None
        
        passes_time, time_msg = self.signal_filter.check_time_filters(symbol, direction, timestamp)
        if not passes_time:
            return None
        
        passes_regime, regime_msg = self.signal_filter.check_regime_compatibility(regime, direction)
        if not passes_regime:
            return None
        
        # SL/TP Logic
        current_price = primary_df['close'].iloc[-1]
        atr = primary_df['atr'].iloc[-1] if 'atr' in primary_df.columns else current_price * 0.02
        stop_loss = current_price - (atr * 2) if direction == 'long' else current_price + (atr * 2)
        take_profit = current_price + (atr * 3) if direction == 'long' else current_price - (atr * 3)
        
        position_sizing = self.risk_manager.calculate_position_size(
            symbol=symbol, direction=direction, entry_price=current_price,
            stop_loss=stop_loss, take_profit=take_profit, account_balance=account_balance
        )
        
        if not self.risk_manager.check_portfolio_heat(position_sizing.max_loss_pct):
            return None
        
        return TradingSignal(
            timestamp=timestamp, symbol=symbol, direction=direction,
            confidence='high' if confluence.confidence > 0.7 else 'moderate',
            confidence_score=confluence.confidence, entry_price=current_price,
            stop_loss=stop_loss, take_profit=take_profit,
            position_size={
                'size_pct': position_sizing.size_pct,
                'size_units': position_sizing.size_units,
                'leverage': position_sizing.leverage,
                'max_loss_pct': position_sizing.max_loss_pct
            },
            timeframe_confluence={
                'score': confluence.overall_score,
                'dominant_category': confluence.dominant_category,
                'agreement_ratio': confluence.agreement_ratio,
                'signals': [{'name': s.name, 'direction': s.direction.name} for s in confluence.signals[:5]]
            },
            indicators=self.indicator_orchestrator.get_indicator_summary(primary_df),
            risk_metrics=self.risk_manager.get_risk_metrics().__dict__ if hasattr(self.risk_manager.get_risk_metrics(), '__dict__') else {},
            ml_prediction=None,
            expiration=timestamp + timedelta(hours=4)
        )

    def generate_all_signals(self, all_data: Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]], account_balance: float = 10000) -> List[TradingSignal]:
        signals = []
        for key, value in all_data.items():
            try:
                # Normalizing the input from data_fetcher
                symbol = key.split('_')[0] + "/" + key.split('_')[1] if '_' in key else key
                t_data = value if isinstance(value, dict) else {'1h': value}
                
                signal = self.generate_signal(symbol, t_data, account_balance)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating signal for {key}: {e}")
        
        signals.sort(key=lambda x: x.confidence_score, reverse=True)
        return signals[:5]

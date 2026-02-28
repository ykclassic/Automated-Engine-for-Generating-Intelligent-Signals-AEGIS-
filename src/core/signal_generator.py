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

from ..indicators import IndicatorOrchestrator
from ..analysis.market_regime import RegimeDetector, MarketRegime
from ..analysis.confluence import ConfluenceEngine, ConfluenceScore
from ..analysis.correlation import MultiTimeframeAnalyzer
from .risk_manager import RiskManager, RiskLevel

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
    """
    Filters signals based on quality criteria
    """
    
    def __init__(self):
        self.min_confidence = 0.6
        self.min_risk_reward = 1.5
        self.max_daily_signals = 10
        self.signal_cooldown = timedelta(hours=4)
        
        # Track signal history
        self.recent_signals: List[Dict] = []
        self.daily_signal_count = 0
        self.last_reset = datetime.now()
    
    def check_signal_quality(self, signal: ConfluenceScore) -> Tuple[bool, str]:
        """
        Check if signal meets quality thresholds
        """
        # Confidence check
        if signal.confidence < self.min_confidence:
            return False, f"Confidence {signal.confidence:.2f} below threshold {self.min_confidence}"
        
        # Direction clarity
        if signal.overall_direction.value == 0:  # Neutral
            return False, "Signal direction is neutral"
        
        # Agreement check
        if signal.agreement_ratio < 0.6:
            return False, f"Low agreement ratio: {signal.agreement_ratio:.2f}"
        
        return True, "Passed quality checks"
    
    def check_time_filters(
        self,
        symbol: str,
        direction: str,
        timestamp: datetime
    ) -> Tuple[bool, str]:
        """
        Check time-based filters (cooldown, daily limits)
        """
        # Reset daily count if new day
        if timestamp.date() != self.last_reset.date():
            self.daily_signal_count = 0
            self.last_reset = timestamp
        
        # Daily limit
        if self.daily_signal_count >= self.max_daily_signals:
            return False, "Daily signal limit reached"
        
        # Cooldown check
        for recent in self.recent_signals:
            if (recent['symbol'] == symbol and 
                recent['direction'] == direction and
                timestamp - recent['timestamp'] < self.signal_cooldown):
                return False, f"Signal in cooldown period for {symbol}"
        
        return True, "Passed time filters"
    
    def check_regime_compatibility(
        self,
        regime: MarketRegime,
        direction: str
    ) -> Tuple[bool, str]:
        """
        Check if signal direction matches market regime
        """
        # Don't trade against strong trends
        if regime == MarketRegime.STRONG_UPTREND and direction == 'short':
            return False, "Short signal in strong uptrend"
        
        if regime == MarketRegime.STRONG_DOWNTREND and direction == 'long':
            return False, "Long signal in strong downtrend"
        
        # Avoid ranging markets for trend following
        if regime == MarketRegime.RANGING:
            # Could still allow mean reversion signals
            pass
        
        return True, "Regime compatible"
    
    def record_signal(self, signal: Dict):
        """Record signal for tracking"""
        self.recent_signals.append({
            'symbol': signal['symbol'],
            'direction': signal['direction'],
            'timestamp': signal['timestamp']
        })
        
        # Keep only recent history
        cutoff = datetime.now() - timedelta(days=7)
        self.recent_signals = [
            s for s in self.recent_signals if s['timestamp'] > cutoff
        ]
        
        self.daily_signal_count += 1


class SignalGenerator:
    """
    Main signal generation orchestrator
    Combines all analysis layers into actionable signals
    """
    
    def __init__(
        self,
        risk_level: RiskLevel = RiskLevel.MODERATE,
        use_ml: bool = True
    ):
        self.indicator_orchestrator = IndicatorOrchestrator()
        self.regime_detector = RegimeDetector()
        self.confluence_engine = ConfluenceEngine()
        self.tf_analyzer = MultiTimeframeAnalyzer()
        self.risk_manager = RiskManager(risk_level=risk_level)
        self.signal_filter = SignalFilter()
        self.use_ml = use_ml
        
        if use_ml:
            try:
                from ..ml.predict import PredictionEngine
                self.ml_engine = PredictionEngine()
            except Exception as e:
                logger.warning(f"ML engine not available: {e}")
                self.ml_engine = None
    
    def generate_signal(
        self,
        symbol: str,
        timeframe_data: Dict[str, pd.DataFrame],
        account_balance: float = 10000
    ) -> Optional[TradingSignal]:
        """
        Generate complete trading signal for symbol
        """
        timestamp = datetime.now()
        
        # Check if we can trade
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.info(f"Cannot trade: {reason}")
            return None
        
        # Get primary timeframe (1h)
        if '1h' not in timeframe_data:
            logger.error("Primary timeframe (1h) not available")
            return None
        
        primary_df = timeframe_data['1h']
        
        # Calculate indicators
        primary_df = self.indicator_orchestrator.calculate_all(primary_df)
        
        # Detect regime
        regime = self.regime_detector.detect_regime(primary_df)
        
        # Calculate confluence
        confluence = self.confluence_engine.calculate_confluence(primary_df)
        
        # Quality filter
        passes_quality, quality_msg = self.signal_filter.check_signal_quality(confluence)
        if not passes_quality:
            logger.debug(f"Signal quality check failed: {quality_msg}")
            return None
        
        # Direction
        direction_map = {
            -1: 'short',
            0: 'neutral',
            1: 'long'
        }
        direction = direction_map[confluence.overall_direction.value]
        
        if direction == 'neutral':
            return None
        
        # Time filters
        passes_time, time_msg = self.signal_filter.check_time_filters(
            symbol, direction, timestamp
        )
        if not passes_time:
            logger.debug(f"Time filter failed: {time_msg}")
            return None
        
        # Regime compatibility
        passes_regime, regime_msg = self.signal_filter.check_regime_compatibility(
            regime, direction
        )
        if not passes_regime:
            logger.debug(f"Regime check failed: {regime_msg}")
            return None
        
        # Multi-timeframe analysis
        tf_signal = self.tf_analyzer.generate_multi_timeframe_signal(timeframe_data)
        
        # Check alignment
        if not tf_signal['alignment']:
            logger.debug("Multi-timeframe alignment check failed")
            # Could still proceed with reduced confidence
        
        # ML prediction (if available)
        ml_pred = None
        if self.ml_engine and self.use_ml:
            try:
                ml_pred = self.ml_engine.predict(primary_df)
                # Check ML agreement with technical signal
                if ml_pred['direction'] != direction:
                    logger.debug(f"ML disagreement: ML={ml_pred['direction']}, Tech={direction}")
                    # Could reduce confidence or reject signal
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
        
        # Calculate entry and exit levels
        current_price = primary_df['close'].iloc[-1]
        
        # Stop loss based on ATR or structure
        atr = primary_df['atr'].iloc[-1] if 'atr' in primary_df.columns else current_price * 0.02
        if direction == 'long':
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 3)  # 1.5:1 R:R minimum
        else:
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 3)
        
        # Override with support/resistance if available
        if 'support_levels' in tf_signal and direction == 'long':
            nearest_support = tf_signal['support_levels'][0]['price'] if tf_signal['support_levels'] else stop_loss
            stop_loss = max(stop_loss, nearest_support * 0.99)  # Slight buffer
        
        if 'resistance_levels' in tf_signal and direction == 'short':
            nearest_resistance = tf_signal['resistance_levels'][0]['price'] if tf_signal['resistance_levels'] else stop_loss
            stop_loss = min(stop_loss, nearest_resistance * 1.01)
        
        # Position sizing
        position_sizing = self.risk_manager.calculate_position_size(
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            account_balance=account_balance
        )
        
        # Check portfolio heat
        if not self.risk_manager.check_portfolio_heat(position_sizing.max_loss_pct):
            logger.info("Portfolio heat limit would be exceeded")
            return None
        
        # Create signal
        signal = TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            direction=direction,
            confidence='high' if confluence.confidence > 0.7 else 'moderate',
            confidence_score=confluence.confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
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
                'signals': [
                    {'name': s.name, 'category': s.category, 'direction': s.direction.name}
                    for s in confluence.signals[:5]
                ]
            },
            indicators=self.indicator_orchestrator.get_indicator_summary(primary_df),
            risk_metrics={
                'portfolio_heat': self.risk_manager.get_risk_metrics().portfolio_heat,
                'daily_var': self.risk_manager.get_risk_metrics().daily_var,
                'current_drawdown': self.risk_manager.get_risk_metrics().current_drawdown
            },
            ml_prediction=ml_pred,
            expiration=timestamp + timedelta(hours=4)
        )
        
        # Record signal
        self.signal_filter.record_signal({
            'symbol': symbol,
            'direction': direction,
            'timestamp': timestamp
        })
        
        return signal
    
    def generate_all_signals(
        self,
        all_data: Dict[str, Dict[str, pd.DataFrame]],
        account_balance: float = 10000
    ) -> List[TradingSignal]:
        """
        Generate signals for all symbols
        """
        signals = []
        
        for symbol, timeframe_data in all_data.items():
            try:
                signal = self.generate_signal(symbol, timeframe_data, account_balance)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # Take top N to avoid overtrading
        max_signals = 5
        return signals[:max_signals]


# Convenience function
def get_signal(
    symbol: str,
    df: pd.DataFrame,
    account: float = 10000
) -> Optional[TradingSignal]:
    """Quick signal generation"""
    generator = SignalGenerator()
    return generator.generate_signal(symbol, {'1h': df}, account)

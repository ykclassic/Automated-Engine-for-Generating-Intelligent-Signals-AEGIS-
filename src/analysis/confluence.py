"""
AEGIS Confluence Scoring System
Multi-indicator consensus with weighted scoring
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    NEUTRAL = 0
    BULLISH = 1
    BEARISH = -1


@dataclass
class IndicatorSignal:
    """Individual indicator signal"""
    name: str
    category: str
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    message: str


@dataclass
class ConfluenceScore:
    """Final confluence calculation"""
    overall_direction: SignalDirection
    overall_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    signals: List[IndicatorSignal]
    dominant_category: str
    agreement_ratio: float


class ConfluenceEngine:
    """
    Calculates weighted confluence scores across multiple indicators
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        
        # Default weights (can be customized)
        self.category_weights = {
            'trend': 0.30,
            'momentum': 0.25,
            'volatility': 0.15,
            'volume': 0.20,
            'regime': 0.10
        }
        
        # Minimum signals required for valid confluence
        self.min_signals = 3
        self.min_confidence = 0.6
    
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def calculate_trend_signals(self, df: pd.DataFrame) -> List[IndicatorSignal]:
        """
        Extract trend signals from dataframe
        """
        signals = []
        latest = df.iloc[-1]
        
        # EMA alignment
        if 'ema_8' in latest and 'ema_21' in latest:
            if latest['ema_8'] > latest['ema_21']:
                strength = min(abs(latest['ema_8'] / latest['ema_21'] - 1) * 100, 1.0)
                signals.append(IndicatorSignal(
                    name='EMA_8_21',
                    category='trend',
                    direction=SignalDirection.BULLISH,
                    strength=strength,
                    confidence=0.8,
                    message=f"EMA8 ({latest['ema_8']:.2f}) > EMA21 ({latest['ema_21']:.2f})"
                ))
            else:
                strength = min(abs(latest['ema_21'] / latest['ema_8'] - 1) * 100, 1.0)
                signals.append(IndicatorSignal(
                    name='EMA_8_21',
                    category='trend',
                    direction=SignalDirection.BEARISH,
                    strength=strength,
                    confidence=0.8,
                    message=f"EMA8 ({latest['ema_8']:.2f}) < EMA21 ({latest['ema_21']:.2f})"
                ))
        
        # MACD
        if 'macd_line' in latest and 'macd_signal' in latest:
            if latest['macd_line'] > latest['macd_signal']:
                hist = abs(latest.get('macd_histogram', 0))
                strength = min(hist / abs(latest['macd_signal']) if latest['macd_signal'] != 0 else 0.5, 1.0)
                signals.append(IndicatorSignal(
                    name='MACD',
                    category='trend',
                    direction=SignalDirection.BULLISH,
                    strength=strength,
                    confidence=0.75,
                    message="MACD above signal line"
                ))
            else:
                signals.append(IndicatorSignal(
                    name='MACD',
                    category='trend',
                    direction=SignalDirection.BEARISH,
                    strength=0.5,
                    confidence=0.75,
                    message="MACD below signal line"
                ))
        
        # ADX trend strength
        if 'adx' in latest:
            if latest['adx'] > 50:
                signals.append(IndicatorSignal(
                    name='ADX_Strong',
                    category='trend',
                    direction=SignalDirection.BULLISH if latest.get('plus_di', 0) > latest.get('minus_di', 0) else SignalDirection.BEARISH,
                    strength=latest['adx'] / 100,
                    confidence=0.9,
                    message=f"Strong trend (ADX: {latest['adx']:.1f})"
                ))
        
        # Ichimoku Cloud
        if 'price_above_cloud' in latest:
            if latest['price_above_cloud']:
                signals.append(IndicatorSignal(
                    name='Ichimoku',
                    category='trend',
                    direction=SignalDirection.BULLISH,
                    strength=0.8,
                    confidence=0.85,
                    message="Price above Ichimoku cloud"
                ))
            elif latest.get('price_below_cloud', 0):
                signals.append(IndicatorSignal(
                    name='Ichimoku',
                    category='trend',
                    direction=SignalDirection.BEARISH,
                    strength=0.8,
                    confidence=0.85,
                    message="Price below Ichimoku cloud"
                ))
        
        # SuperTrend
        if 'supertrend_direction' in latest:
            direction = SignalDirection.BULLISH if latest['supertrend_direction'] == 1 else SignalDirection.BEARISH
            signals.append(IndicatorSignal(
                name='SuperTrend',
                category='trend',
                direction=direction,
                strength=0.7,
                confidence=0.8,
                message=f"SuperTrend {'bullish' if direction == SignalDirection.BULLISH else 'bearish'}"
            ))
        
        return signals
    
    def calculate_momentum_signals(self, df: pd.DataFrame) -> List[IndicatorSignal]:
        """
        Extract momentum signals
        """
        signals = []
        latest = df.iloc[-1]
        
        # RSI
        if 'rsi' in latest:
            rsi = latest['rsi']
            if rsi < 30:
                signals.append(IndicatorSignal(
                    name='RSI',
                    category='momentum',
                    direction=SignalDirection.BULLISH,
                    strength=(30 - rsi) / 30,
                    confidence=0.8,
                    message=f"RSI oversold ({rsi:.1f})"
                ))
            elif rsi > 70:
                signals.append(IndicatorSignal(
                    name='RSI',
                    category='momentum',
                    direction=SignalDirection.BEARISH,
                    strength=(rsi - 70) / 30,
                    confidence=0.8,
                    message=f"RSI overbought ({rsi:.1f})"
                ))
            else:
                # Neutral zone - check slope
                if 'rsi_slope' in latest and latest['rsi_slope'] > 0:
                    signals.append(IndicatorSignal(
                        name='RSI',
                        category='momentum',
                        direction=SignalDirection.BULLISH,
                        strength=0.3,
                        confidence=0.6,
                        message=f"RSI rising ({rsi:.1f})"
                    ))
                elif 'rsi_slope' in latest and latest['rsi_slope'] < 0:
                    signals.append(IndicatorSignal(
                        name='RSI',
                        category='momentum',
                        direction=SignalDirection.BEARISH,
                        strength=0.3,
                        confidence=0.6,
                        message=f"RSI falling ({rsi:.1f})"
                    ))
        
        # RSI Divergence
        if 'rsi_divergence' in latest:
            div = latest['rsi_divergence']
            if div == 1:
                signals.append(IndicatorSignal(
                    name='RSI_Divergence',
                    category='momentum',
                    direction=SignalDirection.BULLISH,
                    strength=0.9,
                    confidence=0.85,
                    message="Bullish RSI divergence detected"
                ))
            elif div == -1:
                signals.append(IndicatorSignal(
                    name='RSI_Divergence',
                    category='momentum',
                    direction=SignalDirection.BEARISH,
                    strength=0.9,
                    confidence=0.85,
                    message="Bearish RSI divergence detected"
                ))
        
        # Stochastic
        if 'stoch_k' in latest and 'stoch_d' in latest:
            k, d = latest['stoch_k'], latest['stoch_d']
            if k < 20 and d < 20:
                signals.append(IndicatorSignal(
                    name='Stochastic',
                    category='momentum',
                    direction=SignalDirection.BULLISH,
                    strength=(20 - k) / 20,
                    confidence=0.75,
                    message=f"Stochastic oversold (K:{k:.1f}, D:{d:.1f})"
                ))
            elif k > 80 and d > 80:
                signals.append(IndicatorSignal(
                    name='Stochastic',
                    category='momentum',
                    direction=SignalDirection.BEARISH,
                    strength=(k - 80) / 20,
                    confidence=0.75,
                    message=f"Stochastic overbought (K:{k:.1f}, D:{d:.1f})"
                ))
        
        # MACD Histogram
        if 'macd_histogram' in latest:
            hist = latest['macd_histogram']
            prev_hist = df['macd_histogram'].iloc[-2] if len(df) > 1 else 0
            
            if hist > 0 and hist > prev_hist:
                signals.append(IndicatorSignal(
                    name='MACD_Histogram',
                    category='momentum',
                    direction=SignalDirection.BULLISH,
                    strength=min(abs(hist) / 10, 1.0),
                    confidence=0.7,
                    message="MACD histogram increasing"
                ))
            elif hist < 0 and hist < prev_hist:
                signals.append(IndicatorSignal(
                    name='MACD_Histogram',
                    category='momentum',
                    direction=SignalDirection.BEARISH,
                    strength=min(abs(hist) / 10, 1.0),
                    confidence=0.7,
                    message="MACD histogram decreasing"
                ))
        
        return signals
    
    def calculate_volatility_signals(self, df: pd.DataFrame) -> List[IndicatorSignal]:
        """
        Extract volatility-based signals
        """
        signals = []
        latest = df.iloc[-1]
        
        # Bollinger Bands
        if 'bb_percent_b' in latest:
            percent_b = latest['bb_percent_b']
            if percent_b < 0.1:
                signals.append(IndicatorSignal(
                    name='Bollinger',
                    category='volatility',
                    direction=SignalDirection.BULLISH,
                    strength=0.8,
                    confidence=0.75,
                    message=f"Price at lower band ({percent_b:.2f})"
                ))
            elif percent_b > 0.9:
                signals.append(IndicatorSignal(
                    name='Bollinger',
                    category='volatility',
                    direction=SignalDirection.BEARISH,
                    strength=0.8,
                    confidence=0.75,
                    message=f"Price at upper band ({percent_b:.2f})"
                ))
        
        # Bollinger Squeeze
        if 'bb_squeeze' in latest and latest['bb_squeeze']:
            signals.append(IndicatorSignal(
                name='BB_Squeeze',
                category='volatility',
                direction=SignalDirection.NEUTRAL,
                strength=0.5,
                confidence=0.8,
                message="Bollinger Squeeze - volatility expansion likely"
            ))
        
        # ATR-based volatility regime
        if 'atr_normalized' in latest:
            atr_norm = latest['atr_normalized']
            if atr_norm > 1.5:
                signals.append(IndicatorSignal(
                    name='ATR',
                    category='volatility',
                    direction=SignalDirection.NEUTRAL,
                    strength=min(atr_norm / 3, 1.0),
                    confidence=0.7,
                    message=f"High volatility regime (ATR: {atr_norm:.2f})"
                ))
            elif atr_norm < 0.6:
                signals.append(IndicatorSignal(
                    name='ATR',
                    category='volatility',
                    direction=SignalDirection.NEUTRAL,
                    strength=0.3,
                    confidence=0.7,
                    message=f"Low volatility regime (ATR: {atr_norm:.2f})"
                ))
        
        return signals
    
    def calculate_volume_signals(self, df: pd.DataFrame) -> List[IndicatorSignal]:
        """
        Extract volume-based signals
        """
        signals = []
        latest = df.iloc[-1]
        
        # Volume confirmation
        if 'volume_price_confirmation' in latest:
            conf = latest['volume_price_confirmation']
            if conf == 1:
                signals.append(IndicatorSignal(
                    name='Volume_Confirmation',
                    category='volume',
                    direction=SignalDirection.BULLISH,
                    strength=min(latest.get('relative_volume', 1) / 3, 1.0),
                    confidence=0.8,
                    message="Volume confirms price increase"
                ))
            elif conf == -1:
                signals.append(IndicatorSignal(
                    name='Volume_Confirmation',
                    category='volume',
                    direction=SignalDirection.BEARISH,
                    strength=min(latest.get('relative_volume', 1) / 3, 1.0),
                    confidence=0.8,
                    message="Volume confirms price decrease"
                ))
        
        # OBV trend
        if 'obv_trend' in latest:
            if latest['obv_trend'] == 1:
                signals.append(IndicatorSignal(
                    name='OBV',
                    category='volume',
                    direction=SignalDirection.BULLISH,
                    strength=0.6,
                    confidence=0.7,
                    message="OBV trending up"
                ))
            else:
                signals.append(IndicatorSignal(
                    name='OBV',
                    category='volume',
                    direction=SignalDirection.BEARISH,
                    strength=0.6,
                    confidence=0.7,
                    message="OBV trending down"
                ))
        
        # VWAP position
        if 'price_vs_vwap' in latest:
            vwap_pos = latest['price_vs_vwap']
            if vwap_pos > 0.01:  # 1% above VWAP
                signals.append(IndicatorSignal(
                    name='VWAP',
                    category='volume',
                    direction=SignalDirection.BULLISH,
                    strength=min(vwap_pos * 10, 1.0),
                    confidence=0.75,
                    message=f"Price {vwap_pos*100:.1f}% above VWAP"
                ))
            elif vwap_pos < -0.01:
                signals.append(IndicatorSignal(
                    name='VWAP',
                    category='volume',
                    direction=SignalDirection.BEARISH,
                    strength=min(abs(vwap_pos) * 10, 1.0),
                    confidence=0.75,
                    message=f"Price {abs(vwap_pos)*100:.1f}% below VWAP"
                ))
        
        # MFI
        if 'mfi' in latest:
            mfi = latest['mfi']
            if mfi < 20:
                signals.append(IndicatorSignal(
                    name='MFI',
                    category='volume',
                    direction=SignalDirection.BULLISH,
                    strength=(20 - mfi) / 20,
                    confidence=0.75,
                    message=f"MFI oversold ({mfi:.1f})"
                ))
            elif mfi > 80:
                signals.append(IndicatorSignal(
                    name='MFI',
                    category='volume',
                    direction=SignalDirection.BEARISH,
                    strength=(mfi - 80) / 20,
                    confidence=0.75,
                    message=f"MFI overbought ({mfi:.1f})"
                ))
        
        return signals
    
    def calculate_regime_signals(self, df: pd.DataFrame) -> List[IndicatorSignal]:
        """
        Extract market regime signals
        """
        signals = []
        latest = df.iloc[-1]
        
        if 'market_regime' in latest:
            regime = latest['market_regime']
            
            # Only generate signals for clear regimes
            if regime == 'strong_uptrend':
                signals.append(IndicatorSignal(
                    name='Regime',
                    category='regime',
                    direction=SignalDirection.BULLISH,
                    strength=0.9,
                    confidence=0.9,
                    message="Strong uptrend regime"
                ))
            elif regime == 'strong_downtrend':
                signals.append(IndicatorSignal(
                    name='Regime',
                    category='regime',
                    direction=SignalDirection.BEARISH,
                    strength=0.9,
                    confidence=0.9,
                    message="Strong downtrend regime"
                ))
            elif regime == 'ranging':
                signals.append(IndicatorSignal(
                    name='Regime',
                    category='regime',
                    direction=SignalDirection.NEUTRAL,
                    strength=0.5,
                    confidence=0.8,
                    message="Ranging market - reduce position size"
                ))
            elif regime == 'high_volatility':
                signals.append(IndicatorSignal(
                    name='Regime',
                    category='regime',
                    direction=SignalDirection.NEUTRAL,
                    strength=0.7,
                    confidence=0.7,
                    message="High volatility - caution advised"
                ))
        
        return signals
    
    def calculate_confluence(self, df: pd.DataFrame) -> ConfluenceScore:
        """
        Calculate overall confluence score from all indicators
        """
        # Collect all signals
        all_signals = []
        all_signals.extend(self.calculate_trend_signals(df))
        all_signals.extend(self.calculate_momentum_signals(df))
        all_signals.extend(self.calculate_volatility_signals(df))
        all_signals.extend(self.calculate_volume_signals(df))
        all_signals.extend(self.calculate_regime_signals(df))
        
        if len(all_signals) < self.min_signals:
            return ConfluenceScore(
                overall_direction=SignalDirection.NEUTRAL,
                overall_score=0.0,
                confidence=0.0,
                signals=all_signals,
                dominant_category='none',
                agreement_ratio=0.0
            )
        
        # Calculate weighted scores by category
        category_scores = {}
        category_confidences = {}
        
        for category, weight in self.category_weights.items():
            cat_signals = [s for s in all_signals if s.category == category]
            if not cat_signals:
                continue
            
            # Calculate category direction and strength
            bullish_score = sum(
                s.strength * s.confidence 
                for s in cat_signals 
                if s.direction == SignalDirection.BULLISH
            )
            bearish

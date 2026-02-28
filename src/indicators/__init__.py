"""
AEGIS Indicators Module
Centralized indicator calculation and management
"""

from .trend import TrendIndicators
from .momentum import MomentumIndicators
from .volatility import VolatilityIndicators
from .volume import VolumeIndicators

__all__ = [
    'TrendIndicators',
    'MomentumIndicators',
    'VolatilityIndicators',
    'VolumeIndicators',
    'IndicatorOrchestrator'
]


class IndicatorOrchestrator:
    """
    Centralized indicator calculation manager
    Ensures proper calculation order and dependencies
    """
    
    def __init__(self):
        self.trend = TrendIndicators()
        self.momentum = MomentumIndicators()
        self.volatility = VolatilityIndicators()
        self.volume = VolumeIndicators()
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators in proper order:
        1. Trend (base moving averages)
        2. Volatility (needs trend indicators)
        3. Momentum (needs volatility for normalization)
        4. Volume (independent but uses price)
        """
        df = df.copy()
        
        # Step 1: Trend indicators (foundational)
        df = self.trend.calculate_all(df)
        
        # Step 2: Volatility (uses price and trend)
        df = self.volatility.calculate_all(df)
        
        # Step 3: Momentum (uses trend and volatility info)
        df = self.momentum.calculate_all(df)
        
        # Step 4: Volume indicators
        df = self.volume.calculate_all(df)
        
        return df
    
    def get_indicator_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary of current indicator values
        """
        latest = df.iloc[-1]
        
        return {
            'trend': {
                'ema_8_21': 'bullish' if latest.get('ema_8', 0) > latest.get('ema_21', 0) else 'bearish',
                'macd': 'bullish' if latest.get('macd_line', 0) > latest.get('macd_signal', 0) else 'bearish',
                'adx': f"{latest.get('adx', 0):.1f}",
                'supertrend': 'bullish' if latest.get('supertrend_direction', 0) == 1 else 'bearish'
            },
            'momentum': {
                'rsi': f"{latest.get('rsi', 0):.1f}",
                'stoch_k': f"{latest.get('stoch_k', 0):.1f}",
                'momentum_score': f"{latest.get('momentum_score', 0):.2f}"
            },
            'volatility': {
                'atr_pct': f"{latest.get('atr_pct', 0):.2%}",
                'bb_position': f"{latest.get('bb_percent_b', 0):.2f}",
                'regime': latest.get('volatility_regime', 'unknown')
            },
            'volume': {
                'relative_volume': f"{latest.get('relative_volume', 0):.2f}",
                'obv_trend': 'up' if latest.get('obv_trend', 0) == 1 else 'down',
                'vwap_position': 'above' if latest.get('above_vwap', 0) == 1 else 'below'
            }
        }

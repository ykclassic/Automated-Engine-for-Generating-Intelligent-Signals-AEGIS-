"""
AEGIS Multi-Timeframe Analysis Module
Aligns and correlates signals across timeframes
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MultiTimeframeAnalyzer:
    """
    Analyzes confluence across multiple timeframes
    Higher timeframes provide bias, lower timeframes provide entry timing
    """
    
    def __init__(self):
        self.timeframe_weights = {
            '1w': 0.35,  # Macro trend (highest weight)
            '1d': 0.30,  # Daily trend
            '4h': 0.20,  # Intermediate
            '1h': 0.15   # Entry timing (lowest weight)
        }
    
    def align_timeframes(
        self,
        timeframe_data: Dict[str, pd.DataFrame],
        target_timeframe: str = '1h'
    ) -> pd.DataFrame:
        """
        Align all timeframe data to target timeframe (usually lowest)
        Uses forward-fill to ensure no lookahead bias
        """
        if target_timeframe not in timeframe_data:
            raise ValueError(f"Target timeframe {target_timeframe} not in data")
        
        base_df = timeframe_data[target_timeframe].copy()
        
        for tf, df in timeframe_data.items():
            if tf == target_timeframe:
                continue
            
            # Resample higher timeframe data to base timeframe
            # Use asof merge to get last known value (causal)
            df_resampled = df.resample(target_timeframe).last().ffill()
            
            # Select key confluence features to propagate
            key_features = [
                'confluence_score',
                'market_regime',
                'trend_bullish_score',
                'rsi',
                'adx',
                'ema_50_200_ratio'
            ]
            
            for feat in key_features:
                if feat in df.columns:
                    # Reindex to base timeframe and forward fill
                    aligned = df[feat].reindex(base_df.index, method='ffill')
                    base_df[f'{feat}_{tf}'] = aligned
        
        return base_df
    
    def calculate_timeframe_confluence(
        self,
        timeframe_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, any]:
        """
        Calculate confluence score across all timeframes
        """
        scores = {}
        
        for tf, df in timeframe_data.items():
            if 'confluence_score' in df.columns:
                latest = df['confluence_score'].iloc[-1]
                scores[tf] = {
                    'score': latest,
                    'direction': 'bullish' if latest > 0.2 else ('bearish' if latest < -0.2 else 'neutral'),
                    'weight': self.timeframe_weights.get(tf, 0.1)
                }
        
        # Calculate weighted confluence
        weighted_score = sum(
            s['score'] * s['weight'] 
            for s in scores.values()
        )
        
        # Check alignment (all timeframes agree)
        directions = [s['direction'] for s in scores.values()]
        alignment = len(set(directions)) == 1 and directions[0] != 'neutral'
        
        return {
            'timeframe_scores': scores,
            'weighted_score': weighted_score,
            'overall_direction': 'bullish' if weighted_score > 0.2 else ('bearish' if weighted_score < -0.2 else 'neutral'),
            'alignment': alignment,
            'confidence': abs(weighted_score) * (1.5 if alignment else 1.0)
        }
    
    def find_confluence_zones(
        self,
        timeframe_data: Dict[str, pd.DataFrame]
    ) -> List[Dict]:
        """
        Find price zones where multiple timeframes show support/resistance
        """
        zones = []
        
        # Use daily and 4h for zone detection
        for tf in ['1d', '4h']:
            if tf not in timeframe_data:
                continue
            
            df = timeframe_data[tf]
            
            # Find pivot points (simplified)
            highs = df['high'].rolling(window=5, center=True).max()
            lows = df['low'].rolling(window=5, center=True).min()
            
            resistance = df[df['high'] == highs]['high'].iloc[-3:].tolist()
            support = df[df['low'] == lows]['low'].iloc[-3:].tolist()
            
            for level in resistance:
                zones.append({
                    'price': level,
                    'type': 'resistance',
                    'timeframe': tf,
                    'strength': 2 if tf == '1d' else 1
                })
            
            for level in support:
                zones.append({
                    'price': level,
                    'type': 'support',
                    'timeframe': tf,
                    'strength': 2 if tf == '1d' else 1
                })
        
        # Sort by strength and proximity to current price
        current_price = timeframe_data['1h']['close'].iloc[-1]
        zones.sort(key=lambda x: (abs(x['price'] - current_price), -x['strength']))
        
        return zones[:5]  # Return top 5 zones
    
    def generate_multi_timeframe_signal(
        self,
        timeframe_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Generate final trading signal with multi-timeframe context
        """
        # Align data
        aligned = self.align_timeframes(timeframe_data)
        
        # Calculate timeframe confluence
        tf_confluence = self.calculate_timeframe_confluence(timeframe_data)
        
        # Get current price action
        current = aligned.iloc[-1]
        price = current['close']
        
        # Determine entry type based on alignment
        if tf_confluence['alignment']:
            if tf_confluence['overall_direction'] == 'bullish':
                entry_type = 'strong_long'
                stop_loss = current.get('supertrend', price * 0.97) if 'supertrend' in current else price * 0.95
            else:
                entry_type = 'strong_short'
                stop_loss = current.get('supertrend', price * 1.03) if 'supertrend' in current else price * 1.05
        else:
            # Mixed signals - wait for alignment
            entry_type = 'neutral'
            stop_loss = None
        
        # Find nearest support/resistance
        zones = self.find_confluence_zones(timeframe_data)
        nearest_support = next((z['price'] for z in zones if z['type'] == 'support'), price * 0.95)
        nearest_resistance = next((z['price'] for z in zones if z['type'] == 'resistance'), price * 1.05)
        
        return {
            'signal_type': entry_type,
            'direction': tf_confluence['overall_direction'],
            'confidence': min(tf_confluence['confidence'], 1.0),
            'alignment': tf_confluence['alignment'],
            'timeframe_breakdown': tf_confluence['timeframe_scores'],
            'entry_price': price,
            'stop_loss': stop_loss,
            'take_profit': nearest_resistance if entry_type == 'strong_long' else nearest_support,
            'support_levels': [z for z in zones if z['type'] == 'support'][:2],
            'resistance_levels': [z for z in zones if z['type'] == 'resistance'][:2],
            'risk_reward': abs(nearest_resistance - price) / abs(price - stop_loss) if stop_loss and entry_type == 'strong_long' else None
        }


# Convenience function
def analyze_timeframes(timeframe_data: Dict[str, pd.DataFrame]) -> Dict:
    """Quick multi-timeframe analysis"""
    analyzer = MultiTimeframeAnalyzer()
    return analyzer.generate_multi_timeframe_signal(timeframe_data)

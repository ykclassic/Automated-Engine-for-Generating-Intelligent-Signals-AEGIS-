"""
AEGIS Notification Formatter
Rich signal formatting for Discord/webhooks
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd

from ..core.signal_generator import TradingSignal

logger = logging.getLogger(__name__)


class SignalFormatter:
    """
    Formats trading signals for notifications
    """
    
    def __init__(self):
        self.max_signals_per_message = 5
    
    def format_signal_embed(self, signal: TradingSignal) -> Dict:
        """
        Format single signal as Discord embed
        """
        # Color based on direction
        color_map = {
            'long': 3066993,   # Green
            'short': 15158332, # Red
            'neutral': 9807270 # Gray
        }
        color = color_map.get(signal.direction, 9807270)
        
        # Emoji based on confidence
        confidence_emoji = {
            'very_high': '🔥',
            'high': '✅',
            'moderate': '⚠️',
            'low': '❓'
        }
        emoji = confidence_emoji.get(signal.confidence, '')
        
        # Build fields
        fields = [
            {
                'name': '📊 Direction',
                'value': f"{emoji} {signal.direction.upper()}",
                'inline': True
            },
            {
                'name': '🎯 Confidence',
                'value': f"{signal.confidence} ({signal.confidence_score:.1%})",
                'inline': True
            },
            {
                'name': '💰 Entry Price',
                'value': f"${signal.entry_price:,.2f}",
                'inline': True
            },
            {
                'name': '🛑 Stop Loss',
                'value': f"${signal.stop_loss:,.2f} ({abs(signal.stop_loss/signal.entry_price-1)*100:.1f}%)",
                'inline': True
            },
            {
                'name': '🎯 Take Profit',
                'value': f"${signal.take_profit:,.2f} ({abs(signal.take_profit/signal.entry_price-1)*100:.1f}%)",
                'inline': True
            },
            {
                'name': '📈 R:R Ratio',
                'value': f"{abs(signal.take_profit - signal.entry_price) / abs(signal.entry_price - signal.stop_loss):.2f}:1",
                'inline': True
            }
        ]
        
        # Position sizing
        fields.append({
            'name': '💼 Position Size',
            'value': (
                f"Size: {signal.position_size['size_pct']:.1%} of portfolio\n"
                f"Leverage: {signal.position_size['leverage']:.1f}x\n"
                f"Max Loss: {signal.position_size['max_loss_pct']:.2%}"
            ),
            'inline': False
        })
        
        # Indicator summary
        if signal.timeframe_confluence:
            confluence = signal.timeframe_confluence
            fields.append({
                'name': '📊 Confluence Analysis',
                'value': (
                    f"Score: {confluence['score']:+.2f}\n"
                    f"Dominant: {confluence['dominant_category']}\n"
                    f"Agreement: {confluence['agreement_ratio']:.1%}"
                ),
                'inline': True
            })
        
        # Risk metrics
        if signal.risk_metrics:
            risk = signal.risk_metrics
            fields.append({
                'name': '⚠️ Risk Metrics',
                'value': (
                    f"Portfolio Heat: {risk['portfolio_heat']:.1%}\n"
                    f"Daily VaR: {risk['daily_var']:.2%}\n"
                    f"Drawdown: {risk['current_drawdown']:.2%}"
                ),
                'inline': True
            })
        
        # ML prediction
        if signal.ml_prediction:
            ml = signal.ml_prediction
            fields.append({
                'name': '🤖 ML Prediction',
                'value': (
                    f"Direction: {ml['direction']}\n"
                    f"Confidence: {ml.get('confidence', 0):.1%}\n"
                    f"Level: {ml.get('confidence_level', 'unknown')}"
                ),
                'inline': False
            })
        
        embed = {
            'title': f'{emoji} {signal.symbol} Signal Generated',
            'color': color,
            'timestamp': signal.timestamp.isoformat(),
            'fields': fields,
            'footer': {
                'text': f'AEGIS Signal Generator • Expires: {signal.expiration.strftime("%H:%M UTC") if signal.expiration else "N/A"}'
            }
        }
        
        return embed
    
    def format_summary_embed(
        self,
        signals: List[TradingSignal],
        market_overview: Optional[Dict] = None
    ) -> Dict:
        """
        Format summary of multiple signals
        """
        if not signals:
            return {
                'title': '📊 AEGIS Market Scan Complete',
                'description': 'No high-quality signals found in current scan.',
                'color': 9807270,
                'timestamp': datetime.now().isoformat()
            }
        
        # Count by direction
        long_count = sum(1 for s in signals if s.direction == 'long')
        short_count = sum(1 for s in signals if s.direction == 'short')
        
        description = f"Found {len(signals)} tradeable signals:\n"
        description += f"🟢 Long: {long_count} | 🔴 Short: {short_count}\n\n"
        
        # List top signals
        for i, signal in enumerate(signals[:self.max_signals_per_message], 1):
            emoji = '🔥' if signal.confidence == 'very_high' else '✅'
            description += (
                f"{i}. {emoji} **{signal.symbol}** | "
                f"{signal.direction.upper()} | "
                f"Conf: {signal.confidence_score:.0%} | "
                f"R:R {abs(signal.take_profit - signal.entry_price) / abs(signal.entry_price - signal.stop_loss):.1f}:1\n"
            )
        
        fields = []
        
        # Market overview
        if market_overview:
            fields.append({
                'name': '🌍 Market Overview',
                'value': (
                    f"Regime: {market_overview.get('dominant_regime', 'unknown')}\n"
                    f"Volatility: {market_overview.get('avg_volatility', 'normal')}\n"
                    f"Correlation: {market_overview.get('correlation_state', 'normal')}"
                ),
                'inline': False
            })
        
        embed = {
            'title': '🛡️ AEGIS Signal Summary',
            'description': description,
            'color': 3447003,
            'fields': fields,
            'timestamp': datetime.now().isoformat(),
            'footer': {
                'text': f'AEGIS v1.0 • {len(signals)} signals generated'
            }
        }
        
        return embed
    
    def format_alert_embed(
        self,
        alert_type: str,
        message: str,
        details: Optional[Dict] = None
    ) -> Dict:
        """
        Format system alerts (circuit breaker, errors, etc.)
        """
        color_map = {
            'error': 15158332,    # Red
            'warning': 16776960,  # Yellow
            'info': 3447003,      # Blue
            'success': 3066993    # Green
        }
        
        emoji_map = {
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️',
            'success': '✅'
        }
        
        embed = {
            'title': f"{emoji_map.get(alert_type, 'ℹ️')} System Alert",
            'description': message,
            'color': color_map.get(alert_type, 3447003),
            'timestamp': datetime.now().isoformat()
        }
        
        if details:
            fields = []
            for key, value in details.items():
                fields.append({
                    'name': key.replace('_', ' ').title(),
                    'value': str(value),
                    'inline': True
                })
            embed['fields'] = fields
        
        return embed
    
    def format_performance_report(
        self,
        performance: Dict,
        period: str = "24h"
    ) -> Dict:
        """
        Format performance report
        """
        fields = [
            {
                'name': '💰 Total Return',
                'value': f"{performance.get('total_return', 0):.2%}",
                'inline': True
            },
            {
                'name': '📈 Win Rate',
                'value': f"{performance.get('win_rate', 0):.1%}",
                'inline': True
            },
            {
                'name': '⚖️ Profit Factor',
                'value': f"{performance.get('profit_factor', 0):.2f}",
                'inline': True
            },
            {
                'name': '📊 Sharpe Ratio',
                'value': f"{performance.get('sharpe_ratio', 0):.2f}",
                'inline': True
            },
            {
                'name': '📉 Max Drawdown',
                'value': f"{performance.get('max_drawdown', 0):.2%}",
                'inline': True
            },
            {
                'name': '🔥 Portfolio Heat',
                'value': f"{performance.get('portfolio_heat', 0):.1%}",
                'inline': True
            }
        ]
        
        embed = {
            'title': f'📊 Performance Report ({period})',
            'color': 3066993 if performance.get('total_return', 0) >= 0 else 15158332,
            'fields': fields,
            'timestamp': datetime.now().isoformat()
        }
        
        return embed


# Convenience function
def format_signal(signal: TradingSignal) -> Dict:
    """Quick signal formatting"""
    formatter = SignalFormatter()
    return formatter.format_signal_embed(signal)

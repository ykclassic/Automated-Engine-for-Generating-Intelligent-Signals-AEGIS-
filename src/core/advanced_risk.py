"""
AEGIS Advanced Risk Controls
Correlation filters, volatility targeting, and dynamic hedging
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class CorrelationMatrix:
    """Correlation matrix with timestamps"""
    matrix: pd.DataFrame
    timestamp: pd.Timestamp
    lookback: int


class CorrelationRiskManager:
    """
    Manages correlation risk across positions
    """
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.correlation_history: List[CorrelationMatrix] = []
        self.max_correlation = 0.7  # Don't add if correlation > 0.7
    
    def calculate_correlation(
        self,
        returns_dict: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix from returns
        """
        # Align all series
        df = pd.DataFrame(returns_dict)
        
        # Calculate correlation
        corr = df.corr()
        
        # Store history
        self.correlation_history.append(CorrelationMatrix(
            matrix=corr,
            timestamp=pd.Timestamp.now(),
            lookback=self.lookback
        ))
        
        # Keep only recent history
        if len(self.correlation_history) > 100:
            self.correlation_history = self.correlation_history[-100:]
        
        return corr
    
    def check_correlation_risk(
        self,
        new_symbol: str,
        existing_symbols: List[str],
        correlation_matrix: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        Check if new symbol is too correlated with existing positions
        """
        if new_symbol not in correlation_matrix.columns:
            return True, "No correlation data"
        
        for existing in existing_symbols:
            if existing not in correlation_matrix.columns:
                continue
            
            corr = correlation_matrix.loc[new_symbol, existing]
            
            if abs(corr) > self.max_correlation:
                return False, f"Correlation {corr:.2f} with {existing} exceeds {self.max_correlation}"
        
        return True, "Correlation risk acceptable"
    
    def get_diversification_score(
        self,
        symbols: List[str],
        weights: List[float],
        correlation_matrix: pd.DataFrame
    ) -> float:
        """
        Calculate portfolio diversification score
        Higher is better (less correlated)
        """
        if len(symbols) < 2:
            return 1.0
        
        # Portfolio variance calculation
        # σ²_p = w'Σw
        # where w is weight vector, Σ is covariance matrix
        
        # Extract sub-matrix
        sub_corr = correlation_matrix.loc[symbols, symbols]
        
        # Calculate portfolio correlation (simplified)
        # Average pairwise correlation weighted by position sizes
        total_corr = 0
        count = 0
        
        for i, s1 in enumerate(symbols):
            for j, s2 in enumerate(symbols):
                if i != j:
                    total_corr += sub_corr.loc[s1, s2] * weights[i] * weights[j]
                    count += 1
        
        avg_corr = total_corr / count if count > 0 else 0
        
        # Diversification score: 1 - avg_correlation
        diversification = 1 - abs(avg_corr)
        
        return diversification


class VolatilityTargeter:
    """
    Dynamic volatility targeting for position sizing
    """
    
    def __init__(
        self,
        target_volatility: float = 0.15,  # 15% annualized
        max_leverage: float = 2.0
    ):
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
    
    def calculate_volatility_adjustment(
        self,
        historical_returns: pd.Series,
        lookback: int = 20
    ) -> float:
        """
        Calculate position size adjustment based on current volatility
        """
        if len(historical_returns) < lookback:
            return 1.0
        
        # Calculate realized volatility (annualized)
        current_vol = historical_returns.tail(lookback).std() * np.sqrt(365 * 24)
        
        if current_vol == 0:
            return 1.0
        
        # Volatility scaling factor
        # If vol is high, reduce size; if low, increase size
        vol_ratio = self.target_volatility / current_vol
        
        # Cap leverage
        adjustment = min(vol_ratio, self.max_leverage)
        
        return adjustment
    
    def apply_volatility_targeting(
        self,
        base_position_size: float,
        returns: pd.Series
    ) -> float:
        """
        Apply volatility targeting to position size
        """
        adjustment = self.calculate_volatility_adjustment(returns)
        adjusted_size = base_position_size * adjustment
        
        return adjusted_size


class DynamicStopLoss:
    """
    Dynamic stop loss adjustment based on volatility
    """
    
    def __init__(self):
        self.atr_multiplier = 2.0
        self.min_stop_pct = 0.005  # 0.5%
        self.max_stop_pct = 0.05   # 5%
    
    def calculate_stop(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        volatility_regime: str = 'normal'
    ) -> float:
        """
        Calculate dynamic stop loss
        """
        # Base stop on ATR
        stop_distance = atr * self.atr_multiplier
        
        # Adjust for volatility regime
        regime_multipliers = {
            'very_low': 1.5,
            'low': 1.2,
            'normal': 2.0,
            'high': 2.5,
            'extreme': 3.0
        }
        
        multiplier = regime_multipliers.get(volatility_regime, 2.0)
        stop_distance *= multiplier
        
        # Calculate stop price
        if direction == 'long':
            stop_price = entry_price - stop_distance
            # Ensure minimum stop distance
            min_stop = entry_price * (1 - self.max_stop_pct)
            stop_price = max(stop_price, min_stop)
        else:
            stop_price = entry_price + stop_distance
            max_stop = entry_price * (1 + self.max_stop_pct)
            stop_price = min(stop_price, max_stop)
        
        return stop_price
    
    def trailing_stop(
        self,
        current_price: float,
        highest_price: float,
        lowest_price: float,
        direction: str,
        atr: float,
        trailing_pct: float = 0.02
    ) -> float:
        """
        Calculate trailing stop loss
        """
        if direction == 'long':
            # Trail below highest price
            trail_distance = max(atr * 1.5, highest_price * trailing_pct)
            return highest_price - trail_distance
        else:
            # Trail above lowest price
            trail_distance = max(atr * 1.5, lowest_price * trailing_pct)
            return lowest_price + trail_distance


class RiskOfRuinCalculator:
    """
    Calculate risk of ruin using various methods
    """
    
    def __init__(self):
        pass
    
    def gambler_ruin(
        self,
        win_rate: float,
        payoff_ratio: float,
        risk_per_trade: float,
        ruin_level: float = 0.5
    ) -> float:
        """
        Calculate risk of ruin using Gambler's Ruin formula
        
        Args:
            win_rate: Probability of winning
            payoff_ratio: Average win / average loss
            risk_per_trade: Risk per trade as fraction of capital
            ruin_level: Fraction of capital considered ruin
        
        Returns:
            Probability of ruin (0-1)
        """
        if win_rate <= 0 or payoff_ratio <= 0:
            return 1.0
        
        # Edge calculation
        edge = (win_rate * payoff_ratio) - (1 - win_rate)
        
        if edge <= 0:
            return 1.0  # Certain ruin with negative edge
        
        # Gambler's ruin formula for unequal probabilities
        # R = [(1 - p) / p] ^ (C / r)
        # where p = win rate, C = capital, r = risk per trade
        
        # Simplified version
        p = win_rate
        q = 1 - win_rate
        ratio = q / p
        
        # Number of trades to ruin
        n_trades = ruin_level / risk_per_trade
        
        if ratio >= 1:
            risk_of_ruin = 1.0
        else:
            risk_of_ruin = ratio ** n_trades
        
        return min(risk_of_ruin, 1.0)
    
    def monte_carlo_ruin(
        self,
        returns: pd.Series,
        n_simulations: int = 1000,
        max_trades: int = 1000,
        ruin_threshold: float = 0.5
    ) -> float:
        """
        Estimate risk of ruin using Monte Carlo simulation
        """
        if len(returns) < 10:
            return 1.0
        
        # Fit distribution to returns
        mean = returns.mean()
        std = returns.std()
        
        ruin_count = 0
        
        for _ in range(n_simulations):
            equity = 1.0
            for _ in range(max_trades):
                # Generate random return
                ret = np.random.normal(mean, std)
                equity *= (1 + ret)
                
                if equity < ruin_threshold:
                    ruin_count += 1
                    break
        
        return ruin_count / n_simulations


# Convenience functions
def calculate_correlation_risk(
    symbols: List[str],
    returns: Dict[str, pd.Series]
) -> Dict:
    """Quick correlation risk check"""
    manager = CorrelationRiskManager()
    corr_matrix = manager.calculate_correlation(returns)
    
    # Find highest correlation
    max_corr = 0
    max_pair = None
    
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i+1:]:
            if s1 in corr_matrix.columns and s2 in corr_matrix.columns:
                c = abs(corr_matrix.loc[s1, s2])
                if c > max_corr:
                    max_corr = c
                    max_pair = (s1, s2)
    
    return {
        'max_correlation': max_corr,
        'max_pair': max_pair,
        'diversification_score': 1 - max_corr
    }

def apply_volatility_target(
    base_size: float,
    returns: pd.Series,
    target_vol: float = 0.15
) -> float:
    """Quick volatility targeting"""
    targeter = VolatilityTargeter(target_volatility=target_vol)
    return targeter.apply_volatility_targeting(base_size, returns)

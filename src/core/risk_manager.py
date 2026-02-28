"""
AEGIS Risk Management Module
Institutional-grade risk controls and position sizing
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class PositionSizing:
    """Position sizing recommendation"""
    symbol: str
    direction: str
    size_pct: float  # Percentage of portfolio
    size_units: float  # Absolute units
    leverage: float
    max_loss_pct: float  # Maximum loss as % of portfolio
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float


@dataclass
class RiskMetrics:
    """Current risk metrics"""
    portfolio_heat: float  # Total risk exposure
    max_drawdown: float
    current_drawdown: float
    daily_var: float  # Value at Risk
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    risk_of_ruin: float


class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing
    """
    
    def __init__(self, fraction: float = 0.5):
        """
        Args:
            fraction: Half-Kelly (0.5) or Full-Kelly (1.0)
        """
        self.fraction = fraction
    
    def calculate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly fraction
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average win amount (as multiple of risk)
            avg_loss: Average loss amount (as multiple of risk, positive number)
        
        Returns:
            Optimal position size as fraction of portfolio
        """
        if avg_loss == 0:
            return 0
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win rate, q = loss rate, b = avg_win/avg_loss
        b = avg_win / avg_loss
        q = 1 - win_rate
        
        kelly = (win_rate * b - q) / b
        
        # Apply fraction (Half-Kelly for safety)
        kelly = kelly * self.fraction
        
        # Cap at reasonable limits
        return np.clip(kelly, 0, 0.25)  # Max 25% per position


class RiskManager:
    """
    Central risk management system
    """
    
    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        risk_level: RiskLevel = RiskLevel.MODERATE
    ):
        self.config = self._load_config(config_path)
        self.risk_level = risk_level
        
        # Risk limits based on level
        self.risk_limits = self._set_risk_limits()
        
        # Portfolio state
        self.open_positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.daily_pnl: List[float] = []
        self.peak_equity: float = 1.0
        self.current_equity: float = 1.0
        
        # Circuit breakers
        self.circuit_breaker_triggered: bool = False
        self.circuit_breaker_reason: Optional[str] = None
        self.circuit_breaker_time: Optional[datetime] = None
    
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _set_risk_limits(self) -> Dict:
        """Set risk limits based on risk level"""
        limits = {
            RiskLevel.CONSERVATIVE: {
                'max_position_pct': 0.05,      # 5% per position
                'max_portfolio_heat': 0.10,     # 10% total risk
                'max_drawdown_limit': 0.05,     # 5% max DD
                'daily_var_limit': 0.02,        # 2% daily VaR
                'max_leverage': 1.0,
                'kelly_fraction': 0.25          # Quarter Kelly
            },
            RiskLevel.MODERATE: {
                'max_position_pct': 0.10,       # 10% per position
                'max_portfolio_heat': 0.15,     # 15% total risk
                'max_drawdown_limit': 0.10,     # 10% max DD
                'daily_var_limit': 0.03,        # 3% daily VaR
                'max_leverage': 2.0,
                'kelly_fraction': 0.5           # Half Kelly
            },
            RiskLevel.AGGRESSIVE: {
                'max_position_pct': 0.20,       # 20% per position
                'max_portfolio_heat': 0.25,     # 25% total risk
                'max_drawdown_limit': 0.15,     # 15% max DD
                'daily_var_limit': 0.05,        # 5% daily VaR
                'max_leverage': 3.0,
                'kelly_fraction': 0.75          # Three-quarter Kelly
            }
        }
        return limits[self.risk_level]
    
    def calculate_position_size(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        account_balance: float,
        win_rate: float = 0.55,
        avg_win_loss_ratio: float = 1.5
    ) -> PositionSizing:
        """
        Calculate optimal position size using Kelly Criterion
        """
        # Calculate Kelly fraction
        kelly = KellyCriterion(fraction=self.risk_limits['kelly_fraction'])
        
        # Estimate from historical performance
        avg_win = avg_win_loss_ratio
        avg_loss = 1.0
        
        kelly_fraction = kelly.calculate(win_rate, avg_win, avg_loss)
        
        # Limit by max position size
        max_position_pct = self.risk_limits['max_position_pct']
        position_pct = min(kelly_fraction, max_position_pct)
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss) / entry_price
        
        # Adjust for volatility
        # Reduce size in high volatility
        # This would use ATR in real implementation
        
        # Calculate actual position size
        risk_amount = account_balance * position_pct
        position_value = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
        
        # Apply leverage
        leverage = min(self.risk_limits['max_leverage'], 2.0)  # Cap at 2x for safety
        size_units = (position_value * leverage) / entry_price
        
        # Calculate max loss
        max_loss_pct = position_pct * risk_per_unit * leverage
        
        # Risk-reward ratio
        reward = abs(take_profit - entry_price)
        risk = abs(entry_price - stop_loss)
        risk_reward = reward / risk if risk > 0 else 0
        
        return PositionSizing(
            symbol=symbol,
            direction=direction,
            size_pct=position_pct,
            size_units=size_units,
            leverage=leverage,
            max_loss_pct=max_loss_pct,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            risk_reward_ratio=risk_reward
        )
    
    def check_portfolio_heat(self, new_position_risk: float = 0) -> bool:
        """
        Check if adding new position would exceed heat limit
        """
        current_heat = sum(
            pos.get('risk_pct', 0) for pos in self.open_positions.values()
        )
        
        total_heat = current_heat + new_position_risk
        
        if total_heat > self.risk_limits['max_portfolio_heat']:
            logger.warning(
                f"Portfolio heat {total_heat:.2%} would exceed limit "
                f"{self.risk_limits['max_portfolio_heat']:.2%}"
            )
            return False
        
        return True
    
    def check_correlation_risk(
        self,
        symbol: str,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> bool:
        """
        Check correlation with existing positions
        Avoid taking multiple correlated positions
        """
        if correlation_matrix is None or symbol not in correlation_matrix.columns:
            return True
        
        for open_symbol in self.open_positions.keys():
            if open_symbol in correlation_matrix.columns:
                corr = correlation_matrix.loc[symbol, open_symbol]
                if abs(corr) > 0.8:  # High correlation
                    logger.warning(
                        f"High correlation ({corr:.2f}) between {symbol} and {open_symbol}"
                    )
                    return False
        
        return True
    
    def update_drawdown(self, pnl: float):
        """
        Update drawdown calculations
        """
        self.current_equity += pnl
        self.daily_pnl.append(pnl)
        
        # Update peak equity
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        # Calculate current drawdown
        current_dd = (self.peak_equity - self.current_equity) / self.peak_equity
        
        # Check circuit breaker
        if current_dd > self.risk_limits['max_drawdown_limit']:
            self.trigger_circuit_breaker(
                f"Max drawdown exceeded: {current_dd:.2%}"
            )
        
        return current_dd
    
    def trigger_circuit_breaker(self, reason: str):
        """
        Trigger trading halt
        """
        self.circuit_breaker_triggered = True
        self.circuit_breaker_reason = reason
        self.circuit_breaker_time = datetime.now()
        
        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")
        
        # Close all positions (in live trading)
        self.open_positions.clear()
    
    def reset_circuit_breaker(self):
        """
        Reset circuit breaker after cooling off period
        """
        if not self.circuit_breaker_triggered:
            return
        
        # Require 24-hour cooling off
        if self.circuit_breaker_time:
            hours_elapsed = (datetime.now() - self.circuit_breaker_time).total_seconds() / 3600
            if hours_elapsed > 24:
                self.circuit_breaker_triggered = False
                self.circuit_breaker_reason = None
                self.circuit_breaker_time = None
                logger.info("Circuit breaker reset")
    
    def calculate_var(self, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk using historical simulation
        """
        if len(self.daily_pnl) < 30:
            return 0
        
        returns = pd.Series(self.daily_pnl) / self.current_equity
        var = np.percentile(returns, (1 - confidence) * 100)
        
        return abs(var)
    
    def get_risk_metrics(self) -> RiskMetrics:
        """
        Calculate current risk metrics
        """
        if len(self.daily_pnl) < 10:
            return RiskMetrics(
                portfolio_heat=0,
                max_drawdown=0,
                current_drawdown=0,
                daily_var=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                win_rate=0,
                profit_factor=0,
                risk_of_ruin=0
            )
        
        returns = pd.Series(self.daily_pnl)
        
        # Basic metrics
        current_dd = (self.peak_equity - self.current_equity) / self.peak_equity
        portfolio_heat = sum(
            pos.get('risk_pct', 0) for pos in self.open_positions.values()
        )
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation only)
        downside_returns = returns[returns < 0]
        sortino = returns.mean() / downside_returns.std() * np.sqrt(365) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Risk of ruin (simplified)
        # Probability of losing X% of account
        risk_of_ruin = self._calculate_risk_of_ruin(win_rate, returns.std())
        
        return RiskMetrics(
            portfolio_heat=portfolio_heat,
            max_drawdown=self.risk_limits['max_drawdown_limit'],
            current_drawdown=current_dd,
            daily_var=self.calculate_var(),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            win_rate=win_rate,
            profit_factor=profit_factor,
            risk_of_ruin=risk_of_ruin
        )
    
    def _calculate_risk_of_ruin(
        self,
        win_rate: float,
        volatility: float,
        ruin_threshold: float = 0.5
    ) -> float:
        """
        Calculate probability of ruin using Gambler's Ruin formula
        """
        if win_rate <= 0.5:
            return 1.0
        
        # Simplified calculation
        # In reality, this would use more sophisticated methods
        edge = win_rate - 0.5
        risk_of_ruin = np.exp(-2 * edge * ruin_threshold / (volatility ** 2))
        
        return min(risk_of_ruin, 1.0)
    
    def can_trade(self) -> Tuple[bool, Optional[str]]:
        """
        Check if trading is allowed
        """
        if self.circuit_breaker_triggered:
            return False, f"Circuit breaker active: {self.circuit_breaker_reason}"
        
        if self.calculate_var() > self.risk_limits['daily_var_limit']:
            return False, "Daily VaR limit exceeded"
        
        return True, None


# Convenience functions
def calculate_position_size(
    entry: float,
    stop: float,
    target: float,
    account: float,
    risk_pct: float = 0.02
) -> Dict:
    """Quick position size calculation"""
    risk = abs(entry - stop)
    reward = abs(target - entry)
    
    if risk == 0:
        return {'error': 'Invalid stop loss'}
    
    risk_amount = account * risk_pct
    position_size = risk_amount / risk
    
    return {
        'position_size': position_size,
        'risk_amount': risk_amount,
        'risk_reward': reward / risk,
        'max_loss_pct': risk_pct,
        'leverage': 1.0
    }

"""
AEGIS Time-Series Cross Validation Module
Walk-forward validation with purging and embargo
"""

import logging
from typing import List, Tuple, Generator, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)


@dataclass
class PurgedKFoldConfig:
    """Configuration for purged cross-validation"""
    n_splits: int = 5
    purge_pct: float = 0.01  # Purge 1% of data between train and test
    embargo_pct: float = 0.01  # Embargo 1% of test data


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold cross-validation for time series
    
    Purging: Remove observations between train and test to prevent leakage
    Embargo: Remove observations from test set that overlap with train
    """
    
    def __init__(self, n_splits: int = 5, purge_pct: float = 0.01, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
    
    def split(self, X, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate fold sizes
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Test set is the current fold
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)
            
            # Calculate purge and embargo sizes
            purge_size = int(fold_size * self.purge_pct)
            embargo_size = int(fold_size * self.embargo_pct)
            
            # Training set is everything before test, minus purge
            train_end = max(0, test_start - purge_size)
            train_indices = indices[:train_end]
            
            # Test set minus embargo
            test_indices = indices[test_start + embargo_size:test_end]
            
            if len(train_indices) == 0 or len(test_indices) == 0:
                logger.warning(f"Empty train or test set in fold {i}")
                continue
            
            yield train_indices, test_indices


class WalkForwardValidator:
    """
    Walk-forward validation with expanding or fixed window
    """
    
    def __init__(
        self,
        min_train_size: int = 1000,
        test_size: int = 100,
        step_size: int = 50,
        expanding: bool = True
    ):
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.expanding = expanding
    
    def split(self, df: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generate walk-forward splits
        """
        n_samples = len(df)
        
        if n_samples < self.min_train_size + self.test_size:
            raise ValueError("Not enough data for walk-forward validation")
        
        start_idx = 0
        
        while start_idx + self.min_train_size + self.test_size <= n_samples:
            # Training set
            if self.expanding:
                train_end = start_idx + self.min_train_size
            else:
                train_end = start_idx + self.min_train_size
            
            train_df = df.iloc[start_idx:train_end]
            
            # Test set
            test_start = train_end
            test_end = min(test_start + self.test_size, n_samples)
            test_df = df.iloc[test_start:test_end]
            
            yield train_df, test_df
            
            # Move forward
            start_idx += self.step_size
    
    def get_n_splits(self, df: pd.DataFrame) -> int:
        """Calculate number of splits"""
        n_samples = len(df)
        usable = n_samples - self.min_train_size - self.test_size
        return max(0, usable // self.step_size + 1)


class CombinatorialPurgedCV:
    """
    Combinatorial purged cross-validation (advanced)
    Used for strategy backtesting with multiple paths
    """
    
    def __init__(
        self,
        n_splits: int = 10,
        n_test_splits: int = 2,
        purge_pct: float = 0.01
    ):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_pct = purge_pct
    
    def split(self, df: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, List[pd.DataFrame]], None, None]:
        """
        Generate combinatorial splits
        
        Returns training set and list of test sets (multiple paths)
        """
        from itertools import combinations
        
        n_samples = len(df)
        split_size = n_samples // self.n_splits
        
        # Generate all combinations of test splits
        test_combinations = list(combinations(range(self.n_splits), self.n_test_splits))
        
        for test_indices in test_combinations:
            # Calculate train and test ranges
            train_indices = [i for i in range(self.n_splits) if i not in test_indices]
            
            # Build training set
            train_dfs = []
            for idx in train_indices:
                start = idx * split_size
                end = min((idx + 1) * split_size, n_samples)
                train_dfs.append(df.iloc[start:end])
            
            train_df = pd.concat(train_dfs).sort_index()
            
            # Build test sets (multiple paths)
            test_dfs = []
            for idx in test_indices:
                start = idx * split_size
                end = min((idx + 1) * split_size, n_samples)
                
                # Apply embargo
                embargo = int(split_size * self.purge_pct)
                test_dfs.append(df.iloc[start + embargo:end])
            
            yield train_df, test_dfs


class ValidationMetrics:
    """
    Calculate validation metrics for time series models
    """
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> dict:
        """
        Calculate comprehensive validation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, log_loss, confusion_matrix
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }
        
        # Calculate per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, (p, r) in enumerate(zip(precision_per_class, recall_per_class)):
            metrics[f'precision_class_{i}'] = p
            metrics[f'recall_class_{i}'] = r
        
        # Probability-based metrics
        if y_prob is not None:
            try:
                # For multi-class, use ROC AUC with OvR
                if y_prob.shape[1] > 2:
                    metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                
                metrics['log_loss'] = log_loss(y_true, y_prob)
            except Exception as e:
                logger.warning(f"Could not calculate probability metrics: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Directional accuracy (for financial time series)
        if len(y_true) > 1:
            # Calculate if direction predictions are correct
            true_direction = np.sign(y_true)
            pred_direction = np.sign(y_pred)
            metrics['directional_accuracy'] = np.mean(true_direction == pred_direction)
        
        return metrics
    
    @staticmethod
    def calculate_financial_metrics(
        returns: pd.Series,
        predictions: pd.Series,
        transaction_cost: float = 0.001
    ) -> dict:
        """
        Calculate financial performance metrics
        """
        # Strategy returns (only trade when prediction != 0)
        strategy_returns = returns * np.sign(predictions) - transaction_cost * (predictions != 0)
        
        metrics = {
            'total_return': (1 + strategy_returns).prod() - 1,
            'annualized_return': strategy_returns.mean() * 252 * 24,  # Hourly data
            'annualized_volatility': strategy_returns.std() * np.sqrt(252 * 24),
            'sharpe_ratio': (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252 * 24) if strategy_returns.std() > 0 else 0,
            'max_drawdown': (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min(),
            'win_rate': (strategy_returns > 0).mean(),
            'profit_factor': abs(strategy_returns[strategy_returns > 0].sum() / strategy_returns[strategy_returns < 0].sum()) if strategy_returns[strategy_returns < 0].sum() != 0 else np.inf,
            'calmar_ratio': (strategy_returns.mean() * 252 * 24) / abs((strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()) if (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min() != 0 else 0
        }
        
        return metrics


# Convenience functions
def walk_forward_validate(model, df: pd.DataFrame, feature_cols: List[str], target_col: str = 'target'):
    """Quick walk-forward validation"""
    validator = WalkForwardValidator()
    metrics_list = []
    
    for train_df, test_df in validator.split(df):
        # Train
        model.fit(train_df[feature_cols], train_df[target_col])
        
        # Predict
        predictions = model.predict(test_df[feature_cols])
        probabilities = model.predict_proba(test_df[feature_cols]) if hasattr(model, 'predict_proba') else None
        
        # Metrics
        metrics = ValidationMetrics.calculate_metrics(
            test_df[target_col].values,
            predictions,
            probabilities
        )
        metrics_list.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in metrics_list[0].keys():
        if isinstance(metrics_list[0][key], (int, float)):
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])
    
    return avg_metrics

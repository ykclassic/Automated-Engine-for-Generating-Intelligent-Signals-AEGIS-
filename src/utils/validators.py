"""
AEGIS Data Validators
Comprehensive validation for market data quality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class DataQualityIssue(Enum):
    """Enumeration of data quality issues"""
    MISSING_DATA = "missing_data"
    PRICE_GAP = "price_gap"
    VOLUME_SPIKE = "volume_spike"
    OUTLIER_PRICE = "outlier_price"
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    NON_MONOTONIC = "non_monotonic"
    ZERO_VOLUME = "zero_volume"
    NEGATIVE_PRICE = "negative_price"


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    issues: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    repair_suggestions: List[str]


class DataValidator:
    """
    Institutional-grade data validator for OHLCV data
    """
    
    def __init__(
        self,
        max_missing_pct: float = 1.0,
        max_gap_minutes: int = 60,
        outlier_std_threshold: float = 4.0,
        volume_spike_threshold: float = 5.0
    ):
        self.max_missing_pct = max_missing_pct
        self.max_gap_minutes = max_gap_minutes
        self.outlier_std_threshold = outlier_std_threshold
        self.volume_spike_threshold = volume_spike_threshold
    
    def validate_ohlcv(
        self,
        df: pd.DataFrame,
        timeframe: str,
        symbol: str
    ) -> ValidationResult:
        """
        Comprehensive OHLCV data validation
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Data timeframe (e.g., '1h', '4h')
            symbol: Trading pair symbol
        
        Returns:
            ValidationResult with issues and statistics
        """
        issues = []
        stats = {}
        
        # Basic structure validation
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            return ValidationResult(
                is_valid=False,
                issues=[{
                    "type": DataQualityIssue.MISSING_DATA.value,
                    "message": f"Missing columns: {missing_cols}"
                }],
                statistics={},
                repair_suggestions=[]
            )
        
        # Ensure timestamp is datetime
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # 1. Check for missing data
        missing_stats = self._check_missing_data(df)
        stats['missing_data'] = missing_stats
        if missing_stats['total_missing_pct'] > self.max_missing_pct:
            issues.append({
                "type": DataQualityIssue.MISSING_DATA.value,
                "severity": "high",
                "message": f"Missing data exceeds threshold: {missing_stats['total_missing_pct']:.2f}%",
                "details": missing_stats
            })
        
        # 2. Check timestamp continuity (gaps)
        gap_issues = self._check_timestamp_gaps(df, timeframe)
        if gap_issues:
            issues.extend([{
                "type": DataQualityIssue.PRICE_GAP.value,
                "severity": "medium",
                "message": f"Found {len(gap_issues)} timestamp gaps",
                "details": gap_issues[:5]  # First 5 gaps
            }])
        
        # 3. Check for duplicates
        dup_stats = self._check_duplicates(df)
        stats['duplicates'] = dup_stats
        if dup_stats['count'] > 0:
            issues.append({
                "type": DataQualityIssue.DUPLICATE_TIMESTAMP.value,
                "severity": "high",
                "message": f"Found {dup_stats['count']} duplicate timestamps",
                "details": dup_stats
            })
        
        # 4. Check price validity
        price_issues = self._check_price_validity(df)
        if price_issues:
            issues.extend(price_issues)
        
        # 5. Check for outliers
        outlier_stats = self._check_outliers(df)
        stats['outliers'] = outlier_stats
        if outlier_stats['count'] > 0:
            issues.append({
                "type": DataQualityIssue.OUTLIER_PRICE.value,
                "severity": "medium",
                "message": f"Found {outlier_stats['count']} price outliers",
                "details": outlier_stats
            })
        
        # 6. Check volume anomalies
        volume_stats = self._check_volume_anomalies(df)
        stats['volume'] = volume_stats
        if volume_stats['spike_count'] > 0:
            issues.append({
                "type": DataQualityIssue.VOLUME_SPIKE.value,
                "severity": "low",
                "message": f"Found {volume_stats['spike_count']} volume spikes",
                "details": volume_stats
            })
        
        # 7. OHLC logic validation
        ohlc_issues = self._check_ohlc_logic(df)
        if ohlc_issues:
            issues.extend(ohlc_issues)
        
        # Generate repair suggestions
        repairs = self._generate_repair_suggestions(issues)
        
        # Determine overall validity
        critical_issues = [i for i in issues if i.get('severity') == 'high']
        is_valid = len(critical_issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            statistics=stats,
            repair_suggestions=repairs
        )
    
    def _check_missing_data(self, df: pd.DataFrame) -> Dict:
        """Calculate missing data statistics"""
        missing_counts = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        total_missing = missing_counts.sum()
        
        return {
            'total_missing': int(total_missing),
            'total_cells': int(total_cells),
            'total_missing_pct': (total_missing / total_cells) * 100 if total_cells > 0 else 0,
            'by_column': missing_counts.to_dict()
        }
    
    def _check_timestamp_gaps(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> List[Dict]:
        """Check for gaps in timestamp continuity"""
        # Parse timeframe to timedelta
        tf_minutes = self._timeframe_to_minutes(timeframe)
        expected_diff = pd.Timedelta(minutes=tf_minutes)
        
        # Calculate actual differences
        diffs = df['timestamp'].diff().dropna()
        gaps = diffs[diffs > expected_diff * 1.5]  # Allow 50% tolerance
        
        gap_list = []
        for idx, gap in gaps.items():
            gap_list.append({
                'index': int(idx),
                'timestamp': df.loc[idx, 'timestamp'].isoformat(),
                'gap_minutes': int(gap.total_seconds() / 60),
                'expected_minutes': tf_minutes
            })
        
        return gap_list
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate timestamps"""
        dup_counts = df['timestamp'].duplicated().sum()
        dup_timestamps = df[df['timestamp'].duplicated()]['timestamp'].tolist()
        
        return {
            'count': int(dup_counts),
            'timestamps': [str(t) for t in dup_timestamps[:10]]  # First 10
        }
    
    def _check_price_validity(self, df: pd.DataFrame) -> List[Dict]:
        """Check for invalid price values"""
        issues = []
        
        # Check for negative prices
        for col in ['open', 'high', 'low', 'close']:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                issues.append({
                    "type": DataQualityIssue.NEGATIVE_PRICE.value,
                    "severity": "high",
                    "message": f"Found {neg_count} negative values in {col}",
                    "column": col
                })
        
        # Check for zero prices
        for col in ['close']:
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                issues.append({
                    "type": DataQualityIssue.MISSING_DATA.value,
                    "severity": "high",
                    "message": f"Found {zero_count} zero values in {col}",
                    "column": col
                })
        
        return issues
    
    def _check_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect price outliers using Z-score"""
        # Calculate returns
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 30:
            return {'count': 0, 'indices': []}
        
        z_scores = np.abs((returns - returns.mean()) / returns.returns.std())
        outliers = z_scores[z_scores > self.outlier_std_threshold]
        
        return {
            'count': int(len(outliers)),
            'indices': outliers.index.tolist()[:20],
            'max_z_score': float(z_scores.max()) if len(z_scores) > 0 else 0,
            'threshold': self.outlier_std_threshold
        }
    
    def _check_volume_anomalies(self, df: pd.DataFrame) -> Dict:
        """Detect volume spikes"""
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_std = df['volume'].rolling(window=20).std()
        
        # Avoid division by zero
        threshold = volume_ma + (self.volume_spike_threshold * volume_std)
        spikes = df['volume'] > threshold
        
        return {
            'spike_count': int(spikes.sum()),
            'zero_volume_count': int((df['volume'] == 0).sum()),
            'mean_volume': float(df['volume'].mean()),
            'max_volume': float(df['volume'].max())
        }
    
    def _check_ohlc_logic(self, df: pd.DataFrame) -> List[Dict]:
        """Validate OHLC relationships"""
        issues = []
        
        # High should be >= Low
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            issues.append({
                "type": DataQualityIssue.NON_MONOTONIC.value,
                "severity": "high",
                "message": f"High < Low in {invalid_hl} candles"
            })
        
        # High should be >= Open, Close
        invalid_high = ((df['high'] < df['open']) | (df['high'] < df['close'])).sum()
        if invalid_high > 0:
            issues.append({
                "type": DataQualityIssue.NON_MONOTONIC.value,
                "severity": "high",
                "message": f"High < Open or Close in {invalid_high} candles"
            })
        
        # Low should be <= Open, Close
        invalid_low = ((df['low'] > df['open']) | (df['low'] > df['close'])).sum()
        if invalid_low > 0:
            issues.append({
                "type": DataQualityIssue.NON_MONOTONIC.value,
                "severity": "high",
                "message": f"Low > Open or Close in {invalid_low} candles"
            })
        
        return issues
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        multipliers = {
            'm': 1,
            'h': 60,
            'd': 1440,
            'w': 10080
        }
        
        unit = timeframe[-1].lower()
        value = int(timeframe[:-1])
        
        return value * multipliers.get(unit, 60)
    
    def _generate_repair_suggestions(self, issues: List[Dict]) -> List[str]:
        """Generate repair suggestions based on issues"""
        suggestions = []
        
        issue_types = [i['type'] for i in issues]
        
        if DataQualityIssue.MISSING_DATA.value in issue_types:
            suggestions.append("Consider forward-fill for minor gaps or fetch missing data")
        
        if DataQualityIssue.PRICE_GAP.value in issue_types:
            suggestions.append("Validate gaps against other exchanges for arbitrage opportunities")
        
        if DataQualityIssue.DUPLICATE_TIMESTAMP.value in issue_types:
            suggestions.append("Remove duplicates keeping last occurrence")
        
        if DataQualityIssue.OUTLIER_PRICE.value in issue_types:
            suggestions.append("Review outliers - may be valid flash crashes or exchange errors")
        
        if DataQualityIssue.NEGATIVE_PRICE.value in issue_types:
            suggestions.append("CRITICAL: Data source error - exclude affected timestamps")
        
        return suggestions


class CausalValidator:
    """
    Validates that features are calculated without look-ahead bias
    Critical for ML pipeline integrity
    """
    
    @staticmethod
    def verify_causal_calculation(
        df: pd.DataFrame,
        feature_col: str,
        calculation_window: int
    ) -> bool:
        """
        Verify that a feature is calculated causally
        
        Args:
            df: DataFrame with features
            feature_col: Feature column name
            calculation_window: Window used for calculation
        
        Returns:
            True if causal, False if look-ahead bias detected
        """
        # For a causal calculation, the first (window-1) values should be NaN
        # and there should be no correlation with future values
        
        feature = df[feature_col]
        
        # Check initial NaN period
        first_valid_idx = feature.first_valid_index()
        if first_valid_idx is None:
            return False
        
        expected_nans = calculation_window - 1
        actual_nans = feature.iloc[:expected_nans].isna().sum()
        
        if actual_nans < expected_nans:
            return False
        
        # Check no correlation with future prices (basic test)
        if len(feature.dropna()) > 50:
            current_feature = feature.shift(1).dropna()  # Lagged feature
            future_return = df['close'].pct_change().shift(-1).dropna()  # Future return
            
            min_len = min(len(current_feature), len(future_return))
            if min_len > 10:
                correlation = np.corrcoef(
                    current_feature.iloc[:min_len],
                    future_return.iloc[:min_len]
                )[0, 1]
                
                # Suspicious if correlation is too high (indicates look-ahead)
                if abs(correlation) > 0.5:
                    return False
        
        return True

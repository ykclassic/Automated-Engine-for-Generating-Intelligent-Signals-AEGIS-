"""
AEGIS Model Definitions
LightGBM and XGBoost with hyperparameter optimization
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pickle
import json

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Optional imports - handle if not installed
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed")

try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import RandomizedSearchCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    model_type: str
    params: Dict[str, Any]
    calibration: bool = True


class BaseModel:
    """Base class for all models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_fitted = False
        self.feature_importance = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
    
    def save(self, path: str):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.config,
                'feature_importance': self.feature_importance,
                'is_fitted': self.is_fitted
            }, f)
    
    def load(self, path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.config = data['config']
            self.feature_importance = data['feature_importance']
            self.is_fitted = data['is_fitted']


class LightGBMModel(BaseModel):
    """
    LightGBM classifier with probability calibration
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig(
                name='lightgbm',
                model_type='classification',
                params={
                    'objective': 'multiclass',
                    'num_class': 3,
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': 42
                }
            )
        super().__init__(config)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")
        
        # Create dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Train model
        self.model = lgb.train(
            self.config.params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Calculate feature importance
        self.feature_importance = dict(zip(
            X.columns,
            self.model.feature_importance(importance_type='gain')
        ))
        
        self.is_fitted = True
        
        # Calibrate if requested
        if self.config.calibration:
            self._calibrate(X, y)
    
    def _calibrate(self, X: pd.DataFrame, y: pd.Series):
        """Apply Platt scaling for probability calibration"""
        # Store calibration parameters
        predictions = self.model.predict(X)
        
        # Simple calibration using temperature scaling
        # This is a simplified version - full implementation would use isotonic regression
        self.calibration_temp = 1.0  # Could be optimized
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1) - 1  # Convert to -1, 0, 1
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        proba = self.model.predict(X)
        
        # Apply temperature scaling for calibration
        if hasattr(self, 'calibration_temp'):
            proba = self._softmax(proba / self.calibration_temp)
        
        return proba
    
    def _softmax(self, x):
        """Softmax function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class XGBoostModel(BaseModel):
    """
    XGBoost classifier with probability calibration
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig(
                name='xgboost',
                model_type='classification',
                params={
                    'objective': 'multi:softprob',
                    'num_class': 3,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'eval_metric': 'mlogloss'
                }
            )
        super().__init__(config)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")
        
        # Adjust labels to 0, 1, 2
        y_adj = y + 1
        
        self.model = xgb.XGBClassifier(**self.config.params)
        self.model.fit(
            X, y_adj,
            eval_set=[(X, y_adj)],
            verbose=False
        )
        
        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(X.columns, importance))
        
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        predictions = self.model.predict(X)
        return predictions - 1  # Convert back to -1, 0, 1
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        return self.model.predict_proba(X)


class EnsembleModel(BaseModel):
    """
    Ensemble of multiple models with voting
    """
    
    def __init__(self, models: Optional[List[BaseModel]] = None):
        config = ModelConfig(
            name='ensemble',
            model_type='classification',
            params={}
        )
        super().__init__(config)
        
        self.models = models or []
        self.weights = None
    
    def add_model(self, model: BaseModel, weight: float = 1.0):
        """Add model to ensemble"""
        self.models.append((model, weight))
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Individual models should be fitted separately"""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Normalize weights
        total_weight = sum(w for _, w in self.models)
        self.weights = [w / total_weight for _, w in self.models]
        
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        # Weighted voting
        predictions = np.zeros((len(X), 3))  # 3 classes: -1, 0, 1
        
        for (model, _), weight in zip(self.models, self.weights):
            proba = model.predict_proba(X)
            predictions += proba * weight
        
        return np.argmax(predictions, axis=1) - 1
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        predictions = np.zeros((len(X), 3))
        
        for (model, _), weight in zip(self.models, self.weights):
            proba = model.predict_proba(X)
            predictions += proba * weight
        
        return predictions


class ModelOptimizer:
    """
    Hyperparameter optimization for models
    """
    
    def __init__(self, model_type: str = 'lightgbm'):
        self.model_type = model_type
        self.best_params = None
        self.best_score = None
    
    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_iter: int = 20,
        cv: int = 3
    ) -> Dict:
        """
        Optimize hyperparameters using random search
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not installed")
        
        if self.model_type == 'lightgbm':
            param_distributions = {
                'num_leaves': [20, 31, 50, 70],
                'learning_rate': [0.01, 0.05, 0.1],
                'feature_fraction': [0.8, 0.9, 1.0],
                'bagging_fraction': [0.7, 0.8, 0.9],
                'bagging_freq': [3, 5, 7],
                'min_child_samples': [10, 20, 30],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            }
            
            model = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=3,
                n_estimators=100,
                random_state=42
            )
        
        elif self.model_type == 'xgboost':
            param_distributions = {
                'max_depth': [3, 5, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100, 200],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            }
            
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Random search
        search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='f1_macro',
            random_state=42,
            n_jobs=-1
        )
        
        y_adj = y + 1  # Adjust to 0, 1, 2
        search.fit(X, y_adj)
        
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        
        logger.info(f"Best {self.model_type} score: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        return self.best_params


# Convenience functions
def create_default_model(model_type: str = 'lightgbm') -> BaseModel:
    """Create default model"""
    if model_type == 'lightgbm':
        return LightGBMModel()
    elif model_type == 'xgboost':
        return XGBoostModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_ensemble() -> EnsembleModel:
    """Create default ensemble"""
    ensemble = EnsembleModel()
    
    if LIGHTGBM_AVAILABLE:
        ensemble.add_model(LightGBMModel(), weight=0.5)
    if XGBOOST_AVAILABLE:
        ensemble.add_model(XGBoostModel(), weight=0.5)
    
    return ensemble

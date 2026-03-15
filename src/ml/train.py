"""
AEGIS Model Training Pipeline
End-to-end training with validation and calibration
"""

import logging
import json
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.preprocessing import LabelEncoder

# Suppress noise
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

from .features import MLFeatureEngineer, engineer_ml_features
from .validation import (
    WalkForwardValidator, PurgedKFold, 
    ValidationMetrics, walk_forward_validate
)
from .models import (
    LightGBMModel, XGBoostModel, EnsembleModel,
    ModelOptimizer, create_ensemble, ModelConfig
)

logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.feature_engineer = MLFeatureEngineer()
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoder = LabelEncoder()
        self.training_log = []
    
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_selection: bool = True,
        top_n_features: int = 30
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare data with robust Label Encoding
        """
        logger.info("Engineering features...")
        df_features = engineer_ml_features(df, include_target=True)
        
        # 1. Drop NaNs
        df_clean = df_features.dropna().copy()
        
        # 2. Robust Label Encoding (Forces -1, 0, 1 into 0, 1, 2)
        # This handles cases where one class might be missing in a small fold
        df_clean['target'] = self.label_encoder.fit_transform(df_clean['target'].astype(int))
        
        logger.info(f"Class mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")

        if len(df_clean) < 1000:
            raise ValueError(f"Insufficient data: {len(df_clean)}")
        
        # 3. Feature Selection
        if feature_selection:
            selected_features = self.feature_engineer.get_feature_importance_mask(
                df_clean, top_n=top_n_features
            )
        else:
            selected_features = [c for c in df_clean.columns if c not in ['target', 'target_return']]
            
        return df_clean, selected_features
    
    def train_single_model(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        model_type: str = 'lightgbm',
        optimize: bool = False
    ) -> Tuple[object, Dict]:
        """
        Train a single model with XGBoost/LightGBM class safety
        """
        logger.info(f"Training {model_type}...")
        
        if optimize:
            optimizer = ModelOptimizer(model_type)
            best_params = optimizer.optimize(df[feature_cols], df['target'])
            config = ModelConfig(name=model_type, model_type='classification', params=best_params)
            model = LightGBMModel(config) if model_type == 'lightgbm' else XGBoostModel(config)
        else:
            model = LightGBMModel() if model_type == 'lightgbm' else XGBoostModel()
        
        validator = WalkForwardValidator(min_train_size=1000, test_size=200, step_size=100)
        fold_metrics = []
        
        for train_df, test_df in validator.split(df):
            X_tr, y_tr = train_df[feature_cols], train_df['target']
            X_te, y_te = test_df[feature_cols], test_df['target']
            
            # Ensure the fold has all required classes for XGBoost
            if len(np.unique(y_tr)) < 2:
                logger.warning("Skipping fold: insufficient class diversity.")
                continue

            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            probs = model.predict_proba(X_te)
            
            fold_metrics.append(ValidationMetrics.calculate_metrics(y_te.values, preds, probs))
        
        avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) 
                      for k in fold_metrics[0].keys() 
                      if isinstance(fold_metrics[0][key if 'key' in locals() else k], (int, float))}
        
        model.fit(df[feature_cols], df['target'])
        return model, avg_metrics

    # ... [Rest of TrainingPipeline methods (save_model, train_ensemble) remain the same] ...

    def run_full_training(self, df: pd.DataFrame, model_types=['lightgbm', 'xgboost', 'ensemble']):
        results = {}
        df_clean, feature_cols = self.prepare_data(df)
        
        for m_type in model_types:
            try:
                if m_type == 'ensemble':
                    model, metrics = self.train_ensemble(df_clean, feature_cols)
                else:
                    model, metrics = self.train_single_model(df_clean, feature_cols, m_type)
                
                self.save_model(model, m_type, feature_cols, metrics)
                results[m_type] = {'status': 'success', 'metrics': metrics}
            except Exception as e:
                logger.error(f"Failed {m_type}: {e}")
                results[m_type] = {'status': 'failed', 'error': str(e)}
        return results

def train_model(df: pd.DataFrame, model_type: str = 'ensemble') -> Tuple[object, Dict]:
    pipeline = TrainingPipeline()
    df_clean, feature_cols = pipeline.prepare_data(df)
    res = pipeline.run_full_training(df_clean, [model_type])
    return res

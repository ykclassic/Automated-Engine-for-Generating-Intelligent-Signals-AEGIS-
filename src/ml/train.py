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

# Suppress technical noise from LightGBM
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
    """
    End-to-end training pipeline with dynamic label correction
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.feature_engineer = MLFeatureEngineer()
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.training_log = []
    
    def _load_config(self, path: str) -> dict:
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_selection: bool = True,
        top_n_features: int = 30
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare data with strict alignment and dynamic label encoding
        """
        logger.info("Engineering features and aligning indices...")
        df_features = engineer_ml_features(df, include_target=True)
        
        # 1. Strict Clean
        df_clean = df_features.dropna().copy()
        
        if len(df_clean) < 1000:
            raise ValueError(f"Insufficient samples: {len(df_clean)}")

        # 2. Dynamic Zero-Indexed Label Mapping
        # This guarantees labels are always [0, 1, 2...] regardless of the raw input format
        unique_classes = sorted(df_clean['target'].unique())
        dynamic_map = {val: idx for idx, val in enumerate(unique_classes)}
        logger.info(f"Applying dynamic label map: {dynamic_map}")
        
        df_clean['target'] = df_clean['target'].map(dynamic_map).astype(int)
        
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
        Train a single model with class-diversity validation safety
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
        
        # Total classes required
        total_classes = len(np.unique(df['target']))

        for fold, (train_df, test_df) in enumerate(validator.split(df)):
            X_tr, y_tr = train_df[feature_cols], train_df['target']
            X_te, y_te = test_df[feature_cols], test_df['target']
            
            # XGBoost check: Skip fold if a class is missing in the training slice
            if len(np.unique(y_tr)) < total_classes:
                logger.warning(f"Skipping fold {fold}: Missing classes in training slice.")
                continue

            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            probs = model.predict_proba(X_te)
            
            fold_metrics.append(ValidationMetrics.calculate_metrics(y_te.values, preds, probs))
        
        if not fold_metrics:
            raise ValueError("All validation folds failed class diversity check. Increase data or window size.")

        avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) 
                      for k in fold_metrics[0].keys() 
                      if isinstance(fold_metrics[0][k], (int, float))}
        
        # Final training on full history
        model.fit(df[feature_cols], df['target'])
        
        self.training_log.append({
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'metrics': avg_metrics
        })
        
        return model, avg_metrics

    def train_ensemble(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[EnsembleModel, Dict]:
        """
        Train ensemble of multiple models with robust fallback
        """
        logger.info("Training ensemble...")
        ensemble = EnsembleModel()
        
        for m_type in ['lightgbm', 'xgboost']:
            try:
                m_obj, _ = self.train_single_model(df, feature_cols, m_type)
                ensemble.add_model(m_obj, weight=0.5)
            except Exception as e:
                logger.error(f"Ensemble failed to include {m_type}: {e}")
        
        if not ensemble.models:
            raise RuntimeError("Ensemble has no valid models.")

        ensemble.fit(df[feature_cols], df['target'])
        
        # Cross-validation for the ensemble
        validator = WalkForwardValidator(min_train_size=1000, test_size=200, step_size=100)
        e_metrics = []
        total_classes = len(np.unique(df['target']))
        
        for _, test_df in validator.split(df):
            if len(np.unique(test_df['target'])) < 2: continue
            
            probs = ensemble.predict_proba(test_df[feature_cols])
            preds = ensemble.predict(test_df[feature_cols])
            e_metrics.append(ValidationMetrics.calculate_metrics(test_df['target'].values, preds, probs))
            
        avg_metrics = {k: np.mean([m[k] for m in e_metrics]) 
                      for k in e_metrics[0].keys() 
                      if isinstance(e_metrics[0][k], (int, float))}
        
        return ensemble, avg_metrics

    def save_model(self, model: object, model_name: str, feature_cols: List[str], metrics: Dict):
        """
        Save model and metadata with JSON serialization safety
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"{model_name}_{timestamp}.joblib"
        meta_path = self.models_dir / f"{model_name}_{timestamp}_meta.json"
        
        joblib.dump(model, model_path)
        
        clean_metrics = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                        for k, v in metrics.items()}
        
        metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'features': feature_cols,
            'metrics': clean_metrics
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        joblib.dump(model, self.models_dir / f"{model_name}_latest.joblib")
        with open(self.models_dir / f"{model_name}_latest_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def run_full_training(
        self,
        df: pd.DataFrame,
        model_types: List[str] = ['lightgbm', 'xgboost', 'ensemble']
    ) -> Dict[str, Dict]:
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
                logger.error(f"Critical failure in {m_type}: {e}")
                results[m_type] = {'status': 'failed', 'error': str(e)}
                
        return results

def train_model(df: pd.DataFrame, model_type: str = 'ensemble') -> Tuple[object, Dict]:
    pipeline = TrainingPipeline()
    results = pipeline.run_full_training(df, [model_type])
    return results

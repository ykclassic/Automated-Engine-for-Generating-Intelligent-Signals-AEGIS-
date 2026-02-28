"""
AEGIS Model Training Pipeline
End-to-end training with validation and calibration
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import yaml
import joblib

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
    End-to-end training pipeline
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.feature_engineer = MLFeatureEngineer()
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
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
        Prepare data for training
        """
        logger.info("Engineering features...")
        df_features = engineer_ml_features(df, include_target=True)
        
        # Remove rows with missing targets
        df_clean = df_features.dropna(subset=['target'])
        
        if len(df_clean) < 1000:
            raise ValueError(f"Insufficient data: {len(df_clean)} samples")
        
        # Feature selection
        if feature_selection:
            logger.info(f"Selecting top {top_n_features} features...")
            selected_features = self.feature_engineer.get_feature_importance_mask(
                df_clean, top_n=top_n_features
            )
        else:
            selected_features = [
                c for c in df_clean.columns 
                if c not in ['target', 'target_return']
            ]
        
        logger.info(f"Final dataset: {len(df_clean)} samples, {len(selected_features)} features")
        
        return df_clean, selected_features
    
    def train_single_model(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        model_type: str = 'lightgbm',
        optimize: bool = False
    ) -> Tuple[object, Dict]:
        """
        Train a single model with validation
        """
        logger.info(f"Training {model_type} model...")
        
        # Optimize hyperparameters if requested
        if optimize:
            optimizer = ModelOptimizer(model_type)
            best_params = optimizer.optimize(df[feature_cols], df['target'])
            
            config = ModelConfig(
                name=model_type,
                model_type='classification',
                params=best_params
            )
            
            if model_type == 'lightgbm':
                model = LightGBMModel(config)
            else:
                model = XGBoostModel(config)
        else:
            if model_type == 'lightgbm':
                model = LightGBMModel()
            else:
                model = XGBoostModel()
        
        # Walk-forward validation
        logger.info("Running walk-forward validation...")
        validator = WalkForwardValidator(
            min_train_size=1000,
            test_size=200,
            step_size=100
        )
        
        fold_metrics = []
        for fold, (train_df, test_df) in enumerate(validator.split(df)):
            logger.info(f"Fold {fold + 1}/{validator.get_n_splits(df)}")
            
            # Train
            model.fit(train_df[feature_cols], train_df['target'])
            
            # Predict
            predictions = model.predict(test_df[feature_cols])
            probabilities = model.predict_proba(test_df[feature_cols])
            
            # Metrics
            metrics = ValidationMetrics.calculate_metrics(
                test_df['target'].values,
                predictions,
                probabilities
            )
            fold_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in fold_metrics[0].keys():
            if isinstance(fold_metrics[0][key], (int, float)):
                avg_metrics[key] = np.mean([m[key] for m in fold_metrics])
        
        # Final training on full dataset
        logger.info("Training final model on full dataset...")
        model.fit(df[feature_cols], df['target'])
        
        # Log training
        self.training_log.append({
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'n_samples': len(df),
            'n_features': len(feature_cols),
            'metrics': avg_metrics,
            'feature_importance': model.feature_importance
        })
        
        return model, avg_metrics
    
    def train_ensemble(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[EnsembleModel, Dict]:
        """
        Train ensemble of multiple models
        """
        logger.info("Training ensemble...")
        
        ensemble = EnsembleModel()
        
        # Train LightGBM
        try:
            lgb_model, lgb_metrics = self.train_single_model(
                df, feature_cols, 'lightgbm', optimize=False
            )
            ensemble.add_model(lgb_model, weight=0.5)
            logger.info(f"LightGBM F1: {lgb_metrics['f1_macro']:.4f}")
        except Exception as e:
            logger.warning(f"LightGBM training failed: {e}")
        
        # Train XGBoost
        try:
            xgb_model, xgb_metrics = self.train_single_model(
                df, feature_cols, 'xgboost', optimize=False
            )
            ensemble.add_model(xgb_model, weight=0.5)
            logger.info(f"XGBoost F1: {xgb_metrics['f1_macro']:.4f}")
        except Exception as e:
            logger.warning(f"XGBoost training failed: {e}")
        
        # Fit ensemble (just sets weights)
        ensemble.fit(df[feature_cols], df['target'])
        
        # Validate ensemble
        validator = WalkForwardValidator(min_train_size=1000, test_size=200, step_size=100)
        fold_metrics = []
        
        for train_df, test_df in validator.split(df):
            predictions = ensemble.predict(test_df[feature_cols])
            probabilities = ensemble.predict_proba(test_df[feature_cols])
            
            metrics = ValidationMetrics.calculate_metrics(
                test_df['target'].values,
                predictions,
                probabilities
            )
            fold_metrics.append(metrics)
        
        avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) 
                      for k in fold_metrics[0].keys() 
                      if isinstance(fold_metrics[0][k], (int, float))}
        
        return ensemble, avg_metrics
    
    def save_model(
        self,
        model: object,
        model_name: str,
        feature_cols: List[str],
        metrics: Dict
    ):
        """
        Save model and metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"{model_name}_{timestamp}.joblib"
        meta_path = self.models_dir / f"{model_name}_{timestamp}_meta.json"
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'features': feature_cols,
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in metrics.items()},
            'training_log': self.training_log
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved: {model_path}")
        
        # Also save as 'latest'
        latest_path = self.models_dir / f"{model_name}_latest.joblib"
        latest_meta_path = self.models_dir / f"{model_name}_latest_meta.json"
        joblib.dump(model, latest_path)
        
        with open(latest_meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def run_full_training(
        self,
        df: pd.DataFrame,
        model_types: List[str] = ['lightgbm', 'xgboost', 'ensemble']
    ) -> Dict[str, Dict]:
        """
        Run complete training pipeline for multiple models
        """
        results = {}
        
        # Prepare data
        df_clean, feature_cols = self.prepare_data(df)
        
        # Train each model type
        for model_type in model_types:
            try:
                if model_type == 'ensemble':
                    model, metrics = self.train_ensemble(df_clean, feature_cols)
                else:
                    model, metrics = self.train_single_model(
                        df_clean, feature_cols, model_type, optimize=False
                    )
                
                # Save model
                self.save_model(model, model_type, feature_cols, metrics)
                
                results[model_type] = {
                    'status': 'success',
                    'metrics': metrics,
                    'n_features': len(feature_cols)
                }
                
                logger.info(f"{model_type} training complete. F1: {metrics['f1_macro']:.4f}")
                
            except Exception as e:
                logger.error(f"{model_type} training failed: {e}")
                results[model_type] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results


# Convenience function
def train_model(df: pd.DataFrame, model_type: str = 'ensemble') -> Tuple[object, Dict]:
    """Quick model training"""
    pipeline = TrainingPipeline()
    df_clean, feature_cols = pipeline.prepare_data(df)
    
    if model_type == 'ensemble':
        model, metrics = pipeline.train_ensemble(df_clean, feature_cols)
    else:
        model, metrics = pipeline.train_single_model(df_clean, feature_cols, model_type)
    
    pipeline.save_model(model, model_type, feature_cols, metrics)
    return model, metrics

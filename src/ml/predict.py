"""
AEGIS Prediction Engine
Real-time inference with probability calibration
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

from .features import MLFeatureEngineer

logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Real-time prediction engine for signal generation
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_engineer = MLFeatureEngineer()
        self.feature_cols = None
        self.model_metadata = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, path: str):
        """
        Load trained model
        """
        path = Path(path)
        
        if not path.exists():
            # Try to find latest model
            models_dir = Path("data/models")
            latest = models_dir / f"{path.stem}_latest.joblib"
            if latest.exists():
                path = latest
            else:
                raise FileNotFoundError(f"Model not found: {path}")
        
        logger.info(f"Loading model from {path}")
        self.model = joblib.load(path)
        
        # Load metadata
        meta_path = path.with_suffix('').as_posix() + "_meta.json"
        if Path(meta_path).exists():
            import json
            with open(meta_path) as f:
                self.model_metadata = json.load(f)
                self.feature_cols = self.model_metadata.get('features')
        
        self.is_loaded = True
    
    def predict(
        self,
        df: pd.DataFrame,
        return_proba: bool = True
    ) -> Dict:
        """
        Generate prediction for current data
        
        Returns:
            Dictionary with prediction, probability, and confidence
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Engineer features
        df_features = self.feature_engineer.create_feature_matrix(df, include_target=False)
        
        if len(df_features) < 1:
            raise ValueError("Insufficient data for prediction")
        
        # Get latest row
        latest = df_features.iloc[-1:]
        
        # Ensure all required features are present
        if self.feature_cols:
            missing = [f for f in self.feature_cols if f not in latest.columns]
            if missing:
                logger.warning(f"Missing features: {missing}")
                # Fill with zeros
                for f in missing:
                    latest[f] = 0
            
            latest = latest[self.feature_cols]
        
        # Predict
        prediction = self.model.predict(latest)[0]
        
        result = {
            'prediction': int(prediction),
            'direction': 'bullish' if prediction == 1 else ('bearish' if prediction == -1 else 'neutral'),
            'timestamp': datetime.now().isoformat()
        }
        
        if return_proba and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(latest)[0]
            
            # Class order: -1, 0, 1 (or 0, 1, 2 in model)
            proba_dict = {
                'bearish': float(probabilities[0]),
                'neutral': float(probabilities[1]),
                'bullish': float(probabilities[2])
            }
            
            result['probabilities'] = proba_dict
            result['confidence'] = float(probabilities[prediction + 1])  # Adjust index
            
            # Calibration check
            if result['confidence'] > 0.8:
                result['confidence_level'] = 'very_high'
            elif result['confidence'] > 0.6:
                result['confidence_level'] = 'high'
            elif result['confidence'] > 0.4:
                result['confidence_level'] = 'moderate'
            else:
                result['confidence_level'] = 'low'
        
        return result
    
    def predict_batch(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate predictions for entire dataframe (backtesting)
        """
        df_features = self.feature_engineer.create_feature_matrix(df, include_target=False)
        
        if self.feature_cols:
            df_features = df_features[self.feature_cols]
        
        predictions = self.model.predict(df_features)
        
        df_result = df.copy()
        df_result['ml_prediction'] = predictions
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(df_features)
            df_result['ml_proba_bearish'] = probabilities[:, 0]
            df_result['ml_proba_neutral'] = probabilities[:, 1]
            df_result['ml_proba_bullish'] = probabilities[:, 2]
            df_result['ml_confidence'] = probabilities[
                np.arange(len(predictions)), predictions + 1
            ]
        
        return df_result
    
    def get_feature_importance(self) -> Optional[Dict]:
        """
        Get feature importance from model
        """
        if self.model and hasattr(self.model, 'feature_importance'):
            return self.model.feature_importance
        return None
    
    def explain_prediction(self, df: pd.DataFrame) -> Dict:
        """
        Explain prediction using SHAP values (if available)
        """
        try:
            import shap
            
            df_features = self.feature_engineer.create_feature_matrix(df, include_target=False)
            latest = df_features.iloc[-1:]
            
            if self.feature_cols:
                latest = latest[self.feature_cols]
            
            explainer = shap.TreeExplainer(self.model.model)
            shap_values = explainer.shap_values(latest)
            
            # Get top contributing features
            feature_importance = list(zip(latest.columns, shap_values[0]))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return {
                'top_positive': [(f, float(v)) for f, v in feature_importance[:3] if v > 0],
                'top_negative': [(f, float(v)) for f, v in feature_importance[:3] if v < 0],
                'base_value': float(explainer.expected_value)
            }
            
        except ImportError:
            logger.warning("SHAP not installed, cannot explain prediction")
            return {}
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {}


# Convenience functions
def load_prediction_engine(model_name: str = 'ensemble_latest') -> PredictionEngine:
    """Load default prediction engine"""
    return PredictionEngine(f"data/models/{model_name}.joblib")

def predict_signal(df: pd.DataFrame, model_path: Optional[str] = None) -> Dict:
    """Quick prediction"""
    engine = PredictionEngine(model_path)
    return engine.predict(df)

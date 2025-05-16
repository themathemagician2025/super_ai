"""
Enhanced Prediction AGI with Advanced Data Processing

Features:
- Multi-Model Ensemble
- Dynamic Weight Adjustment
- Automatic Data Preprocessing
- Real-time Data Integration
- Memory-Based Learning
- Anomaly Detection
- Auto-Rollback
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import logging
from pathlib import Path
import asyncio

from ..data.preprocessing import DataPreprocessor, DataStats
from ..data.real_time_fetcher import RealTimeDataFetcher

logger = logging.getLogger(__name__)

class ModelConfig:
    """Configuration for prediction models"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load model configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return json.load(f)
        return {
            'models': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'initial_weight': 0.4
                },
                'linear_regression': {
                    'fit_intercept': True,
                    'initial_weight': 0.2
                },
                'lasso': {
                    'cv': 5,
                    'max_iter': 1000,
                    'initial_weight': 0.2
                },
                'ridge': {
                    'cv': 5,
                    'initial_weight': 0.2
                }
            },
            'memory_bank': {
                'max_size': 1000,
                'error_threshold': 0.1
            },
            'checkpointing': {
                'save_interval': 100,  # predictions
                'max_checkpoints': 5
            },
            'adaptation': {
                'weight_adjustment_rate': 0.05,
                'min_weight': 0.1,
                'max_weight': 0.9
            }
        }

class RichPredictionAGI:
    """Enhanced prediction AGI with advanced features"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = ModelConfig(config_path)
        self.preprocessor = DataPreprocessor()
        self.data_fetcher = RealTimeDataFetcher()
        
        # Initialize models
        self.models = {
            "random_forest": RandomForestRegressor(**self.config.config['models']['random_forest']),
            "linear_regression": LinearRegression(**self.config.config['models']['linear_regression']),
            "lasso": LassoCV(**self.config.config['models']['lasso']),
            "ridge": RidgeCV(**self.config.config['models']['ridge'])
        }
        
        # Initialize weights
        self.model_weights = {
            name: config['initial_weight'] 
            for name, config in self.config.config['models'].items()
        }
        
        # Memory and checkpointing
        self.memory_bank = []
        self.prediction_count = 0
        self.checkpoints = []
        self.best_checkpoint = None
        
        # Performance tracking
        self.model_performance = {name: [] for name in self.models}
        
    async def fetch_real_time_data(self, 
                                 forex_currencies: List[str],
                                 sports_list: List[str],
                                 stock_symbols: List[str]) -> pd.DataFrame:
        """Fetch and preprocess real-time data"""
        async with self.data_fetcher as fetcher:
            data = await fetcher.fetch_all_data(
                forex_currencies=forex_currencies,
                sports_list=sports_list,
                stock_symbols=stock_symbols
            )
            
        # Convert to DataFrame
        df = pd.DataFrame()
        
        # Process forex data
        for currency, rates in data['forex'].items():
            if isinstance(rates, dict):
                for target, rate in rates.get('rates', {}).items():
                    df[f'forex_{currency}_{target}'] = rate
                    
        # Process sports data
        for sport, odds in data['sports'].items():
            if isinstance(odds, list):
                for game in odds:
                    df[f'odds_{sport}_{game["id"]}'] = game.get('price', np.nan)
                    
        # Process stock data
        for symbol, quote in data['stocks'].items():
            if isinstance(quote, dict):
                df[f'stock_{symbol}_price'] = quote.get('price', np.nan)
                df[f'stock_{symbol}_volume'] = quote.get('volume', np.nan)
                
        return df
        
    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Enhanced preprocessing with statistics"""
        # Remove outliers and get statistics
        processed_data, stats = self.preprocessor.preprocess(
            data=df.values,
            timestamps=[datetime.now()] * len(df)
        )
        
        # Convert back to DataFrame
        processed_df = pd.DataFrame(
            processed_data,
            columns=df.columns
        )
        
        return processed_df, stats
        
    def detect_anomalies(self, df: pd.DataFrame, threshold: float = 3.0) -> Tuple[pd.DataFrame, np.ndarray]:
        """Enhanced anomaly detection with multiple methods"""
        # Z-score method
        z_scores = (df - df.mean()) / df.std()
        z_score_mask = (np.abs(z_scores) < threshold).all(axis=1)
        
        # IQR method
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        iqr_mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
        
        # Combine methods
        final_mask = z_score_mask & iqr_mask
        return df[final_mask], final_mask
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train all models with preprocessing"""
        # Clean and preprocess
        X_clean, anomaly_mask = self.detect_anomalies(X)
        y_clean = y[anomaly_mask]
        X_processed, preprocess_stats = self.preprocess(X_clean)
        
        # Train each model
        for model_name, model in self.models.items():
            model.fit(X_processed, y_clean)
            logger.info(f"Trained {model_name} on {len(X_processed)} samples")
            
        # Save checkpoint
        self.save_checkpoint()
        
    def predict(self, X: pd.DataFrame) -> Tuple[float, float, Dict]:
        """Make prediction with confidence and diagnostics"""
        X_processed, preprocess_stats = self.preprocess(X)
        
        predictions = {}
        model_diagnostics = {}
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X_processed)[0]
                predictions[model_name] = pred
                model_diagnostics[model_name] = {
                    'weight': self.model_weights[model_name],
                    'prediction': pred
                }
            except Exception as e:
                logger.error(f"Error in {model_name} prediction: {str(e)}")
                predictions[model_name] = None
                
        # Filter out failed predictions
        valid_predictions = {k: v for k, v in predictions.items() if v is not None}
        
        if not valid_predictions:
            raise ValueError("All models failed to make predictions")
            
        # Calculate weighted prediction
        final_pred = sum(
            valid_predictions[m] * self.model_weights[m] 
            for m in valid_predictions
        ) / sum(self.model_weights[m] for m in valid_predictions)
        
        # Calculate confidence
        confidence = self.compute_confidence(list(valid_predictions.values()))
        
        # Update prediction count
        self.prediction_count += 1
        
        # Auto-save checkpoint if needed
        if self.prediction_count % self.config.config['checkpointing']['save_interval'] == 0:
            self.save_checkpoint()
            
        return final_pred, confidence, {
            'model_predictions': model_diagnostics,
            'preprocessing_stats': preprocess_stats
        }
        
    def compute_confidence(self, predictions: List[float]) -> float:
        """Enhanced confidence calculation"""
        if not predictions:
            return 0.0
            
        # Calculate various metrics
        variance = np.var(predictions)
        range_ratio = (max(predictions) - min(predictions)) / np.mean(predictions)
        agreement_score = 1.0 - (range_ratio / 2.0)  # Higher agreement = lower range ratio
        
        # Combine metrics
        confidence = max(0.0, min(1.0, (
            (1.0 - variance) * 0.6 +  # Weight variance more heavily
            agreement_score * 0.4
        )))
        
        return confidence
        
    def reward_feedback(self, y_true: float, y_pred: float, metadata: Optional[Dict] = None):
        """Process prediction feedback with metadata"""
        error = abs(y_true - y_pred)
        error_threshold = self.config.config['memory_bank']['error_threshold']
        
        feedback_data = {
            'true_value': y_true,
            'predicted_value': y_pred,
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        if error > error_threshold:
            self.memory_bank.append(feedback_data)
            logger.warning(f"Bad prediction stored (error: {error:.4f})")
            
            # Trigger weight adjustment
            self.self_rewrite_weights()
            
            # Consider rollback if error is very large
            if error > error_threshold * 2:
                self.auto_rollback()
        else:
            logger.info(f"Good prediction (error: {error:.4f})")
            
        # Maintain memory bank size
        if len(self.memory_bank) > self.config.config['memory_bank']['max_size']:
            self.memory_bank.pop(0)
            
    def self_rewrite_weights(self):
        """Adaptive weight adjustment based on performance"""
        if not self.memory_bank:
            return
            
        config = self.config.config['adaptation']
        adjustment_rate = config['weight_adjustment_rate']
        
        # Calculate recent performance for each model
        recent_errors = {}
        for model_name in self.models:
            predictions = [
                entry['metadata'].get('model_predictions', {}).get(model_name, {}).get('prediction')
                for entry in self.memory_bank[-10:]  # Look at last 10 predictions
                if entry['metadata'].get('model_predictions', {}).get(model_name) is not None
            ]
            
            if predictions:
                true_values = [entry['true_value'] for entry in self.memory_bank[-10:]]
                error = mean_absolute_error(true_values[:len(predictions)], predictions)
                recent_errors[model_name] = error
                
        if recent_errors:
            # Normalize errors to get adjustment factors
            total_error = sum(recent_errors.values())
            adjustments = {
                model: -adjustment_rate * (error / total_error)
                for model, error in recent_errors.items()
            }
            
            # Apply adjustments
            for model_name in self.models:
                if model_name in adjustments:
                    current_weight = self.model_weights[model_name]
                    new_weight = np.clip(
                        current_weight + adjustments[model_name],
                        config['min_weight'],
                        config['max_weight']
                    )
                    self.model_weights[model_name] = new_weight
                    
            logger.info(f"Updated model weights: {self.model_weights}")
            
    def save_checkpoint(self):
        """Save checkpoint with metadata"""
        checkpoint = {
            "model_weights": self.model_weights,
            "timestamp": datetime.now().isoformat(),
            "prediction_count": self.prediction_count,
            "performance_metrics": {
                name: np.mean(metrics) if metrics else None
                for name, metrics in self.model_performance.items()
            }
        }
        
        # Save to file
        checkpoint_path = f'checkpoint_{self.prediction_count}.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
            
        # Add to checkpoints list
        self.checkpoints.append(checkpoint_path)
        
        # Update best checkpoint if needed
        if not self.best_checkpoint or checkpoint['performance_metrics'] > self.best_checkpoint['performance_metrics']:
            self.best_checkpoint = checkpoint
            
        # Maintain checkpoint limit
        max_checkpoints = self.config.config['checkpointing']['max_checkpoints']
        while len(self.checkpoints) > max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
    def auto_rollback(self):
        """Smart rollback to best performing checkpoint"""
        if not self.best_checkpoint:
            logger.warning("No checkpoint available for rollback")
            return
            
        # Check if recent performance is significantly worse
        recent_performance = np.mean([entry['error'] for entry in self.memory_bank[-5:]])
        best_performance = np.mean(list(self.best_checkpoint['performance_metrics'].values()))
        
        if recent_performance > best_performance * 1.5:  # 50% worse performance
            self.model_weights = self.best_checkpoint['model_weights']
            logger.warning(f"Rolled back to checkpoint from {self.best_checkpoint['timestamp']}")
            
    async def run_prediction_cycle(self,
                                 forex_currencies: List[str],
                                 sports_list: List[str],
                                 stock_symbols: List[str]) -> Dict:
        """Run complete prediction cycle with real-time data"""
        try:
            # Fetch real-time data
            df = await self.fetch_real_time_data(
                forex_currencies=forex_currencies,
                sports_list=sports_list,
                stock_symbols=stock_symbols
            )
            
            # Make prediction
            prediction, confidence, diagnostics = self.predict(df)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'diagnostics': diagnostics,
                'data_snapshot': df.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error in prediction cycle: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize predictor
        predictor = RichPredictionAGI()
        
        # Example prediction cycle
        result = await predictor.run_prediction_cycle(
            forex_currencies=['USD', 'EUR'],
            sports_list=['nba'],
            stock_symbols=['AAPL', 'GOOGL']
        )
        
        print("Prediction result:", result)
        
    asyncio.run(main()) 
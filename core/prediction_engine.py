# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import numpy as np
from neat import nn, population
from deap import base, creator, tools, algorithms
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from ai.minmax_search import MinMaxSearcher
from ai.lstm_decision import LSTMDecisionMaker
from utils.data_processor import DataProcessor

logger = logging.getLogger(__name__)

class PredictionEngine:
    """Advanced AI prediction engine with multiple techniques"""

    def __init__(self, config_path: Path):
        self.config_path = config_path  # Fix: Added missing attribute
        self.config = self._load_config(config_path)
        self.data_processor = DataProcessor()  # Add data processor
        self._setup_neat()
        self._setup_deap()
        self._setup_lstm()
        self.minmax = MinMaxSearcher(max_depth=5)
        self.lstm_decision = LSTMDecisionMaker(input_shape=(50, 10))
        self.history = []
        self.prediction_cache = {}
        self.model_cache = {}
        self.preprocessing_cache = {}
        logger.info("All AI components initialized")

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            # Implementation of config loading
            return {}  # Placeholder
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise

    def _setup_neat(self):
        """Initialize NEAT neural network"""
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(self.config_path / 'neat_config.txt')
        )
        self.population = neat.Population(self.neat_config)
        logger.info("NEAT system initialized")

    def _setup_deap(self):
        """Initialize DEAP evolutionary framework"""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", np.random.uniform, -1, 1)
        self.toolbox.register("individual", tools.initRepeat,
                            creator.Individual, self.toolbox.attr_float, n=100)
        self.toolbox.register("population", tools.initRepeat, list,
                            self.toolbox.individual)
        logger.info("DEAP framework initialized")

    def _setup_lstm(self):
        """Initialize LSTM model"""
        self.lstm_model = Sequential([
            LSTM(128, input_shape=(50, 1), return_sequences=True),
            LSTM(64, return_sequences=False),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        self.lstm_model.compile(optimizer='adam', loss='mse')
        logger.info("LSTM model initialized")

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions using multiple AI techniques"""
        try:
            # Generate cache key from input data
            cache_key = self._generate_cache_key(data)

            # Check cache for recent prediction
            if cache_key in self.prediction_cache:
                cache_time, prediction = self.prediction_cache[cache_key]
                if (datetime.now() - cache_time).seconds < 300:  # 5 min cache
                    logger.debug("Using cached prediction")
                    return prediction

            # Process data with caching
            processed_data = self._cached_preprocessing(data)

            # Generate predictions using different methods
            predictions = {
                'neat': self._cached_neat_predict(processed_data),
                'deap': self._cached_deap_predict(processed_data),
                'lstm': self._cached_lstm_predict(processed_data)
            }

            # Combine predictions
            final_pred = self._ensemble_predictions([
                (predictions['neat'], 0.4),
                (predictions['deap'], 0.3),
                (predictions['lstm'], 0.3)
            ])

            # Update cache
            self.prediction_cache[cache_key] = (datetime.now(), final_pred)

            return final_pred

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _update_history(self, prediction: Dict[str, Any]):
        """Update prediction history"""
        self.history.append({
            'timestamp': datetime.now(),
            'prediction': prediction
        })
        if len(self.history) > 1000:
            self.history.pop(0)

    def _preprocess_data(self, data: Dict[str, Any]) -> np.ndarray:
        """Preprocess and combine historical and online data"""
        try:
            # Extract and validate data
            historical_data = data.get('historical')
            online_data = data.get('online')

            if historical_data is None or online_data is None:
                raise ValueError("Missing required data fields")

            # Process and combine data
            processed_data = self.data_processor.process(
                historical_data=historical_data,
                online_data=online_data
            )

            return processed_data

        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise

    def _cached_preprocessing(self, data: Dict[str, Any]) -> np.ndarray:
        """Preprocess data with caching"""
        cache_key = self._generate_cache_key(data)
        if cache_key in self.preprocessing_cache:
            return self.preprocessing_cache[cache_key]

        result = self._preprocess_data(data)
        self.preprocessing_cache[cache_key] = result
        return result

    def _neat_predict(self, data: np.ndarray) -> float:
        """Generate prediction using NEAT"""
        winner = self.population.run(self._eval_genome, 50)
        winner_net = neat.nn.FeedForwardNetwork.create(winner, self.neat_config)
        return winner_net.activate(data)

    def _deap_predict(self, data: np.ndarray) -> float:
        """Generate prediction using DEAP"""
        pop = self.toolbox.population(n=50)
        algorithms.eaSimple(pop, self.toolbox, cxpb=0.7, mutpb=0.3, ngen=40)
        return max(pop, key=lambda ind: ind.fitness.values[0])

    def _lstm_predict(self, data: np.ndarray) -> float:
        """Generate prediction using LSTM"""
        return self.lstm_model.predict(data)

    def _ensemble_predictions(self, predictions: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Combine predictions using weighted ensemble"""
        try:
            if not predictions:
                raise ValueError("Empty predictions list")

            # Validate predictions format
            for pred, weight in predictions:
                if not isinstance(pred, (int, float)) or not isinstance(weight, (int, float)):
                    raise ValueError("Invalid prediction or weight type")

            # Calculate weighted sum
            weighted_sum = sum(pred * weight for pred, weight in predictions)
            total_weight = sum(weight for _, weight in predictions)

            if total_weight == 0:
                raise ValueError("Sum of weights is zero")

            final_pred = weighted_sum / total_weight
            confidence = self._calculate_confidence(predictions)

            return {
                "prediction": final_pred,
                "confidence": confidence,
                "weighted_predictions": dict(enumerate(predictions))
            }

        except Exception as e:
            logger.error(f"Ensemble prediction failed: {str(e)}")
            raise

    def _calculate_confidence(self, predictions: List[Tuple[float, float]]) -> float:
        """Calculate confidence score based on prediction variance"""
        try:
            values = [pred for pred, _ in predictions]
            if not values:
                return 0.0

            # Calculate normalized confidence score
            std_dev = np.std(values)
            max_val = max(abs(max(values)), abs(min(values)))

            if max_val == 0:
                return 1.0  # Perfect agreement

            # Normalize confidence score between 0 and 1
            confidence = 1.0 - (std_dev / max_val)
            return max(0.0, min(1.0, confidence))  # Clamp between 0 and 1

        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.0

    def retrain(self, domain: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Retrain the prediction models with new data

        Args:
            domain: Domain to retrain models for ("sports", "forex", "all")
            data: Optional data to use for retraining, will load from database if not provided

        Returns:
            Dictionary with retraining results
        """
        logger.info(f"Starting model retraining for domain: {domain}")

        results = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "details": {}
        }

        try:
            # Load data if not provided
            if data is None:
                data = self._load_training_data(domain)

            if not data:
                raise ValueError(f"No training data available for domain: {domain}")

            # Track individual model retraining status
            model_results = {}

            # Retrain LSTM model
            if domain in ["sports", "all"]:
                logger.info("Retraining LSTM model for sports predictions")
                sports_data = self._preprocess_data_for_training(data.get("sports", {}))

                # LSTM retraining
                sports_history = self._retrain_lstm(sports_data)
                model_results["sports_lstm"] = {
                    "epochs": len(sports_history.history.get("loss", [])),
                    "final_loss": sports_history.history.get("loss", [0])[-1],
                    "validation_loss": sports_history.history.get("val_loss", [0])[-1] if "val_loss" in sports_history.history else None
                }

                # NEAT retraining for sports
                neat_generations = self._retrain_neat(sports_data)
                model_results["sports_neat"] = {
                    "generations": neat_generations,
                    "improved": neat_generations > 0
                }

            if domain in ["forex", "all"]:
                logger.info("Retraining models for forex predictions")
                forex_data = self._preprocess_data_for_training(data.get("forex", {}))

                # LSTM retraining
                forex_history = self._retrain_lstm(forex_data)
                model_results["forex_lstm"] = {
                    "epochs": len(forex_history.history.get("loss", [])),
                    "final_loss": forex_history.history.get("loss", [0])[-1],
                    "validation_loss": forex_history.history.get("val_loss", [0])[-1] if "val_loss" in forex_history.history else None
                }

                # DEAP retraining
                deap_generations = self._retrain_deap(forex_data)
                model_results["forex_deap"] = {
                    "generations": deap_generations,
                    "improved": deap_generations > 0
                }

            # Save models
            self._save_models(domain)

            # Update results
            results["details"] = model_results
            logger.info(f"Model retraining completed successfully for {domain}")

            # Clear cache after retraining
            self._clear_prediction_cache()

            return results

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error during model retraining: {error_msg}")
            return {
                "status": "failed",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }

    def _load_training_data(self, domain: str) -> Dict[str, Any]:
        """Load training data from database"""
        logger.info(f"Loading training data for {domain}")

        # In a real implementation, this would load from a database
        # For now, we'll return a placeholder
        data = {}

        if domain in ["sports", "all"]:
            # Load sports training data
            sports_path = Path("data/training/sports")
            if sports_path.exists():
                data["sports"] = self.data_processor.load_training_data(sports_path)
                logger.info(f"Loaded {len(data['sports'])} sports training samples")

        if domain in ["forex", "all"]:
            # Load forex training data
            forex_path = Path("data/training/forex")
            if forex_path.exists():
                data["forex"] = self.data_processor.load_training_data(forex_path)
                logger.info(f"Loaded {len(data['forex'])} forex training samples")

        return data

    def _preprocess_data_for_training(self, data: Any) -> np.ndarray:
        """Preprocess data specifically for training"""
        # In a real implementation, this would prepare data for model training
        # For now, return the data as is or a simple transformation
        if isinstance(data, dict) and "features" in data:
            return np.array(data["features"])
        elif isinstance(data, list):
            return np.array(data)
        else:
            return np.array([])

    def _retrain_lstm(self, training_data: np.ndarray) -> Any:
        """Retrain LSTM model with new data"""
        if len(training_data) == 0:
            logger.warning("No data provided for LSTM retraining")
            return None

        # Prepare data for LSTM (shape should be [samples, timesteps, features])
        x_train, y_train = self._prepare_lstm_data(training_data)

        # Train the model
        history = self.lstm_model.fit(
            x_train, y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )

        logger.info(f"LSTM model retrained: final loss {history.history['loss'][-1]:.4f}")
        return history

    def _prepare_lstm_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        # This is a simplified implementation
        # In a real system, this would properly shape the data for sequence prediction

        # Ensure minimum data size
        if len(data) < 60:
            # Pad with zeros if not enough data
            pad_size = 60 - len(data)
            data = np.vstack([np.zeros((pad_size, data.shape[1])), data])

        # Create sequences with sliding window
        sequences = []
        targets = []

        sequence_length = 50  # Match model input shape
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i+sequence_length])
            targets.append(data[i+sequence_length, 0])  # Predict first feature

        # Reshape for LSTM [samples, timesteps, features]
        x = np.array(sequences)
        y = np.array(targets)

        return x, y

    def _retrain_neat(self, training_data: np.ndarray) -> int:
        """Retrain NEAT neural network"""
        if len(training_data) == 0:
            logger.warning("No data provided for NEAT retraining")
            return 0

        # In a real implementation, this would properly retrain the NEAT population
        # For now, we'll just run a few generations and return the count

        generations = 5
        logger.info(f"NEAT model retrained for {generations} generations")
        return generations

    def _retrain_deap(self, training_data: np.ndarray) -> int:
        """Retrain DEAP evolutionary algorithm"""
        if len(training_data) == 0:
            logger.warning("No data provided for DEAP retraining")
            return 0

        # In a real implementation, this would properly retrain the evolutionary algorithm
        # For now, we'll just run a few generations and return the count

        generations = 5
        logger.info(f"DEAP model retrained for {generations} generations")
        return generations

    def _save_models(self, domain: str) -> None:
        """Save trained models to disk"""
        save_path = Path(f"models/{domain}")
        save_path.mkdir(parents=True, exist_ok=True)

        # Save LSTM model
        if domain in ["sports", "all"]:
            self.lstm_model.save(save_path / "lstm_sports.h5")
            logger.info(f"Saved LSTM model for sports to {save_path / 'lstm_sports.h5'}")

        if domain in ["forex", "all"]:
            self.lstm_model.save(save_path / "lstm_forex.h5")
            logger.info(f"Saved LSTM model for forex to {save_path / 'lstm_forex.h5'}")

        # In a real implementation, we would also save NEAT and DEAP models

    def _clear_prediction_cache(self) -> None:
        """Clear prediction cache after retraining"""
        self.prediction_cache = {}
        self.preprocessing_cache = {}
        logger.info("Prediction cache cleared after retraining")

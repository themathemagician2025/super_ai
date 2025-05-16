# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class LSTMDecisionMaker:
    """Advanced LSTM model for pattern recognition and decision making"""

    def __init__(self, input_shape: Tuple[int, int], learning_rate: float = 0.001):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.history = []
        logger.info("LSTM Decision Maker initialized")

    def _build_model(self) -> Sequential:
        """Build LSTM neural network architecture"""
        model = Sequential([
            LSTM(128, input_shape=self.input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X: np.ndarray, y: np.ndarray,
             validation_split: float = 0.2,
             epochs: int = 100,
             batch_size: int = 32) -> Dict[str, List[float]]:
        """Train the LSTM model on historical data"""
        try:
            history = self.model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )

            self.history.append({
                'timestamp': np.datetime64('now'),
                'metrics': history.history
            })

            logger.info(f"Training completed. Final loss: {history.history['loss'][-1]:.4f}")
            return history.history

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def predict(self, state: np.ndarray) -> Tuple[float, float]:
        """Generate prediction with confidence score"""
        try:
            prediction = self.model.predict(state, verbose=0)[0][0]
            confidence = self._calculate_confidence(prediction)

            logger.info(f"Generated prediction: {prediction:.4f} "
                       f"(confidence: {confidence:.4f})")
            return prediction, confidence

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _calculate_confidence(self, prediction: float) -> float:
        """Calculate confidence score based on prediction certainty"""
        return 1 - 2 * abs(0.5 - prediction)

    def analyze_patterns(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze patterns in the data using LSTM."""
        predictions = self.model.predict(data)
        return [{"pattern": "detected", "confidence": float(pred)} for pred in predictions]

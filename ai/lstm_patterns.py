# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LSTMPatternRecognizer:
    """Deep learning pattern recognition using LSTM"""

    def __init__(self, sequence_length: int = 50, features: int = 10):
        self.sequence_length = sequence_length
        self.features = features
        self.model = self._build_model()
        logger.info("LSTM pattern recognizer initialized")

    def _build_model(self) -> Sequential:
        """Build LSTM neural network"""
        model = Sequential([
            LSTM(128, input_shape=(self.sequence_length, self.features),
                 return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model

    def find_patterns(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Extract patterns from time series data"""
        try:
            # Normalize data
            normalized = self._normalize_data(data)

            # Detect patterns using sliding window
            patterns = []
            for i in range(len(normalized) - self.sequence_length):
                sequence = normalized[i:i+self.sequence_length]
                prediction = self.model.predict(sequence.reshape(1, -1, self.features))
                if prediction > 0.8:  # Pattern threshold
                    patterns.append({
                        'start_idx': i,
                        'end_idx': i + self.sequence_length,
                        'confidence': float(prediction),
                        'type': self._classify_pattern(sequence)
                    })

            logger.info(f"Found {len(patterns)} patterns in data")
            return patterns

        except Exception as e:
            logger.error(f"Pattern recognition failed: {str(e)}")
            raise

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize input data"""
        return (data - np.mean(data)) / np.std(data)

    def _classify_pattern(self, sequence: np.ndarray) -> str:
        """Classify pattern type using Fibonacci ratios"""
        try:
            # Fibonacci ratios for pattern recognition
            fib_ratios = [0.236, 0.382, 0.500, 0.618, 0.786]

            # Calculate pattern characteristics
            high = np.max(sequence)
            low = np.min(sequence)
            range_size = high - low

            # Check for Fibonacci retracements
            retracements = []
            for ratio in fib_ratios:
                level = high - (range_size * ratio)
                if np.any(np.isclose(sequence, level, rtol=0.01)):
                    retracements.append(ratio)

            if len(retracements) >= 2:
                return f"fibonacci_pattern_{retracements}"
            return "unknown_pattern"

        except Exception as e:
            logger.error(f"Pattern classification failed: {str(e)}")
            return "classification_error"

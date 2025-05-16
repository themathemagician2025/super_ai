# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from pathlib import Path
import random

logger = logging.getLogger(__name__)

class DecisionTree:
    """A simple decision tree implementation for the RandomForest"""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_indices = None
        self.split_feature = None
        self.split_value = None
        self.split_score = None
        self.left = None
        self.right = None
        self.prediction = None

    def fit(self, X: np.ndarray, y: np.ndarray, depth: int = 0, features_subset: List[int] = None):
        """Fit the decision tree to the data"""
        n_samples, n_features = X.shape

        # Store the subset of features considered at this node
        self.feature_indices = features_subset if features_subset is not None else list(range(n_features))

        # Base case 1: Max depth reached
        if depth >= self.max_depth:
            self.prediction = self._calculate_leaf_value(y)
            return

        # Base case 2: Not enough samples to split
        if n_samples < self.min_samples_split:
            self.prediction = self._calculate_leaf_value(y)
            return

        # Base case 3: Pure node (all samples have the same target)
        if len(np.unique(y)) == 1:
            self.prediction = self._calculate_leaf_value(y)
            return

        # Find the best split
        self.split_feature, self.split_value, self.split_score = self._find_best_split(X, y)

        # If no good split is found, make this a leaf node
        if self.split_feature is None:
            self.prediction = self._calculate_leaf_value(y)
            return

        # Split the data
        left_indices = X[:, self.split_feature] <= self.split_value
        right_indices = ~left_indices

        # If either split is empty, make this a leaf node
        if sum(left_indices) == 0 or sum(right_indices) == 0:
            self.prediction = self._calculate_leaf_value(y)
            return

        # Create child nodes
        self.left = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        self.right = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)

        # Recursively build the left and right subtrees
        self.left.fit(X[left_indices], y[left_indices], depth + 1, self.feature_indices)
        self.right.fit(X[right_indices], y[right_indices], depth + 1, self.feature_indices)

    def _calculate_leaf_value(self, y: np.ndarray) -> float:
        """Calculate the prediction value for a leaf node"""
        # For regression, return the mean
        return float(np.mean(y))

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], Optional[float]]:
        """Find the best feature and value to split on"""
        best_feature = None
        best_value = None
        best_score = float('-inf')

        # Random subset of features for this split (feature bagging)
        n_features = len(self.feature_indices)
        n_subset = max(1, int(np.sqrt(n_features)))
        feature_subset = random.sample(self.feature_indices, n_subset)

        for feature_idx in feature_subset:
            feature_values = X[:, feature_idx]
            possible_splits = np.unique(feature_values)

            # Skip if there's only one unique value
            if len(possible_splits) <= 1:
                continue

            # Try different split points for this feature
            for value in possible_splits:
                left_indices = feature_values <= value
                right_indices = ~left_indices

                # Skip if one split is empty
                if sum(left_indices) == 0 or sum(right_indices) == 0:
                    continue

                # Calculate the score for this split
                score = self._calculate_split_score(y, y[left_indices], y[right_indices])

                if score > best_score:
                    best_score = score
                    best_feature = feature_idx
                    best_value = value

        return best_feature, best_value, best_score

    def _calculate_split_score(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Calculate the score for a potential split (reduction in variance for regression)"""
        # Calculate variance before split
        var_before = np.var(y) * len(y)

        # Calculate variance after split
        var_after = np.var(y_left) * len(y_left) + np.var(y_right) * len(y_right)

        # Return reduction in variance
        return var_before - var_after

    def predict_sample(self, sample: np.ndarray) -> float:
        """Predict the target value for a single sample"""
        if self.prediction is not None:  # Leaf node
            return self.prediction

        # Navigate to the appropriate child node
        if sample[self.split_feature] <= self.split_value:
            return self.left.predict_sample(sample)
        else:
            return self.right.predict_sample(sample)


class RandomForestPredictor:
    """Random Forest implementation for prediction tasks"""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 2, max_features: str = 'sqrt',
                 bootstrap: bool = True, random_state: int = None,
                 config_path: Path = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []
        self.feature_importances_ = None

        # Set random seed if provided
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        # Load configuration if provided
        if config_path:
            self._load_config(config_path)

        logger.info(f"RandomForest initialized with {n_estimators} trees")

    def _load_config(self, config_path: Path):
        """Load configuration from file"""
        try:
            import yaml
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            if config:
                self.n_estimators = config.get('n_estimators', self.n_estimators)
                self.max_depth = config.get('max_depth', self.max_depth)
                self.min_samples_split = config.get('min_samples_split', self.min_samples_split)
                self.max_features = config.get('max_features', self.max_features)
                self.bootstrap = config.get('bootstrap', self.bootstrap)

                logger.info(f"Loaded RandomForest configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading RandomForest configuration: {str(e)}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Build the forest (fit the trees)"""
        n_samples, n_features = X.shape
        self.trees = []

        for i in range(self.n_estimators):
            # Create a new decision tree
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)

            # Bootstrap sampling if enabled
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
            else:
                X_bootstrap = X
                y_bootstrap = y

            # Fit the tree
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

            if (i + 1) % 10 == 0 or i == self.n_estimators - 1:
                logger.info(f"Built {i + 1}/{self.n_estimators} trees")

        logger.info("RandomForest training completed")
        return self

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction using the random forest"""
        try:
            # Convert input data to feature vector
            X = self._extract_features(input_data)

            if not self.trees:
                logger.warning("RandomForest has not been trained yet")
                return {}

            # Make predictions with all trees
            predictions = np.array([tree.predict_sample(X) for tree in self.trees])

            # Aggregate predictions (mean for regression)
            prediction_value = float(np.mean(predictions))
            prediction_std = float(np.std(predictions))

            # Create result dictionary
            result = {
                "value": prediction_value,
                "uncertainty": prediction_std,
                "confidence": 1.0 / (1.0 + prediction_std),  # Simple confidence measure
            }

            # Add domain-specific predictions
            if "forex" in input_data or "market" in input_data:
                # For forex/market predictions
                direction = "up" if prediction_value > 0 else "down"
                strength = min(10, abs(prediction_value) * 2)  # Scale to 0-10

                result.update({
                    "trend": {
                        "direction": direction,
                        "strength": strength,
                        "reliability": 1.0 - (prediction_std / (abs(prediction_value) + 1e-6))
                    }
                })

            if "betting" in input_data or "sport" in input_data:
                # For betting/sports predictions
                # Convert to probabilities using a logistic function
                prob_home = 1.0 / (1.0 + np.exp(-prediction_value))
                prob_away = 1.0 - prob_home
                prob_draw = 0.1  # Simplified placeholder

                # Normalize to sum to 1
                total = prob_home + prob_away + prob_draw
                prob_home /= total
                prob_away /= total
                prob_draw /= total

                result.update({
                    "match_result": {
                        "home_win": float(prob_home),
                        "draw": float(prob_draw),
                        "away_win": float(prob_away)
                    }
                })

            return result
        except Exception as e:
            logger.error(f"Error making prediction with RandomForest: {str(e)}")
            return {}

    def _extract_features(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from input data"""
        features = []

        # Extract from historical data
        if "historical" in input_data and isinstance(input_data["historical"], list):
            # Get recent values (e.g., closing prices, volumes)
            values = []
            for item in input_data["historical"]:
                if isinstance(item, (int, float)):
                    values.append(item)
                elif isinstance(item, dict) and "value" in item:
                    values.append(item["value"])

            # Add recent values
            features.extend(values[-min(5, len(values)):])

            # Add some simple technical indicators if enough data
            if len(values) >= 10:
                # Moving averages
                ma5 = np.mean(values[-5:])
                ma10 = np.mean(values[-10:])
                features.extend([ma5, ma10, ma5/ma10])

                # Volatility (standard deviation)
                vol5 = np.std(values[-5:])
                features.append(vol5)

                # Momentum (rate of change)
                momentum = values[-1] / values[-5] - 1 if values[-5] != 0 else 0
                features.append(momentum)

        # Extract from online data
        if "online" in input_data and isinstance(input_data["online"], dict):
            # Extract numerical values
            for key, value in input_data["online"].items():
                if isinstance(value, (int, float)):
                    features.append(value)

        # Extract from sentiment data
        if "sentiment" in input_data and isinstance(input_data["sentiment"], dict):
            if "score" in input_data["sentiment"]:
                features.append(input_data["sentiment"]["score"])

            # Extract sentiment components if available
            for key in ["positive", "negative", "neutral"]:
                if key in input_data["sentiment"]:
                    features.append(input_data["sentiment"][key])

        # Ensure we have at least one feature
        if not features:
            features = [0.0]

        return np.array(features).reshape(1, -1)[0]

    def update(self):
        """Update the random forest model (retrain or add more trees)"""
        # In a real implementation, this would update the model with new data
        # For this example, we'll just log a message
        logger.info("RandomForest model would be updated with new data here")
        return True

    def feature_importance(self) -> Dict[int, float]:
        """Calculate and return feature importance"""
        # This is a simplified placeholder - real RF would compute this from trees
        if not self.trees or not self.feature_importances_:
            return {}

        return {i: imp for i, imp in enumerate(self.feature_importances_)}

    def save(self, file_path: Path):
        """Save the random forest model to a file"""
        try:
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"RandomForest model saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving RandomForest model: {str(e)}")
            return False

    @classmethod
    def load(cls, file_path: Path):
        """Load a random forest model from a file"""
        try:
            import pickle
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"RandomForest model loaded from {file_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading RandomForest model: {str(e)}")
            return None

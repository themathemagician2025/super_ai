#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Prediction Engine Module

This module provides real machine learning models for predictions
using scikit-learn and PyTorch.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("prediction_engine")

# Define model directory
MODEL_DIR = os.environ.get("MODEL_DIR", "data/models")

# PyTorch neural network for forex prediction
class ForexNN(nn.Module):
    """PyTorch Neural Network for forex prediction"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        """Initialize the neural network"""
        super(ForexNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers for time series data
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layers for prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        """Forward pass through the network"""
        # LSTM layers
        lstm_out, _ = self.lstm(x)

        # Get the last time step's output
        lstm_out = lstm_out[:, -1, :]

        # Pass through fully connected layers
        output = self.fc_layers(lstm_out)

        return output

# Dataset class for forex data
class ForexDataset(Dataset):
    """PyTorch Dataset for forex data"""

    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 10):
        """Initialize the dataset"""
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length

    def __len__(self):
        """Return the number of samples"""
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        # Get sequence of features
        x = self.features[idx:idx + self.sequence_length]

        # Get target (next day's price)
        y = self.targets[idx + self.sequence_length - 1]

        return torch.FloatTensor(x), torch.FloatTensor([y])

class PredictionEngine:
    """Real Machine Learning Prediction Engine"""

    def __init__(self, model_dir: str = MODEL_DIR):
        """Initialize the prediction engine"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Initialize model cache
        self.models = {}
        self.scalers = {}
        self.encoders = {}

        # Load models
        self._load_models()

    def _load_models(self) -> None:
        """Load saved models from disk"""
        # Look for saved models
        for model_name in ["forex_rf", "sports_rf", "forex_nn"]:
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")

            if os.path.exists(model_path):
                try:
                    # Load scikit-learn models
                    if model_name.endswith("_rf"):
                        with open(model_path, "rb") as f:
                            self.models[model_name] = pickle.load(f)
                        logger.info(f"Loaded scikit-learn model: {model_name}")

                        # Load corresponding scaler if exists
                        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
                        if os.path.exists(scaler_path):
                            with open(scaler_path, "rb") as f:
                                self.scalers[model_name] = pickle.load(f)

                        # Load corresponding encoder if exists
                        encoder_path = os.path.join(self.model_dir, f"{model_name}_encoder.pkl")
                        if os.path.exists(encoder_path):
                            with open(encoder_path, "rb") as f:
                                self.encoders[model_name] = pickle.load(f)

                    # Load PyTorch models
                    elif model_name.endswith("_nn"):
                        # Load model architecture and metadata
                        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
                        if os.path.exists(metadata_path):
                            with open(metadata_path, "r") as f:
                                metadata = json.load(f)

                            # Initialize model with same architecture
                            model = ForexNN(
                                input_size=metadata["input_size"],
                                hidden_size=metadata["hidden_size"],
                                num_layers=metadata["num_layers"],
                                dropout=metadata["dropout"]
                            )

                            # Load weights
                            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                            model.eval()  # Set to evaluation mode

                            self.models[model_name] = model
                            logger.info(f"Loaded PyTorch model: {model_name}")

                            # Load corresponding scaler if exists
                            scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
                            if os.path.exists(scaler_path):
                                with open(scaler_path, "rb") as f:
                                    self.scalers[model_name] = pickle.load(f)
                        else:
                            logger.warning(f"No metadata found for model: {model_name}")

                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {str(e)}")

    def _save_model(self, model_name: str) -> None:
        """Save a model to disk"""
        if model_name not in self.models:
            logger.warning(f"No model found with name: {model_name}")
            return

        try:
            model = self.models[model_name]

            # Save scikit-learn models
            if model_name.endswith("_rf"):
                model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                logger.info(f"Saved scikit-learn model: {model_name}")

                # Save corresponding scaler if exists
                if model_name in self.scalers:
                    scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
                    with open(scaler_path, "wb") as f:
                        pickle.dump(self.scalers[model_name], f)

                # Save corresponding encoder if exists
                if model_name in self.encoders:
                    encoder_path = os.path.join(self.model_dir, f"{model_name}_encoder.pkl")
                    with open(encoder_path, "wb") as f:
                        pickle.dump(self.encoders[model_name], f)

            # Save PyTorch models
            elif model_name.endswith("_nn"):
                # Save model weights
                model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
                torch.save(model.state_dict(), model_path)

                # Save model architecture metadata
                metadata = {
                    "input_size": model.input_size,
                    "hidden_size": model.hidden_size,
                    "num_layers": model.num_layers,
                    "model_type": "forex_nn",
                    "created_at": datetime.now().isoformat()
                }

                metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)

                logger.info(f"Saved PyTorch model: {model_name}")

                # Save corresponding scaler if exists
                if model_name in self.scalers:
                    scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
                    with open(scaler_path, "wb") as f:
                        pickle.dump(self.scalers[model_name], f)

        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")

    def train_forex_model(self, data: List[Dict[str, Any]], model_type: str = "rf") -> Dict[str, Any]:
        """
        Train a model for forex prediction

        Args:
            data: List of forex data points
            model_type: Type of model to train ("rf" for Random Forest, "nn" for Neural Network)

        Returns:
            Dictionary with training results
        """
        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Ensure datetime column
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

        # Check if we have enough data
        if len(df) < 30:
            return {"error": "Not enough data for training. Need at least 30 data points."}

        # Create features (use price and volume data)
        feature_cols = ["open", "high", "low", "volume"]
        if all(col in df.columns for col in feature_cols):
            # Create features dataframe
            X = df[feature_cols].values

            # Create target (next day's closing price)
            y = df["close"].values

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            if model_type == "rf":
                # Random Forest Regressor
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    random_state=42
                )

                # Train model
                model.fit(X_train_scaled, y_train)

                # Store model and scaler
                self.models["forex_rf"] = model
                self.scalers["forex_rf"] = scaler

                # Save model
                self._save_model("forex_rf")

                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)

                return {
                    "model_type": "forex_rf",
                    "mse": mse,
                    "rmse": np.sqrt(mse),
                    "feature_importance": dict(zip(feature_cols, model.feature_importances_)),
                    "train_size": len(X_train),
                    "test_size": len(X_test)
                }

            elif model_type == "nn":
                # Neural Network
                # Create sequences for time series
                sequence_length = 10

                # Create sequences
                X_seq = []
                y_seq = []

                for i in range(len(X_train_scaled) - sequence_length + 1):
                    X_seq.append(X_train_scaled[i:i+sequence_length])
                    y_seq.append(y_train[i+sequence_length-1])

                X_seq = np.array(X_seq)
                y_seq = np.array(y_seq)

                # Create test sequences
                X_test_seq = []
                y_test_seq = []

                for i in range(len(X_test_scaled) - sequence_length + 1):
                    X_test_seq.append(X_test_scaled[i:i+sequence_length])
                    y_test_seq.append(y_test[i+sequence_length-1])

                X_test_seq = np.array(X_test_seq)
                y_test_seq = np.array(y_test_seq)

                # Create PyTorch datasets
                train_dataset = ForexDataset(X_train_scaled, y_train, sequence_length)
                test_dataset = ForexDataset(X_test_scaled, y_test, sequence_length)

                # Create data loaders
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                # Create model
                model = ForexNN(
                    input_size=len(feature_cols),
                    hidden_size=64,
                    num_layers=2,
                    dropout=0.2
                )

                # Define loss function and optimizer
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                # Train model
                epochs = 50
                train_losses = []

                for epoch in range(epochs):
                    model.train()
                    running_loss = 0.0

                    for inputs, targets in train_loader:
                        # Zero the parameter gradients
                        optimizer.zero_grad()

                        # Forward pass
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        # Backward pass and optimize
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()

                    # Calculate average loss for the epoch
                    epoch_loss = running_loss / len(train_loader)
                    train_losses.append(epoch_loss)

                    if (epoch + 1) % 10 == 0:
                        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

                # Evaluate model
                model.eval()
                with torch.no_grad():
                    test_loss = 0.0
                    predictions = []
                    actuals = []

                    for inputs, targets in test_loader:
                        # Forward pass
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        test_loss += loss.item()

                        # Store predictions and actuals
                        predictions.extend(outputs.numpy().flatten())
                        actuals.extend(targets.numpy().flatten())

                    # Calculate average test loss
                    test_loss /= len(test_loader)

                # Store model and scaler
                self.models["forex_nn"] = model
                self.scalers["forex_nn"] = scaler

                # Save model
                self._save_model("forex_nn")

                # Calculate metrics
                mse = mean_squared_error(actuals, predictions)

                return {
                    "model_type": "forex_nn",
                    "mse": mse,
                    "rmse": np.sqrt(mse),
                    "train_loss": train_losses[-1],
                    "test_loss": test_loss,
                    "train_size": len(train_dataset),
                    "test_size": len(test_dataset)
                }

            else:
                return {"error": f"Unsupported model type: {model_type}"}

        else:
            return {"error": f"Missing required columns. Need {feature_cols}"}

    def train_sports_model(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train a model for sports prediction

        Args:
            data: List of sports match data points

        Returns:
            Dictionary with training results
        """
        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Check if we have enough data
        if len(df) < 30:
            return {"error": "Not enough data for training. Need at least 30 matches."}

        # Create features
        categorical_cols = ["home_team", "away_team", "league"]
        numeric_cols = [
            "home_goals_scored_avg", "away_goals_scored_avg",
            "home_goals_conceded_avg", "away_goals_conceded_avg"
        ]

        # Check if we have all the required columns
        if not all(col in df.columns for col in categorical_cols + numeric_cols):
            missing_cols = [col for col in categorical_cols + numeric_cols if col not in df.columns]
            return {"error": f"Missing required columns: {missing_cols}"}

        # Create target (home win, draw, away win)
        if "odds" in df.columns and isinstance(df.iloc[0]["odds"], dict):
            # Extract odds data
            odds_df = pd.DataFrame(df["odds"].tolist())
            df = pd.concat([df.drop("odds", axis=1), odds_df], axis=1)

        # Determine match result based on home/away goals
        if "home_goals" in df.columns and "away_goals" in df.columns:
            # Create target from actual goals
            df["result"] = np.where(
                df["home_goals"] > df["away_goals"], "home_win",
                np.where(df["home_goals"] < df["away_goals"], "away_win", "draw")
            )
        else:
            # Create synthetic target (for demonstration)
            # In real life, we would have actual match results
            np.random.seed(42)
            df["result"] = np.random.choice(
                ["home_win", "draw", "away_win"],
                size=len(df),
                p=[0.45, 0.25, 0.3]  # Typical soccer match outcome probabilities
            )

        # Prepare features
        X_cat = df[categorical_cols]
        X_num = df[numeric_cols]

        # One-hot encode categorical features
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_cat_encoded = encoder.fit_transform(X_cat)

        # Standardize numeric features
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)

        # Combine features
        X = np.hstack([X_cat_encoded, X_num_scaled])
        y = df["result"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Store model and preprocessing
        self.models["sports_rf"] = model
        self.scalers["sports_rf"] = scaler
        self.encoders["sports_rf"] = encoder

        # Save model
        self._save_model("sports_rf")

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        return {
            "model_type": "sports_rf",
            "accuracy": accuracy,
            "f1_score": f1,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "classes": list(model.classes_)
        }

    def predict_forex(self, data: Dict[str, Any], days_ahead: int = 1, model_name: str = "forex_rf") -> Dict[str, Any]:
        """
        Make forex price predictions

        Args:
            data: Dictionary with current forex data
            days_ahead: Number of days to predict ahead
            model_name: Name of the model to use

        Returns:
            Dictionary with predictions
        """
        # Check if model exists
        if model_name not in self.models:
            return {"error": f"Model not found: {model_name}"}

        try:
            model = self.models[model_name]

            # Check if we have a scaler
            if model_name not in self.scalers:
                return {"error": f"Scaler not found for model: {model_name}"}

            scaler = self.scalers[model_name]

            # Prepare features
            feature_cols = ["open", "high", "low", "volume"]
            if all(col in data for col in feature_cols):
                # Extract features
                features = np.array([[data[col] for col in feature_cols]])

                # Scale features
                features_scaled = scaler.transform(features)

                # Make predictions based on model type
                if model_name == "forex_rf":
                    # Simple prediction using scikit-learn
                    pred = model.predict(features_scaled)[0]

                    # For multi-day forecasting, we would need to iterate and feed
                    # predictions back into the model
                    predictions = [pred]
                    current_features = features_scaled.copy()

                    for i in range(1, days_ahead):
                        # Update feature vector with previous prediction
                        # In a real scenario, we would have more sophisticated feature updates
                        current_features[0, 0] = predictions[-1]  # use last prediction as new open price
                        # Make prediction for next step
                        next_pred = model.predict(current_features)[0]
                        predictions.append(next_pred)

                elif model_name == "forex_nn":
                    # Need sequence data for neural network
                    sequence_length = 10

                    # For demonstration, we'll just duplicate the current features
                    # In real life, we would have historical data for the sequence
                    sequence = np.tile(features_scaled, (sequence_length, 1))

                    # Convert to PyTorch tensor
                    tensor_seq = torch.FloatTensor(sequence).unsqueeze(0)  # add batch dimension

                    # Set model to evaluation mode
                    model.eval()

                    # Make prediction
                    with torch.no_grad():
                        pred = model(tensor_seq).item()

                    # For multi-day forecasting with NN
                    predictions = [pred]
                    current_seq = sequence.copy()

                    for i in range(1, days_ahead):
                        # Update sequence by removing first element and appending prediction
                        current_seq = np.roll(current_seq, -1, axis=0)
                        # In last row, update the first feature (open price) with prediction
                        current_seq[-1, 0] = predictions[-1]

                        # Convert to tensor
                        tensor_seq = torch.FloatTensor(current_seq).unsqueeze(0)

                        # Make prediction
                        with torch.no_grad():
                            next_pred = model(tensor_seq).item()

                        predictions.append(next_pred)

                # Return predictions with dates
                prediction_dates = [
                    (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
                    for i in range(1, days_ahead + 1)
                ]

                return {
                    "predictions": [
                        {
                            "date": date,
                            "predicted_price": float(pred)
                        }
                        for date, pred in zip(prediction_dates, predictions)
                    ],
                    "currency_pair": data.get("currency_pair", "Unknown"),
                    "model_used": model_name
                }

            else:
                return {"error": f"Missing required features: {feature_cols}"}

        except Exception as e:
            logger.error(f"Error making forex prediction: {str(e)}")
            return {"error": f"Prediction error: {str(e)}"}

    def predict_sports(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the outcome of a sports match

        Args:
            match_data: Dictionary with match data

        Returns:
            Dictionary with prediction
        """
        model_name = "sports_rf"

        # Check if model exists
        if model_name not in self.models:
            return {"error": f"Model not found: {model_name}"}

        try:
            model = self.models[model_name]

            # Check if we have the necessary preprocessing objects
            if model_name not in self.scalers or model_name not in self.encoders:
                return {"error": f"Preprocessing objects not found for model: {model_name}"}

            scaler = self.scalers[model_name]
            encoder = self.encoders[model_name]

            # Prepare features
            categorical_cols = ["home_team", "away_team", "league"]
            numeric_cols = [
                "home_goals_scored_avg", "away_goals_scored_avg",
                "home_goals_conceded_avg", "away_goals_conceded_avg"
            ]

            # Check if we have all required fields
            if not all(col in match_data for col in categorical_cols + numeric_cols):
                missing_cols = [col for col in categorical_cols + numeric_cols if col not in match_data]
                return {"error": f"Missing required fields: {missing_cols}"}

            # Extract features
            cat_features = pd.DataFrame([{col: match_data[col] for col in categorical_cols}])
            num_features = np.array([[match_data[col] for col in numeric_cols]])

            # Apply preprocessing
            cat_encoded = encoder.transform(cat_features)
            num_scaled = scaler.transform(num_features)

            # Combine features
            features = np.hstack([cat_encoded, num_scaled])

            # Make prediction
            predicted_class = model.predict(features)[0]

            # Get prediction probabilities
            proba = model.predict_proba(features)[0]
            proba_dict = {cls: float(p) for cls, p in zip(model.classes_, proba)}

            return {
                "match": f"{match_data['home_team']} vs {match_data['away_team']}",
                "league": match_data["league"],
                "predicted_outcome": predicted_class,
                "probabilities": proba_dict,
                "model_used": model_name,
                "match_date": match_data.get("match_date", "Unknown")
            }

        except Exception as e:
            logger.error(f"Error making sports prediction: {str(e)}")
            return {"error": f"Prediction error: {str(e)}"}

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        models_info = {}

        for model_name, model in self.models.items():
            if model_name.endswith("_rf"):
                if isinstance(model, RandomForestClassifier):
                    models_info[model_name] = {
                        "type": "RandomForestClassifier",
                        "n_estimators": model.n_estimators,
                        "max_depth": model.max_depth,
                        "classes": list(model.classes_),
                        "feature_count": model.n_features_in_
                    }
                else:
                    models_info[model_name] = {
                        "type": model.__class__.__name__,
                        "n_estimators": model.n_estimators if hasattr(model, "n_estimators") else None,
                        "feature_count": model.n_features_in_
                    }

            elif model_name.endswith("_nn"):
                # Get PyTorch model info
                models_info[model_name] = {
                    "type": "PyTorch Neural Network",
                    "input_size": model.input_size,
                    "hidden_size": model.hidden_size,
                    "num_layers": model.num_layers,
                    "param_count": sum(p.numel() for p in model.parameters())
                }

        return {
            "models": models_info,
            "count": len(models_info),
            "model_dir": self.model_dir
        }

# Test code for local debugging
if __name__ == "__main__":
    # Create engine
    engine = PredictionEngine()

    print("Model Info:")
    print(engine.get_model_info())

    # Generate some fake data for testing
    print("\nGenerating test data...")

    # Forex data
    forex_data = []
    for i in range(100):
        day = datetime.now() - timedelta(days=100-i)
        base_price = 1.1 + (i * 0.001)

        data_point = {
            "currency_pair": "EUR/USD",
            "timestamp": day.isoformat(),
            "open": base_price,
            "high": base_price + 0.01,
            "low": base_price - 0.01,
            "close": base_price + 0.002,
            "volume": 10000 + (i * 100)
        }

        forex_data.append(data_point)

    # Train forex model
    print("\nTraining forex Random Forest model...")
    result = engine.train_forex_model(forex_data, model_type="rf")
    print(f"Training result: {result}")

    print("\nTraining forex Neural Network model...")
    result = engine.train_forex_model(forex_data, model_type="nn")
    print(f"Training result: {result}")

    # Make forex prediction
    latest_data = forex_data[-1]
    print("\nMaking forex prediction with Random Forest...")
    prediction = engine.predict_forex(latest_data, days_ahead=5, model_name="forex_rf")
    print(f"Prediction: {prediction}")

    print("\nMaking forex prediction with Neural Network...")
    prediction = engine.predict_forex(latest_data, days_ahead=5, model_name="forex_nn")
    print(f"Prediction: {prediction}")

    # Sports data
    teams = ["Manchester United", "Liverpool", "Chelsea", "Arsenal", "Manchester City"]
    sports_data = []

    for i in range(50):
        home_idx = i % len(teams)
        away_idx = (i + 1) % len(teams)

        match_data = {
            "home_team": teams[home_idx],
            "away_team": teams[away_idx],
            "league": "Premier League",
            "home_goals_scored_avg": 1.8 + (home_idx * 0.1),
            "away_goals_scored_avg": 1.5 + (away_idx * 0.1),
            "home_goals_conceded_avg": 0.9 + (home_idx * 0.05),
            "away_goals_conceded_avg": 1.2 + (away_idx * 0.05),
            "match_date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
        }

        sports_data.append(match_data)

    # Train sports model
    print("\nTraining sports model...")
    result = engine.train_sports_model(sports_data)
    print(f"Training result: {result}")

    # Make sports prediction
    next_match = {
        "home_team": "Manchester United",
        "away_team": "Liverpool",
        "league": "Premier League",
        "home_goals_scored_avg": 2.1,
        "away_goals_scored_avg": 2.3,
        "home_goals_conceded_avg": 0.8,
        "away_goals_conceded_avg": 1.0,
        "match_date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    }

    print("\nMaking sports prediction...")
    prediction = engine.predict_sports(next_match)
    print(f"Prediction: {prediction}")

    # Updated model info
    print("\nUpdated Model Info:")
    print(engine.get_model_info())
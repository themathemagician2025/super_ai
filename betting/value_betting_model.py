# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Value Betting Model

This module implements a sports betting model that identifies value bets where
bookmaker odds underestimate the true probability of outcomes, creating positive
expected value (+EV) betting opportunities.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path
import json
import pickle
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ValueBettingModel:
    """
    A model that identifies value betting opportunities across various sports.
    Focuses on finding bets where the bookmaker's implied probability is lower
    than the model's predicted probability, resulting in positive EV.
    """

    def __init__(self, sport: str = None, market: str = None, model_path: Path = None):
        """
        Initialize the value betting model.

        Args:
            sport: Target sport (e.g., 'american_football', 'soccer')
            market: Betting market (e.g., 'moneyline', 'spread', 'over_under')
            model_path: Path to a pre-trained model
        """
        self.sport = sport
        self.market = market
        self.model = None
        self.feature_scaler = None
        self.feature_names = []
        self.performance_history = []
        self.min_ev_threshold = 0.05  # Minimum EV to consider a bet valuable
        self.kelly_fraction = 0.25    # Conservative Kelly criterion fraction

        # Directories
        self.base_dir = Path(__file__).parent.parent.parent
        self.models_dir = self.base_dir / 'models' / 'betting'
        self.data_dir = self.base_dir / 'data' / 'betting'

        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load model if provided
        if model_path:
            self.load_model(model_path)

    def train_model(self, data: pd.DataFrame, model_type: str = 'xgboost',
                   target_col: str = 'outcome', test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train a value betting model using historical data.

        Args:
            data: DataFrame with historical betting data
            model_type: Type of model to train ('logistic', 'xgboost', 'neural_network')
            target_col: Column name for the target variable
            test_size: Proportion of data to use for testing

        Returns:
            Dict with training results and metrics
        """
        logger.info(f"Training {model_type} model for {self.sport} {self.market} betting")

        # Prepare data
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        # Determine features (exclude non-feature columns)
        exclude_cols = [target_col, 'event_id', 'date', 'home_team', 'away_team',
                       'odds', 'bet_amount', 'profit', 'roi']
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        # Handle categorical features
        data_processed = pd.get_dummies(data[feature_cols], drop_first=True)
        self.feature_names = data_processed.columns.tolist()

        # Split data into features and target
        X = data_processed.values
        y = data[target_col].values

        # Scale features
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)

        # Split into training and testing sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42)

        # Train model based on specified type
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
            self.model.fit(X_train, y_train)

        elif model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                random_state=42
            )
            self.model.fit(X_train, y_train)

        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                class_weight='balanced'
            )
            self.model.fit(X_train, y_train)

        elif model_type == 'neural_network':
            # Define PyTorch model
            class BettingNN(nn.Module):
                def __init__(self, input_size):
                    super(BettingNN, self).__init__()
                    self.fc1 = nn.Linear(input_size, 64)
                    self.relu1 = nn.ReLU()
                    self.dropout1 = nn.Dropout(0.3)
                    self.fc2 = nn.Linear(64, 32)
                    self.relu2 = nn.ReLU()
                    self.dropout2 = nn.Dropout(0.2)
                    self.fc3 = nn.Linear(32, 1)
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x):
                    x = self.fc1(x)
                    x = self.relu1(x)
                    x = self.dropout1(x)
                    x = self.fc2(x)
                    x = self.relu2(x)
                    x = self.dropout2(x)
                    x = self.fc3(x)
                    x = self.sigmoid(x)
                    return x

            # Convert data to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))

            # Create model
            torch_model = BettingNN(X_train.shape[1])
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.001)

            # Train the model
            epochs = 100
            batch_size = 32
            for epoch in range(epochs):
                # Mini-batch training
                for i in range(0, len(X_train_tensor), batch_size):
                    batch_X = X_train_tensor[i:i+batch_size]
                    batch_y = y_train_tensor[i:i+batch_size]

                    # Forward pass
                    outputs = torch_model(batch_X)
                    loss = criterion(outputs, batch_y)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if (epoch+1) % 10 == 0:
                    logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

            self.model = torch_model

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Evaluate model performance
        metrics = self._evaluate_model(X_test, y_test, model_type)

        # Save the model
        model_filename = f"{self.sport}_{self.market}_{model_type}_model.pkl"
        self._save_model(model_filename)

        return {
            "model_type": model_type,
            "sport": self.sport,
            "market": self.market,
            "features": self.feature_names,
            "metrics": metrics
        }

    def _evaluate_model(self, X_test, y_test, model_type: str) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            X_test: Test features
            y_test: Test targets
            model_type: Type of model

        Returns:
            Dict with evaluation metrics
        """
        if model_type == 'neural_network':
            # PyTorch evaluation
            X_test_tensor = torch.FloatTensor(X_test)
            self.model.eval()
            with torch.no_grad():
                y_pred_proba = self.model(X_test_tensor).numpy().flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            # Scikit-learn model evaluation
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            y_pred = self.model.predict(X_test)

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

    def _save_model(self, filename: str) -> Path:
        """
        Save the trained model to disk.

        Args:
            filename: Name of the model file

        Returns:
            Path to the saved model
        """
        model_path = self.models_dir / filename

        if not self.model:
            raise ValueError("No model to save")

        # Create dict with model and metadata
        model_data = {
            "model": self.model,
            "feature_scaler": self.feature_scaler,
            "feature_names": self.feature_names,
            "sport": self.sport,
            "market": self.market,
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "version": "1.0",
                "description": f"Value betting model for {self.sport} {self.market}"
            }
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {model_path}")
        return model_path

    def load_model(self, model_path: Path) -> None:
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the model file
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_scaler = model_data["feature_scaler"]
        self.feature_names = model_data["feature_names"]
        self.sport = model_data["sport"]
        self.market = model_data["market"]

        logger.info(f"Loaded {self.sport} {self.market} model from {model_path}")

    def predict_probability(self, event_data: Dict[str, Any]) -> float:
        """
        Predict the probability of an outcome for a specific event.

        Args:
            event_data: Dictionary with event features

        Returns:
            Predicted probability
        """
        if not self.model:
            raise ValueError("Model not loaded")

        # Extract features
        features = self._extract_features(event_data)

        # Scale features
        if self.feature_scaler:
            features_scaled = self.feature_scaler.transform([features])
        else:
            features_scaled = np.array([features])

        # Make prediction
        if isinstance(self.model, nn.Module):
            # PyTorch model
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled)
                probability = self.model(features_tensor).item()
        else:
            # Scikit-learn model
            probability = self.model.predict_proba(features_scaled)[0, 1]

        return probability

    def _extract_features(self, event_data: Dict[str, Any]) -> List[float]:
        """
        Extract features from event data in the correct order.

        Args:
            event_data: Dictionary with event features

        Returns:
            List of feature values
        """
        # Initialize feature vector with zeros
        features = [0.0] * len(self.feature_names)

        # Fill in values for features that exist in event_data
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in event_data:
                features[i] = float(event_data[feature_name])
            else:
                # Handle one-hot encoded categorical features
                for category_name in event_data:
                    if category_name + "_" + str(event_data[category_name]) == feature_name:
                        features[i] = 1.0

        return features

    def calculate_implied_probability(self, odds: float) -> float:
        """
        Calculate implied probability from decimal odds.

        Args:
            odds: Decimal odds (e.g., 2.0 means 2-to-1 payout)

        Returns:
            Implied probability
        """
        return 1.0 / odds if odds > 0 else 0.0

    def calculate_expected_value(self, model_probability: float, odds: float) -> float:
        """
        Calculate expected value (EV) of a bet.

        Args:
            model_probability: Predicted probability of the outcome
            odds: Decimal odds offered by the bookmaker

        Returns:
            Expected value
        """
        # EV = (probability * potential profit) - (1 - probability)
        # Where potential profit = odds - 1
        return (model_probability * (odds - 1)) - (1 - model_probability)

    def calculate_kelly_criterion(self, model_probability: float, odds: float) -> float:
        """
        Calculate Kelly Criterion optimal bet size as a fraction of bankroll.

        Args:
            model_probability: Predicted probability of the outcome
            odds: Decimal odds offered by the bookmaker

        Returns:
            Kelly stake as fraction of bankroll
        """
        # Kelly formula: f* = (p*(b+1) - 1)/b where:
        # f* is the fraction of bankroll to bet
        # p is the probability of winning
        # b is the odds minus 1 (the profit on a successful bet)
        b = odds - 1
        if model_probability * (b + 1) <= 1:  # Negative expectation
            return 0.0

        kelly = (model_probability * (b + 1) - 1) / b

        # Conservative Kelly (fractional Kelly)
        return kelly * self.kelly_fraction

    def analyze_bet(self, event_data: Dict[str, Any], market_odds: float) -> Dict[str, Any]:
        """
        Analyze a potential bet for value.

        Args:
            event_data: Dictionary with event features
            market_odds: Decimal odds offered by the bookmaker

        Returns:
            Dictionary with bet analysis
        """
        # Predict outcome probability
        model_probability = self.predict_probability(event_data)

        # Calculate implied probability
        implied_probability = self.calculate_implied_probability(market_odds)

        # Calculate edge
        edge = model_probability - implied_probability

        # Calculate expected value
        ev = self.calculate_expected_value(model_probability, market_odds)

        # Calculate Kelly stake
        kelly = self.calculate_kelly_criterion(model_probability, market_odds)

        # Determine if this is a value bet
        is_value_bet = ev > self.min_ev_threshold

        # Create result dictionary
        result = {
            "event": self._format_event_name(event_data),
            "sport": self.sport,
            "market": self.market,
            "model_probability": model_probability,
            "implied_probability": implied_probability,
            "market_odds": market_odds,
            "edge": edge,
            "expected_value": ev,
            "kelly_stake": kelly,
            "is_value_bet": is_value_bet,
            "recommendation": "BET" if is_value_bet else "PASS",
            "confidence": min(abs(edge) * 10, 1.0),  # Scale confidence by edge
            "bet_rating": self._calculate_bet_rating(ev, edge, kelly)
        }

        return result

    def _format_event_name(self, event_data: Dict[str, Any]) -> str:
        """Format an event name from event data"""
        if 'home_team' in event_data and 'away_team' in event_data:
            return f"{event_data['home_team']} vs {event_data['away_team']}"
        elif 'event_name' in event_data:
            return event_data['event_name']
        else:
            return "Unknown Event"

    def _calculate_bet_rating(self, ev: float, edge: float, kelly: float) -> int:
        """
        Calculate a 1-5 star rating for a bet based on key metrics.

        Args:
            ev: Expected value of the bet
            edge: Edge (difference between model and implied probabilities)
            kelly: Kelly criterion stake

        Returns:
            Integer rating from 1-5
        """
        if ev <= 0 or edge <= 0:
            return 0

        # Base rating on combination of EV, edge and Kelly
        ev_score = min(ev * 10, 2.5)           # 0-2.5 points based on EV
        edge_score = min(edge * 10, 1.5)       # 0-1.5 points based on edge
        kelly_score = min(kelly * 10, 1.0)     # 0-1.0 points based on Kelly %

        total_score = ev_score + edge_score + kelly_score

        # Convert to 1-5 rating
        if total_score >= 4.0:
            return 5
        elif total_score >= 3.0:
            return 4
        elif total_score >= 2.0:
            return 3
        elif total_score >= 1.0:
            return 2
        else:
            return 1

    def backtest(self, historical_data: pd.DataFrame, bankroll: float = 1000.0,
                stake_strategy: str = 'kelly', ev_threshold: float = 0.05) -> Dict[str, Any]:
        """
        Backtest the model on historical data.

        Args:
            historical_data: DataFrame with historical betting data
            bankroll: Starting bankroll for simulation
            stake_strategy: Strategy for stake sizing ('fixed', 'kelly', or 'fractional_kelly')
            ev_threshold: Minimum EV to place a bet

        Returns:
            Dictionary with backtest results
        """
        if historical_data.empty:
            raise ValueError("No historical data provided for backtesting")

        required_cols = ['odds', 'outcome']
        if not all(col in historical_data.columns for col in required_cols):
            raise ValueError(f"Historical data missing required columns: {required_cols}")

        # Set threshold
        self.min_ev_threshold = ev_threshold

        # Initialize results
        results = []
        current_bankroll = bankroll
        bet_count = 0
        win_count = 0
        loss_count = 0

        # Loop through historical data
        for _, row in historical_data.iterrows():
            # Convert row to dictionary
            event_data = row.to_dict()

            # Calculate probability and EV
            model_probability = self.predict_probability(event_data)
            market_odds = event_data['odds']
            ev = self.calculate_expected_value(model_probability, market_odds)

            # Skip if not a value bet
            if ev <= ev_threshold:
                continue

            # Determine stake
            if stake_strategy == 'fixed':
                stake = 0.02 * bankroll  # 2% of initial bankroll
            elif stake_strategy == 'kelly':
                kelly = self.calculate_kelly_criterion(model_probability, market_odds)
                stake = kelly * current_bankroll
            elif stake_strategy == 'fractional_kelly':
                kelly = self.calculate_kelly_criterion(model_probability, market_odds)
                stake = kelly * current_bankroll * 0.5  # Half Kelly
            else:
                raise ValueError(f"Unknown stake strategy: {stake_strategy}")

            # Ensure stake is valid
            stake = min(stake, current_bankroll)

            # Determine outcome
            actual_outcome = event_data['outcome']

            # Calculate profit/loss
            if actual_outcome == 1:  # Win
                profit = stake * (market_odds - 1)
                win_count += 1
            else:  # Loss
                profit = -stake
                loss_count += 1

            # Update bankroll
            current_bankroll += profit
            bet_count += 1

            # Record result
            results.append({
                "event": self._format_event_name(event_data),
                "predicted_probability": model_probability,
                "odds": market_odds,
                "expected_value": ev,
                "stake": stake,
                "outcome": actual_outcome,
                "profit": profit,
                "bankroll": current_bankroll
            })

        # Calculate overall metrics
        if bet_count > 0:
            win_rate = win_count / bet_count
            total_profit = current_bankroll - bankroll
            roi = total_profit / bankroll

            # Calculate per-bet metrics
            total_staked = sum(r['stake'] for r in results)
            if total_staked > 0:
                yield_pct = total_profit / total_staked * 100
            else:
                yield_pct = 0

        else:
            win_rate = 0
            total_profit = 0
            roi = 0
            yield_pct = 0

        # Create backtest result
        backtest_result = {
            "bets_placed": bet_count,
            "wins": win_count,
            "losses": loss_count,
            "win_rate": win_rate,
            "starting_bankroll": bankroll,
            "final_bankroll": current_bankroll,
            "total_profit": total_profit,
            "roi": roi,
            "yield_percent": yield_pct,
            "bet_details": results
        }

        return backtest_result

    def plot_backtest_results(self, backtest_results: Dict[str, Any],
                             save_path: Optional[Path] = None) -> None:
        """
        Plot the results of a backtest.

        Args:
            backtest_results: Dictionary with backtest results
            save_path: Path to save the plot image
        """
        bet_details = backtest_results.get('bet_details', [])
        if not bet_details:
            logger.warning("No bet details to plot")
            return

        # Extract data
        bankroll_history = [detail['bankroll'] for detail in bet_details]
        events = [detail['event'] for detail in bet_details]

        # Create figure with multiple subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Plot bankroll over time
        ax1.plot(range(len(bankroll_history)), bankroll_history, marker='o', linestyle='-')
        ax1.set_title(f'Backtest Results: {self.sport} {self.market} Betting')
        ax1.set_xlabel('Bet Number')
        ax1.set_ylabel('Bankroll')
        ax1.grid(True)

        # Highlight profitable and unprofitable bets
        profits = [detail['profit'] for detail in bet_details]
        win_indices = [i for i, profit in enumerate(profits) if profit > 0]
        loss_indices = [i for i, profit in enumerate(profits) if profit <= 0]

        ax1.scatter(win_indices, [bankroll_history[i] for i in win_indices],
                   color='green', s=50, label='Win')
        ax1.scatter(loss_indices, [bankroll_history[i] for i in loss_indices],
                   color='red', s=50, label='Loss')

        # Add win rate and ROI in the legend
        win_rate = backtest_results.get('win_rate', 0)
        roi = backtest_results.get('roi', 0)
        legend_text = f'Win Rate: {win_rate:.2%}, ROI: {roi:.2%}'
        ax1.legend(title=legend_text)

        # Plot individual bet profits
        ax2.bar(range(len(profits)), profits, color=['green' if p > 0 else 'red' for p in profits])
        ax2.set_title('Individual Bet Profits/Losses')
        ax2.set_xlabel('Bet Number')
        ax2.set_ylabel('Profit/Loss')
        ax2.grid(True)

        # Adjust layout
        plt.tight_layout()

        # Save or display
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved backtest plot to {save_path}")
        else:
            plt.show()

    def find_value_bets(self, events_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple events to find value betting opportunities.

        Args:
            events_data: List of dictionaries with event data and odds

        Returns:
            List of value bets sorted by expected value
        """
        value_bets = []

        for event_data in events_data:
            # Extract market odds
            if 'market_odds' not in event_data:
                logger.warning(f"No market odds found for event: {self._format_event_name(event_data)}")
                continue

            market_odds = event_data['market_odds']

            # Analyze the bet
            bet_analysis = self.analyze_bet(event_data, market_odds)

            # Add to value bets if it has positive expected value
            if bet_analysis['is_value_bet']:
                value_bets.append(bet_analysis)

        # Sort by expected value (highest first)
        return sorted(value_bets, key=lambda x: x['expected_value'], reverse=True)

    def generate_poisson_predictions(self, home_expected_goals: float,
                                    away_expected_goals: float,
                                    max_goals: int = 10) -> Dict[str, Any]:
        """
        Generate Poisson distribution-based predictions for soccer matches.

        Args:
            home_expected_goals: Expected goals for home team
            away_expected_goals: Expected goals for away team
            max_goals: Maximum number of goals to consider

        Returns:
            Dictionary with various market predictions
        """
        import scipy.stats as stats

        # Calculate probability for each exact score
        score_probs = {}
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                home_prob = stats.poisson.pmf(home_goals, home_expected_goals)
                away_prob = stats.poisson.pmf(away_goals, away_expected_goals)
                score_probs[f"{home_goals}-{away_goals}"] = home_prob * away_prob

        # Calculate 1X2 market probabilities
        home_win_prob = sum(prob for score, prob in score_probs.items()
                           if int(score.split('-')[0]) > int(score.split('-')[1]))
        draw_prob = sum(prob for score, prob in score_probs.items()
                       if int(score.split('-')[0]) == int(score.split('-')[1]))
        away_win_prob = sum(prob for score, prob in score_probs.items()
                           if int(score.split('-')[0]) < int(score.split('-')[1]))

        # Calculate over/under probabilities
        ou_probs = {}
        for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
            over_prob = sum(prob for score, prob in score_probs.items()
                          if (int(score.split('-')[0]) + int(score.split('-')[1])) > threshold)
            ou_probs[f"over_{threshold}"] = over_prob
            ou_probs[f"under_{threshold}"] = 1 - over_prob

        # Calculate both teams to score probability
        btts_yes_prob = sum(prob for score, prob in score_probs.items()
                           if int(score.split('-')[0]) > 0 and int(score.split('-')[1]) > 0)
        btts_no_prob = 1 - btts_yes_prob

        # Return all predictions
        return {
            "expected_goals": {
                "home": home_expected_goals,
                "away": away_expected_goals,
                "total": home_expected_goals + away_expected_goals
            },
            "match_odds": {
                "home_win": home_win_prob,
                "draw": draw_prob,
                "away_win": away_win_prob
            },
            "correct_score": {k: v for k, v in sorted(score_probs.items(), key=lambda x: x[1], reverse=True)},
            "over_under": ou_probs,
            "btts": {
                "yes": btts_yes_prob,
                "no": btts_no_prob
            }
        }

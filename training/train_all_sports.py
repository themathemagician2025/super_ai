# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Train All Sports Models Script

This script automates the training of prediction models for all supported sports.
It generates sample data for each sport and trains a model using sports_predictor.py.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random
import torch

# Add parent directory to path to import from sibling directories
sys.path.append(str(Path(__file__).parent.parent))

from prediction.sports_predictor import SportsPredictor
from data_pipeline.data_preprocessor import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models' / 'sports'
os.makedirs(MODELS_DIR, exist_ok=True)

# List of all supported sports to train
SUPPORTED_SPORTS = [
    'american_football', 'basketball', 'baseball', 'soccer', 'tennis',
    'golf', 'boxing', 'mma', 'cricket', 'rugby_union', 'rugby_league',
    'ice_hockey', 'australian_rules', 'lacrosse', 'volleyball', 'handball',
    'snooker', 'darts', 'table_tennis', 'badminton'
]

class SportDataGenerator:
    """Generate synthetic training data for each sport"""

    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        self.preprocessor = DataPreprocessor()

    def generate_team_names(self, sport, n_teams=20):
        """Generate plausible team names for a sport"""
        prefixes = ['New York', 'Los Angeles', 'Chicago', 'Boston', 'Dallas', 'Miami', 'Seattle',
                   'London', 'Paris', 'Berlin', 'Tokyo', 'Moscow', 'Sydney', 'Madrid', 'Rome']

        suffixes = {
            'american_football': ['Eagles', 'Giants', 'Cowboys', 'Patriots', 'Rams', 'Chargers', 'Bears'],
            'basketball': ['Lakers', 'Celtics', 'Bulls', 'Heat', 'Rockets', 'Warriors', 'Clippers'],
            'baseball': ['Yankees', 'Red Sox', 'Cubs', 'Dodgers', 'Giants', 'Cardinals', 'Astros'],
            'soccer': ['United', 'FC', 'City', 'Wanderers', 'Athletic', 'Rovers', 'Dynamo'],
            'ice_hockey': ['Penguins', 'Bruins', 'Blackhawks', 'Rangers', 'Flames', 'Jets', 'Flyers'],
            'cricket': ['Challengers', 'Kings', 'Indians', 'Warriors', 'Royals', 'Strikers'],
            'rugby_union': ['Warriors', 'Blues', 'Chiefs', 'Tigers', 'Sharks', 'Lions'],
            'rugby_league': ['Broncos', 'Raiders', 'Sharks', 'Warriors', 'Tigers', 'Dragons'],
            'australian_rules': ['Bombers', 'Eagles', 'Hawks', 'Tigers', 'Magpies', 'Swans'],
            'volleyball': ['Sharks', 'Dragons', 'Thunder', 'Lightning', 'Aces', 'Wolves'],
            'handball': ['Lions', 'Tigers', 'Bears', 'Wolves', 'Eagles', 'Hawks']
        }

        # For individual sports, use player names instead
        if sport in ['tennis', 'golf', 'boxing', 'mma', 'snooker', 'darts', 'table_tennis', 'badminton']:
            first_names = ['John', 'David', 'Michael', 'James', 'Robert', 'Mary', 'Patricia', 'Jennifer',
                          'Linda', 'Elizabeth', 'Rafael', 'Serena', 'Roger', 'Novak', 'Maria', 'Tiger']
            last_names = ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson',
                         'Moore', 'Taylor', 'Nadal', 'Williams', 'Federer', 'Djokovic', 'Sharapova', 'Woods']
            return [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n_teams)]

        # Use team names for team sports
        sport_suffixes = suffixes.get(sport, ['United', 'City', 'FC'])
        teams = []
        for _ in range(n_teams):
            prefix = random.choice(prefixes)
            suffix = random.choice(sport_suffixes)
            teams.append(f"{prefix} {suffix}")
        return teams

    def generate_sport_dataset(self, sport):
        """Generate synthetic data for a specific sport"""
        logger.info(f"Generating dataset for {sport}")

        # Generate team/player names
        teams = self.generate_team_names(sport)
        n_teams = len(teams)

        # Create base DataFrame
        data = []

        # Generate matches or events
        start_date = datetime.now() - timedelta(days=365*2)  # 2 years of data
        end_date = datetime.now()

        for _ in range(self.n_samples):
            match_date = start_date + (end_date - start_date) * random.random()

            # For team sports
            if sport not in ['tennis', 'golf', 'boxing', 'mma', 'snooker', 'darts', 'table_tennis', 'badminton']:
                home_team = teams[random.randint(0, n_teams-1)]
                # Ensure away team is different from home team
                away_team = teams[random.randint(0, n_teams-1)]
                while away_team == home_team:
                    away_team = teams[random.randint(0, n_teams-1)]

                # Generate base event data
                event = {
                    'sport': sport,
                    'date': match_date,
                    'home_team': home_team,
                    'away_team': away_team
                }

                # Add sport-specific metrics
                if sport == 'american_football':
                    home_score = random.randint(0, 45)
                    away_score = random.randint(0, 45)
                    event.update({
                        'home_score': home_score,
                        'away_score': away_score,
                        'total_yards': random.randint(200, 600),
                        'turnovers': random.randint(0, 6),
                        'pass_attempts': random.randint(20, 50),
                        'pass_completions': random.randint(10, 40)
                    })

                elif sport == 'basketball':
                    home_score = random.randint(70, 130)
                    away_score = random.randint(70, 130)
                    event.update({
                        'home_score': home_score,
                        'away_score': away_score,
                        'field_goals_attempted': random.randint(60, 100),
                        'field_goals_made': random.randint(30, 60),
                        'three_pointers_attempted': random.randint(20, 40),
                        'three_pointers_made': random.randint(5, 20)
                    })

                elif sport == 'baseball':
                    home_score = random.randint(0, 15)
                    away_score = random.randint(0, 15)
                    event.update({
                        'home_score': home_score,
                        'away_score': away_score,
                        'hits': random.randint(0, 15),
                        'at_bats': random.randint(30, 40),
                        'earned_runs': random.randint(0, 10),
                        'innings_pitched': random.uniform(3, 9)
                    })

                elif sport == 'soccer':
                    home_score = random.randint(0, 5)
                    away_score = random.randint(0, 5)
                    event.update({
                        'home_score': home_score,
                        'away_score': away_score,
                        'shots': random.randint(5, 25),
                        'goals': home_score + away_score,
                        'possession': random.randint(30, 70)
                    })

                elif sport == 'ice_hockey':
                    home_score = random.randint(0, 7)
                    away_score = random.randint(0, 7)
                    event.update({
                        'home_score': home_score,
                        'away_score': away_score,
                        'shots': random.randint(20, 40),
                        'goals': home_score + away_score,
                        'power_play_opportunities': random.randint(1, 8),
                        'power_play_goals': random.randint(0, 3)
                    })

                else:
                    # Generic team sport
                    home_score = random.randint(0, 100)
                    away_score = random.randint(0, 100)
                    event.update({
                        'home_score': home_score,
                        'away_score': away_score
                    })

            else:  # Individual sports
                player1 = teams[random.randint(0, n_teams-1)]
                player2 = teams[random.randint(0, n_teams-1)]
                while player2 == player1:
                    player2 = teams[random.randint(0, n_teams-1)]

                # Generate base event data
                event = {
                    'sport': sport,
                    'date': match_date,
                    'home_team': player1,  # Using player names
                    'away_team': player2,
                    'home_score': random.randint(0, 30),
                    'away_score': random.randint(0, 30)
                }

                # Add sport-specific metrics
                if sport == 'tennis':
                    event.update({
                        'first_serve_attempts': random.randint(50, 100),
                        'first_serve_in': random.randint(30, 70),
                        'break_points_faced': random.randint(0, 10),
                        'break_points_saved': random.randint(0, 10)
                    })

            # Add common environmental factors
            event.update({
                'temperature': random.uniform(0, 35),
                'is_rainy': random.random() > 0.8,
                'wind_speed': random.uniform(0, 30),
                'humidity': random.uniform(30, 90)
            })

            data.append(event)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Apply preprocessing to add additional features
        processed_df = self.preprocessor.add_sports_features(df)

        return processed_df

    def save_dataset(self, df, sport):
        """Save the generated dataset to disk"""
        sport_dir = DATA_DIR / 'sports'
        os.makedirs(sport_dir, exist_ok=True)

        file_path = sport_dir / f"{sport}_data.csv"
        df.to_csv(file_path, index=False)
        logger.info(f"Saved dataset to {file_path}")

        return file_path

def train_sports_model(sport, data_path=None):
    """Train a sports prediction model for a specific sport"""
    logger.info(f"Training model for {sport}")

    try:
        # Generate data if not provided
        if data_path is None:
            generator = SportDataGenerator(n_samples=1000)
            df = generator.generate_sport_dataset(sport)
            data_path = generator.save_dataset(df, sport)

        # Load data
        df = pd.read_csv(data_path)

        # Prepare features and targets
        X = df.drop(['home_score', 'away_score', 'result'], axis=1, errors='ignore')

        # If result exists, use it as target, otherwise create from scores
        if 'result' in df.columns:
            y = df['result'].map({'home_win': 1, 'draw': 0.5, 'away_win': 0})
        elif all(col in df.columns for col in ['home_score', 'away_score']):
            y = (df['home_score'] > df['away_score']).astype(float)
            # Add draws as 0.5
            y[df['home_score'] == df['away_score']] = 0.5
        else:
            raise ValueError(f"Dataset for {sport} missing required target columns")

        # Convert categorical features to numeric
        X = pd.get_dummies(X)

        # Initialize and train the model
        predictor = SportsPredictor()

        # Train the model (simplified)
        model_path = MODELS_DIR / f"{sport}_model.pth"

        # Create features tensor
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)

        # Training would be handled by sports_predictor's train method
        # In a real implementation, this would be a proper training process
        # For now, just save a basic model structure
        model = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

        # Simple training
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        logger.info(f"Training {sport} model with {len(X)} samples")

        # Train for a few epochs (simplified)
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 2 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        # Save model with metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': {
                'sport': sport,
                'input_size': X.shape[1],
                'feature_names': X.columns.tolist(),
                'training_samples': len(X),
                'training_date': datetime.now().isoformat()
            }
        }, model_path)

        logger.info(f"Saved {sport} model to {model_path}")
        return model_path

    except Exception as e:
        logger.error(f"Error training model for {sport}: {e}")
        return None

def main():
    """Train models for all supported sports"""
    logger.info(f"Starting training for {len(SUPPORTED_SPORTS)} sports")

    results = {}
    for sport in SUPPORTED_SPORTS:
        model_path = train_sports_model(sport)
        results[sport] = "Success" if model_path else "Failed"

    # Print summary
    logger.info("Training completed. Summary:")
    for sport, status in results.items():
        logger.info(f"{sport}: {status}")

if __name__ == "__main__":
    main()

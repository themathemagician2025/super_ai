# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Generate Betting Data

This script generates synthetic betting data for testing and development.
"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BettingDataGenerator:
    """Generate synthetic betting data for testing models."""

    def __init__(self):
        """Initialize the betting data generator."""
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / 'data' / 'betting'
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def generate_data(self, sport: str, market: str, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic betting data for a specific sport and market.

        Args:
            sport: Target sport (e.g., 'american_football', 'soccer')
            market: Betting market (e.g., 'moneyline', 'spread', 'over_under')
            n_samples: Number of samples to generate

        Returns:
            DataFrame with synthetic betting data
        """
        logger.info(f"Generating {n_samples} samples for {sport} {market} betting")

        # Generate team/player names
        teams = self._generate_team_names(sport, 20)

        # Initialize data list
        data = []

        # Generate over specified time period
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        days_range = (end_date - start_date).days

        # Set various parameters based on sport and market
        features = self._get_features_for_sport(sport)

        # Generate samples
        for i in range(n_samples):
            # Pick random date
            sample_date = start_date + timedelta(days=random.randint(0, days_range))

            # Pick random teams/players
            home_idx = random.randint(0, len(teams)-1)
            away_idx = random.randint(0, len(teams)-1)
            while away_idx == home_idx:  # Ensure different teams
                away_idx = random.randint(0, len(teams)-1)

            home_team = teams[home_idx]
            away_team = teams[away_idx]

            # Base event dict
            event = {
                'event_id': i+1,
                'date': sample_date.strftime('%Y-%m-%d'),
                'home_team': home_team,
                'away_team': away_team,
                'sport': sport,
                'market': market
            }

            # Add odds and outcome based on market type
            if market == 'moneyline':
                # Moneyline odds (decimal)
                home_implied_prob = random.uniform(0.3, 0.7)  # Random implied probability
                home_odds = round(1 / home_implied_prob, 2)

                # Actual outcome (based on a model with some randomness)
                true_prob = home_implied_prob + random.uniform(-0.05, 0.15)  # Add some edge or vig
                true_prob = min(max(true_prob, 0.1), 0.9)  # Keep in reasonable range
                outcome = 1 if random.random() < true_prob else 0

                # Record for home win
                event_home = event.copy()
                event_home.update({
                    'selection': home_team,
                    'odds': home_odds,
                    'implied_probability': home_implied_prob,
                    'true_probability': true_prob,
                    'outcome': outcome,
                })

                # Record for away win (opposite outcome, different odds)
                away_implied_prob = 1 - home_implied_prob  # Simplified, ignoring vig
                away_odds = round(1 / away_implied_prob, 2)
                event_away = event.copy()
                event_away.update({
                    'selection': away_team,
                    'odds': away_odds,
                    'implied_probability': away_implied_prob,
                    'true_probability': 1 - true_prob,
                    'outcome': 1 - outcome,
                })

                # Add to data list
                data.append(event_home)
                data.append(event_away)

            elif market == 'spread':
                # Point spread
                if sport == 'american_football':
                    spread = random.choice([-13.5, -10.5, -7.5, -6.5, -3.5, -2.5, -1.5, 1.5, 2.5, 3.5, 6.5, 7.5, 10.5, 13.5])
                else:
                    spread = random.choice([-9.5, -7.5, -6.5, -4.5, -3.5, -2.5, -1.5, 1.5, 2.5, 3.5, 4.5, 6.5, 7.5, 9.5])

                # Standard spread odds (around -110 in American odds, which is about 1.91 in decimal)
                odds = round(random.uniform(1.85, 1.95), 2)

                # Actual margin (based on spread with some randomness)
                true_margin = spread + random.normalvariate(0, 7)  # Random margin centered around the spread

                # Outcome (1 if spread is covered, 0 if not)
                outcome = 1 if true_margin > 0 else 0

                event.update({
                    'selection': f"{home_team} {spread:+g}",
                    'spread': spread,
                    'odds': odds,
                    'implied_probability': 1/odds,
                    'true_margin': true_margin,
                    'outcome': outcome,
                })

                data.append(event)

            elif market == 'over_under':
                # Over/Under line
                if sport == 'american_football':
                    ou_line = round(random.uniform(35, 55) * 2) / 2  # Typical NFL totals, rounded to nearest half-point
                elif sport == 'basketball':
                    ou_line = round(random.uniform(200, 240) * 2) / 2  # Typical NBA totals
                elif sport == 'soccer':
                    ou_line = round(random.uniform(1.5, 3.5) * 2) / 2  # Typical soccer totals
                else:
                    ou_line = round(random.uniform(3, 7) * 2) / 2  # Generic total

                # Standard over/under odds
                odds = round(random.uniform(1.85, 1.95), 2)

                # Actual total (based on line with some randomness)
                true_total = ou_line + random.normalvariate(0, ou_line * 0.15)  # Random total centered around the line

                # Two records: one for over, one for under
                event_over = event.copy()
                event_over.update({
                    'selection': f"Over {ou_line}",
                    'total_line': ou_line,
                    'odds': odds,
                    'implied_probability': 1/odds,
                    'true_total': true_total,
                    'outcome': 1 if true_total > ou_line else 0,
                })

                event_under = event.copy()
                event_under.update({
                    'selection': f"Under {ou_line}",
                    'total_line': ou_line,
                    'odds': odds,
                    'implied_probability': 1/odds,
                    'true_total': true_total,
                    'outcome': 1 if true_total < ou_line else 0,
                })

                data.append(event_over)
                data.append(event_under)

            # Add sport-specific features
            sport_features = self._generate_random_features(features)
            for event_dict in [d for d in data if d.get('event_id') == i+1]:
                event_dict.update(sport_features)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Add bet amount and profit/ROI calculations for records
        bet_amount = 100  # Standard bet amount
        df['bet_amount'] = bet_amount
        df['profit'] = df.apply(lambda x: bet_amount * (x['odds'] - 1) if x['outcome'] == 1 else -bet_amount, axis=1)
        df['roi'] = df['profit'] / bet_amount

        return df

    def save_data(self, df: pd.DataFrame, sport: str, market: str) -> Path:
        """
        Save the generated data to a CSV file.

        Args:
            df: DataFrame with betting data
            sport: Sport name
            market: Market type

        Returns:
            Path to the saved file
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"{sport}_{market}_betting_data_{timestamp}.csv"
        file_path = self.data_dir / filename

        # Save to CSV
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(df)} betting records to {file_path}")

        return file_path

    def _generate_team_names(self, sport: str, n_teams: int = 20) -> list:
        """Generate realistic team names for the sport."""
        prefixes = ['New York', 'Los Angeles', 'Chicago', 'Boston', 'Dallas', 'Miami', 'Seattle',
                   'London', 'Paris', 'Berlin', 'Tokyo', 'Sydney', 'Madrid', 'Rome', 'Atlanta']

        suffixes = {
            'american_football': ['Eagles', 'Giants', 'Cowboys', 'Patriots', 'Rams', 'Chargers', 'Bears', 'Vikings'],
            'basketball': ['Lakers', 'Celtics', 'Bulls', 'Heat', 'Rockets', 'Warriors', 'Clippers', 'Knicks'],
            'baseball': ['Yankees', 'Red Sox', 'Cubs', 'Dodgers', 'Giants', 'Cardinals', 'Astros', 'Braves'],
            'soccer': ['United', 'City', 'FC', 'Athletic', 'Rovers', 'Wanderers', 'Rangers', 'Dynamo'],
        }

        # Individual sports names
        if sport in ['tennis', 'golf', 'boxing', 'mma']:
            first_names = ['John', 'David', 'Michael', 'James', 'Robert', 'Emma', 'Sarah', 'Maria', 'Roger', 'Rafael']
            last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Davis', 'Wilson', 'Federer', 'Nadal']
            return [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n_teams)]

        # Team sports
        sport_suffixes = suffixes.get(sport, ['United', 'City', 'FC'])
        teams = []

        for i in range(n_teams):
            if i < len(prefixes):
                prefix = prefixes[i]
            else:
                prefix = random.choice(prefixes)

            suffix = random.choice(sport_suffixes)
            teams.append(f"{prefix} {suffix}")

        return teams

    def _get_features_for_sport(self, sport: str) -> dict:
        """Define features relevant to each sport."""
        features = {
            'american_football': {
                'qb_rating': (60, 120),  # Quarterback rating
                'yards_per_play': (3.5, 6.5),  # Yards per play
                'turnover_margin': (-2, 2),  # Turnover margin last 3 games
                'third_down_pct': (0.3, 0.5),  # Third down conversion percentage
                'pass_yards_per_game': (180, 320),  # Passing yards per game
                'rush_yards_per_game': (90, 180),  # Rushing yards per game
                'points_per_game': (16, 30),  # Points per game
                'points_allowed_per_game': (16, 30),  # Points allowed per game
                'home_field_advantage': (1, 3),  # Home field advantage in points
                'injuries_impact': (0, 1),  # Impact of injuries on team (0-1 scale)
            },
            'basketball': {
                'off_rating': (105, 120),  # Offensive rating
                'def_rating': (105, 120),  # Defensive rating
                'pace': (95, 105),  # Pace factor
                'efg_pct': (0.48, 0.56),  # Effective field goal percentage
                'ts_pct': (0.54, 0.62),  # True shooting percentage
                'orb_pct': (20, 30),  # Offensive rebound percentage
                'ft_rate': (0.2, 0.35),  # Free throw rate
                'tov_pct': (12, 16),  # Turnover percentage
                'rest_days': (1, 4),  # Days of rest
                'injuries_impact': (0, 1),  # Impact of injuries on team (0-1 scale)
            },
            'soccer': {
                'goals_per_game': (0.8, 2.2),  # Goals per game
                'xg_per_game': (0.8, 2.2),  # Expected goals per game
                'shots_per_game': (8, 18),  # Shots per game
                'possession_pct': (40, 60),  # Possession percentage
                'pass_accuracy': (70, 90),  # Pass accuracy percentage
                'save_pct': (60, 80),  # Save percentage
                'clean_sheet_pct': (0.2, 0.4),  # Clean sheet percentage
                'form_rating': (0, 10),  # Current form rating (0-10)
                'home_advantage': (0.1, 0.3),  # Home advantage factor
                'injuries_impact': (0, 1),  # Impact of injuries on team (0-1 scale)
            }
        }

        # Default features for any sport not specifically defined
        default_features = {
            'form_rating': (0, 10),  # Current form (0-10)
            'offense_rating': (60, 90),  # Offense strength (0-100)
            'defense_rating': (60, 90),  # Defense strength (0-100)
            'home_advantage': (0.1, 0.3),  # Home advantage factor
            'injuries_impact': (0, 1),  # Impact of injuries (0-1)
        }

        return features.get(sport, default_features)

    def _generate_random_features(self, features_dict: dict) -> dict:
        """Generate random values for each feature within specified ranges."""
        result = {}

        for feature, (min_val, max_val) in features_dict.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                result[feature] = random.randint(min_val, max_val)
            else:
                result[feature] = round(random.uniform(min_val, max_val), 3)

        return result


def main():
    """Generate synthetic betting data for testing."""
    generator = BettingDataGenerator()

    # Define sports and markets to generate data for
    sports_markets = [
        ('american_football', 'moneyline'),
        ('american_football', 'spread'),
        ('american_football', 'over_under'),
        ('basketball', 'moneyline'),
        ('basketball', 'spread'),
        ('basketball', 'over_under'),
        ('soccer', 'moneyline'),
        ('soccer', 'over_under')
    ]

    for sport, market in sports_markets:
        # Generate data
        df = generator.generate_data(sport=sport, market=market, n_samples=200)

        # Save to CSV
        generator.save_data(df, sport, market)


if __name__ == "__main__":
    main()

# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Data Processor Module

This module processes scraped and fetched data for use in the Super AI prediction system.
It converts raw data into formatted inputs for model training and prediction.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logger
logger = logging.getLogger(__name__)

class DataProcessor:
    """Class for processing scraped data for use in prediction models."""

    def __init__(self, output_dir: str = None):
        """
        Initialize the data processor.

        Args:
            output_dir: Directory to save processed data (defaults to data/processed_data)
        """
        self.output_dir = output_dir or os.path.join('data', 'processed_data')
        os.makedirs(self.output_dir, exist_ok=True)

    def process_sports_data(self, data_path: str) -> pd.DataFrame:
        """
        Process sports fixture data for the sports prediction model.

        Args:
            data_path: Path to raw sports data CSV file

        Returns:
            DataFrame with processed data
        """
        logger.info(f"Processing sports data from {data_path}")

        try:
            # Load the raw data
            df = pd.read_csv(data_path)

            # Check if necessary columns exist
            required_cols = ['home_team', 'away_team', 'league']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.error(f"Missing required columns in sports data: {missing_cols}")
                return pd.DataFrame()

            # Process date column if it exists
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'])
                    df['day_of_week'] = df['date'].dt.dayofweek
                    df['month'] = df['date'].dt.month
                except Exception as e:
                    logger.warning(f"Error processing date column: {str(e)}")
                    # Add default values
                    df['day_of_week'] = 0
                    df['month'] = datetime.datetime.now().month
            else:
                # Add default values
                df['day_of_week'] = 0
                df['month'] = datetime.datetime.now().month

            # Process team data - convert to categorical and then to numeric
            # This creates a numeric ID for each team
            df['home_team_id'] = pd.Categorical(df['home_team']).codes
            df['away_team_id'] = pd.Categorical(df['away_team']).codes
            df['league_id'] = pd.Categorical(df['league']).codes

            # Process odds if available
            for col in ['home_odds', 'draw_odds', 'away_odds']:
                if col in df.columns:
                    # Convert to numeric, filling NaN with median values
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # Add default values if column doesn't exist
                    df[col] = 2.0

            # Calculate additional features
            df['odds_ratio'] = df['home_odds'] / df['away_odds']

            # Create final feature set
            features = [
                'home_team_id', 'away_team_id', 'league_id',
                'day_of_week', 'month', 'home_odds', 'draw_odds',
                'away_odds', 'odds_ratio'
            ]

            # Ensure all features are present
            for feature in features:
                if feature not in df.columns:
                    logger.warning(f"Feature {feature} not found, adding default values")
                    df[feature] = 0

            # Create a mapping file for teams and leagues
            mappings = {
                'home_team': dict(zip(df['home_team_id'], df['home_team'])),
                'away_team': dict(zip(df['away_team_id'], df['away_team'])),
                'league': dict(zip(df['league_id'], df['league']))
            }

            # Save mappings
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            mapping_path = os.path.join(self.output_dir, f"sports_mappings_{timestamp}.json")

            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(mappings, f, indent=4)

            logger.info(f"Saved sports data mappings to {mapping_path}")

            # Return processed data
            processed_df = df[features + ['home_team', 'away_team', 'league']]

            # Save processed data
            output_path = os.path.join(self.output_dir, f"processed_sports_{timestamp}.csv")
            processed_df.to_csv(output_path, index=False)

            logger.info(f"Processed sports data saved to {output_path} ({len(processed_df)} entries)")

            return processed_df

        except Exception as e:
            logger.error(f"Error processing sports data: {str(e)}")
            return pd.DataFrame()

    def process_betting_data(self, sports_data_path: str, market_data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Process betting data by combining sports fixtures with market/financial data.

        Args:
            sports_data_path: Path to processed sports data
            market_data_path: Path to market data (optional)

        Returns:
            DataFrame with processed betting data
        """
        logger.info(f"Processing betting data from {sports_data_path}")

        try:
            # Load the processed sports data
            sports_df = pd.read_csv(sports_data_path)

            # Create betting features
            betting_df = sports_df.copy()

            # Add betting-specific features
            betting_df['implied_prob_home'] = 1 / betting_df['home_odds']
            betting_df['implied_prob_away'] = 1 / betting_df['away_odds']
            betting_df['overround'] = betting_df['implied_prob_home'] + betting_df['implied_prob_away']
            if 'draw_odds' in betting_df.columns:
                betting_df['implied_prob_draw'] = 1 / betting_df['draw_odds']
                betting_df['overround'] += betting_df['implied_prob_draw']

            # Add simulated betting features
            betting_df['volatility'] = np.random.uniform(0.1, 0.5, size=len(betting_df))
            betting_df['market_sentiment'] = np.random.uniform(-1, 1, size=len(betting_df))
            betting_df['historical_roi'] = np.random.uniform(-0.2, 0.4, size=len(betting_df))

            # Add market data if available
            if market_data_path and os.path.exists(market_data_path):
                logger.info(f"Adding market data from {market_data_path}")
                try:
                    market_df = pd.read_csv(market_data_path)

                    # Get latest market indicators
                    market_indicators = {}

                    # Extract forex data
                    forex_data = market_df[market_df['data_type'] == 'forex']
                    if not forex_data.empty:
                        for _, row in forex_data.iterrows():
                            market_indicators[f"forex_{row['symbol']}"] = row['value']

                    # Extract stock market sentiment
                    stock_data = market_df[market_df['data_type'] == 'stock']
                    if not stock_data.empty:
                        # Calculate average stock change
                        stock_data['change_pct'] = (stock_data['close'] - stock_data['open']) / stock_data['open']
                        market_indicators['stock_market_sentiment'] = stock_data['change_pct'].mean()

                    # Extract crypto market sentiment
                    crypto_data = market_df[market_df['data_type'] == 'crypto']
                    if not crypto_data.empty:
                        market_indicators['crypto_market_sentiment'] = crypto_data['change_24h'].mean() / 100

                    # Add market indicators to betting data
                    for indicator, value in market_indicators.items():
                        betting_df[indicator] = value

                    logger.info(f"Added {len(market_indicators)} market indicators to betting data")

                except Exception as e:
                    logger.error(f"Error processing market data: {str(e)}")

            # Get final features
            betting_features = [col for col in betting_df.columns if col not in ['home_team', 'away_team', 'league']]

            # Save processed betting data
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"processed_betting_{timestamp}.csv")
            betting_df.to_csv(output_path, index=False)

            logger.info(f"Processed betting data saved to {output_path} ({len(betting_df)} entries)")

            return betting_df

        except Exception as e:
            logger.error(f"Error processing betting data: {str(e)}")
            return pd.DataFrame()

    def prepare_for_sports_model(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare processed data for the sports prediction model.

        Args:
            data: Processed sports data

        Returns:
            Tuple of (feature array, feature names)
        """
        # Features for sports model
        feature_cols = [
            'home_team_id', 'away_team_id', 'league_id',
            'day_of_week', 'month', 'home_odds', 'away_odds',
            'odds_ratio'
        ]

        # Ensure all features are present
        for col in feature_cols:
            if col not in data.columns:
                logger.warning(f"Feature column {col} not found in data")
                return np.array([]), []

        # Convert to numpy array
        X = data[feature_cols].values

        return X, feature_cols

    def prepare_for_betting_model(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare processed data for the betting prediction model.

        Args:
            data: Processed betting data

        Returns:
            Tuple of (feature array, feature names)
        """
        # Exclude non-feature columns
        exclude_cols = ['home_team', 'away_team', 'league']
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        # Convert to numpy array
        X = data[feature_cols].values

        return X, feature_cols

    def combine_data_sources(self, sports_data: str, market_data: str = None) -> Dict[str, str]:
        """
        Process and combine multiple data sources for model training.

        Args:
            sports_data: Path to sports data CSV
            market_data: Path to market data CSV (optional)

        Returns:
            Dictionary with paths to processed data files
        """
        logger.info("Combining data sources for model training")

        result = {}

        try:
            # Process sports data
            sports_df = self.process_sports_data(sports_data)
            if not sports_df.empty:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                sports_output = os.path.join(self.output_dir, f"sports_training_{timestamp}.csv")
                sports_df.to_csv(sports_output, index=False)
                result['sports_data'] = sports_output

                # Process betting data
                betting_df = self.process_betting_data(sports_output, market_data)
                if not betting_df.empty:
                    betting_output = os.path.join(self.output_dir, f"betting_training_{timestamp}.csv")
                    betting_df.to_csv(betting_output, index=False)
                    result['betting_data'] = betting_output

            logger.info(f"Data sources combined successfully: {result}")

        except Exception as e:
            logger.error(f"Error combining data sources: {str(e)}")

        return result

def process_scraped_data(sports_data_path: str, market_data_path: str = None, output_dir: str = None) -> Dict[str, str]:
    """
    Utility function to process scraped data.

    Args:
        sports_data_path: Path to sports data CSV
        market_data_path: Path to market data CSV (optional)
        output_dir: Directory to save processed data

    Returns:
        Dictionary with paths to processed data files
    """
    processor = DataProcessor(output_dir)
    return processor.combine_data_sources(sports_data_path, market_data_path)

def main():
    """Main function to demonstrate data processor usage."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    try:
        logger.info("Starting data processing")

        # Default paths - adjust as needed
        sports_data_path = os.path.join('data', 'scraped_data', 'football_fixtures_latest.csv')
        market_data_path = os.path.join('data', 'fetched_data', 'processed_market_data_latest.csv')

        # Check if latest data files exist
        if not os.path.exists(sports_data_path):
            # Try to find the most recent sports data file
            scraped_dir = os.path.join('data', 'scraped_data')
            if os.path.exists(scraped_dir):
                csv_files = [f for f in os.listdir(scraped_dir) if f.endswith('.csv') and 'fixtures' in f]
                if csv_files:
                    # Sort by modification time (most recent first)
                    latest_file = sorted(csv_files, key=lambda f: os.path.getmtime(os.path.join(scraped_dir, f)), reverse=True)[0]
                    sports_data_path = os.path.join(scraped_dir, latest_file)
                    logger.info(f"Using most recent sports data file: {sports_data_path}")

        # Process the data
        if os.path.exists(sports_data_path):
            result = process_scraped_data(sports_data_path, market_data_path)
            logger.info(f"Data processing completed: {result}")
        else:
            logger.error(f"Sports data file not found: {sports_data_path}")

    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")

if __name__ == "__main__":
    main()

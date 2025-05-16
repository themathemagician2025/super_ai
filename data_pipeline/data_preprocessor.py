# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

# Set up logging
logger = logging.getLogger(__name__)

# Constants and configuration
DATA_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data'))
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = DATA_DIR / 'models'

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)


class DataPreprocessor:
    """
    Handles data preprocessing, normalization, feature engineering,
    and missing data imputation for the Super AI Prediction System.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DataPreprocessor with optional configuration.

        Args:
            config: Configuration dictionary with preprocessing settings
        """
        self.config = config or {}
        self.scalers = {}

    def normalize_data(self, data: pd.DataFrame, method: str = 'minmax',
                      columns: Optional[List[str]] = None,
                      save_scaler: bool = True) -> pd.DataFrame:
        """
        Normalize/standardize data.

        Args:
            data: DataFrame to normalize
            method: Normalization method ('minmax' or 'standard')
            columns: Specific columns to normalize (if None, all numeric columns)
            save_scaler: Whether to save the scaler for later use

        Returns:
            Normalized DataFrame
        """
        if data.empty:
            logger.warning("Empty DataFrame provided for normalization")
            return data

        # If no specific columns provided, use all numeric columns
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            logger.warning("No numeric columns found for normalization")
            return data

        logger.info(f"Normalizing {len(columns)} columns using {method} method")

        # Create a copy to avoid modifying the original DataFrame
        normalized_data = data.copy()

        # Create a unique identifier for this scaling operation
        scaling_id = f"{','.join(columns)}_{method}"

        # Create the appropriate scaler
        if method.lower() == 'minmax':
            scaler = MinMaxScaler()
        elif method.lower() == 'standard':
            scaler = StandardScaler()
        else:
            logger.warning(f"Unknown normalization method: {method}. Using MinMaxScaler.")
            scaler = MinMaxScaler()

        # Apply the scaler only to the specified columns
        try:
            # Extract only the columns we want to normalize
            subset = normalized_data[columns].values

            # Handle all-NaN columns (which would cause StandardScaler to fail)
            if np.isnan(subset).all(axis=0).any():
                logger.warning("Some columns contain only NaN values. Filling with zeros.")
                subset = np.nan_to_num(subset, nan=0.0)

            # Fit and transform
            normalized_subset = scaler.fit_transform(subset)

            # Put the normalized values back into the DataFrame
            normalized_data[columns] = normalized_subset

            # Save the scaler if requested
            if save_scaler:
                self.scalers[scaling_id] = scaler
                scaler_path = MODELS_DIR / f"scaler_{scaling_id.replace(',', '_')}.joblib"
                joblib.dump(scaler, scaler_path)
                logger.info(f"Saved scaler to {scaler_path}")

            return normalized_data

        except Exception as e:
            logger.error(f"Error during normalization: {e}")
            return data

    def handle_missing_data(self, data: pd.DataFrame, method: str = 'interpolate',
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing data in DataFrame.

        Args:
            data: DataFrame with missing values
            method: Method to handle missing values ('drop', 'interpolate', 'forward_fill', 'mean')
            columns: Specific columns to process (if None, all columns)

        Returns:
            DataFrame with missing values handled
        """
        if data.empty:
            logger.warning("Empty DataFrame provided for missing data handling")
            return data

        # Get the columns to process
        if columns is None:
            columns = data.columns.tolist()

        logger.info(f"Handling missing data in {len(columns)} columns using {method} method")

        # Create a copy to avoid modifying the original DataFrame
        processed_data = data.copy()

        # Get initial missing value count
        initial_missing = processed_data[columns].isna().sum().sum()

        try:
            if method == 'drop':
                # Drop rows with any missing values in the specified columns
                processed_data.dropna(subset=columns, inplace=True)
                logger.info(f"Dropped rows with missing values. {len(processed_data)} rows remaining.")

            elif method == 'interpolate':
                # Linear interpolation
                processed_data[columns] = processed_data[columns].interpolate(
                    method='linear', limit_direction='both')

                # If there are still NaNs at the beginning or end, fill them
                processed_data[columns] = processed_data[columns].fillna(method='bfill').fillna(method='ffill')

            elif method == 'forward_fill':
                # Forward fill followed by backward fill for any remaining NaNs
                processed_data[columns] = processed_data[columns].fillna(method='ffill').fillna(method='bfill')

            elif method == 'mean':
                # Fill with column means (for numeric columns only)
                numeric_cols = processed_data[columns].select_dtypes(include=[np.number]).columns.tolist()

                for col in numeric_cols:
                    col_mean = processed_data[col].mean()
                    processed_data[col].fillna(col_mean, inplace=True)

                # For non-numeric columns, use forward fill
                non_numeric_cols = list(set(columns) - set(numeric_cols))
                if non_numeric_cols:
                    processed_data[non_numeric_cols] = processed_data[non_numeric_cols].fillna(
                        method='ffill').fillna(method='bfill')
            else:
                logger.warning(f"Unknown missing data method: {method}. No action taken.")

            # Report on remaining missing values
            remaining_missing = processed_data[columns].isna().sum().sum()
            logger.info(f"Missing values: {initial_missing} before, {remaining_missing} after")

            return processed_data

        except Exception as e:
            logger.error(f"Error handling missing data: {e}")
            return data

    def remove_outliers(self, data: pd.DataFrame, method: str = 'zscore',
                       threshold: float = 3.0,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove outliers from the dataset.

        Args:
            data: DataFrame to process
            method: Method to detect outliers ('zscore' or 'iqr')
            threshold: Threshold for outlier detection
            columns: Specific columns to check (if None, all numeric columns)

        Returns:
            DataFrame with outliers removed
        """
        if data.empty:
            logger.warning("Empty DataFrame provided for outlier removal")
            return data

        # If no specific columns provided, use all numeric columns
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            logger.warning("No numeric columns found for outlier removal")
            return data

        logger.info(f"Removing outliers from {len(columns)} columns using {method} method")

        # Create a copy to avoid modifying the original DataFrame
        clean_data = data.copy()
        initial_rows = len(clean_data)

        try:
            if method == 'zscore':
                # Z-score method
                for col in columns:
                    if clean_data[col].std() == 0:
                        logger.warning(f"Column {col} has zero standard deviation. Skipping.")
                        continue

                    # Calculate z-scores
                    z_scores = np.abs((clean_data[col] - clean_data[col].mean()) / clean_data[col].std())

                    # Filter out rows with high z-scores
                    clean_data = clean_data[z_scores < threshold]

            elif method == 'iqr':
                # IQR method
                for col in columns:
                    # Calculate IQR
                    Q1 = clean_data[col].quantile(0.25)
                    Q3 = clean_data[col].quantile(0.75)
                    IQR = Q3 - Q1

                    # Define bounds
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR

                    # Filter out rows outside the bounds
                    clean_data = clean_data[(clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)]
            else:
                logger.warning(f"Unknown outlier removal method: {method}. No action taken.")

            # Report on removed rows
            removed_rows = initial_rows - len(clean_data)
            logger.info(f"Removed {removed_rows} outliers ({removed_rows/initial_rows*100:.2f}%)")

            return clean_data

        except Exception as e:
            logger.error(f"Error removing outliers: {e}")
            return data

    def add_forex_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators for forex data.

        Args:
            data: DataFrame with OHLC(V) data

        Returns:
            DataFrame with additional features
        """
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            logger.warning("Cannot add forex features. Missing required OHLC columns.")
            return data

        logger.info("Adding technical indicators for forex data")

        # Create a copy to avoid modifying the original DataFrame
        df = data.copy()

        try:
            # Calculate Moving Averages
            windows = [5, 10, 20, 50, 200]
            for window in windows:
                df[f'ma_{window}'] = df['close'].rolling(window=window).mean()

            # Calculate Exponential Moving Averages
            for window in [12, 26]:
                df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()

            # Calculate MACD (Moving Average Convergence Divergence)
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # Calculate RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # Calculate Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

            # Calculate volatility indicators
            df['atr'] = df['high'] - df['low']  # Simplified ATR
            df['daily_range_pct'] = (df['high'] - df['low']) / df['close'] * 100

            # Price momentum
            for period in [1, 5, 10, 20]:
                df[f'momentum_{period}'] = df['close'].pct_change(periods=period) * 100

            # Volume indicators (if volume is available)
            if 'volume' in df.columns:
                df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
                df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma_20']

            # Add day of week (if a date column is available)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df['day_of_week'] = df['date'].dt.dayofweek

            logger.info(f"Added {len(df.columns) - len(data.columns)} new technical indicators")

            return df

        except Exception as e:
            logger.error(f"Error adding forex features: {e}")
            return data

    def add_sports_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add features for sports betting data.

        Args:
            data: DataFrame with sports match data

        Returns:
            DataFrame with additional features
        """
        required_cols = ['home_team', 'away_team']
        if not all(col in data.columns for col in required_cols):
            logger.warning("Cannot add sports features. Missing team columns.")
            return data

        logger.info("Adding features for sports betting data")

        # Create a copy to avoid modifying the original DataFrame
        df = data.copy()

        try:
            # Convert date to datetime if present
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df['day_of_week'] = df['date'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # 5=Sat, 6=Sun
                df['month'] = df['date'].dt.month  # Season effects
                df['year'] = df['date'].dt.year    # Long-term trends

            # Add sport-specific features
            if 'sport' in df.columns:
                # Convert sport names to lowercase for consistency
                df['sport'] = df['sport'].str.lower()

                # American Football specific features
                if any(df['sport'].str.contains('football')):
                    if 'total_yards' in df.columns and 'turnovers' in df.columns:
                        df['yards_per_turnover'] = df['total_yards'] / (df['turnovers'] + 0.5)
                    if 'pass_attempts' in df.columns and 'pass_completions' in df.columns:
                        df['completion_percentage'] = df['pass_completions'] / df['pass_attempts'] * 100

                # Basketball specific features
                if any(df['sport'].str.contains('basketball')):
                    if 'field_goals_attempted' in df.columns and 'field_goals_made' in df.columns:
                        df['field_goal_percentage'] = df['field_goals_made'] / df['field_goals_attempted'] * 100
                    if 'three_pointers_attempted' in df.columns and 'three_pointers_made' in df.columns:
                        df['three_point_percentage'] = df['three_pointers_made'] / df['three_pointers_attempted'] * 100

                # Baseball specific features
                if any(df['sport'].str.contains('baseball')):
                    if 'hits' in df.columns and 'at_bats' in df.columns:
                        df['batting_average'] = df['hits'] / df['at_bats']
                    if 'earned_runs' in df.columns and 'innings_pitched' in df.columns:
                        df['era'] = df['earned_runs'] * 9 / df['innings_pitched']

                # Soccer specific features
                if any(df['sport'].str.contains('soccer')):
                    if 'shots' in df.columns and 'goals' in df.columns:
                        df['shot_conversion'] = df['goals'] / df['shots']
                    if 'possession' in df.columns:
                        df['possession_efficiency'] = df['goals'] / (df['possession'] * 0.01)

                # Tennis specific features
                if any(df['sport'].str.contains('tennis')):
                    if 'first_serve_in' in df.columns and 'first_serve_attempts' in df.columns:
                        df['first_serve_percentage'] = df['first_serve_in'] / df['first_serve_attempts'] * 100
                    if 'break_points_saved' in df.columns and 'break_points_faced' in df.columns:
                        df['break_point_save_percentage'] = df['break_points_saved'] / df['break_points_faced'] * 100

                # Hockey specific features
                if any(df['sport'].str.contains('hockey')):
                    if 'shots' in df.columns and 'goals' in df.columns:
                        df['shooting_percentage'] = df['goals'] / df['shots'] * 100
                    if 'power_play_opportunities' in df.columns and 'power_play_goals' in df.columns:
                        df['power_play_percentage'] = df['power_play_goals'] / df['power_play_opportunities'] * 100

                # Cricket specific features
                if any(df['sport'].str.contains('cricket')):
                    if 'runs' in df.columns and 'balls_faced' in df.columns:
                        df['run_rate'] = df['runs'] / (df['balls_faced'] / 6)  # Runs per over
                    if 'wickets' in df.columns and 'overs_bowled' in df.columns:
                        df['economy_rate'] = df['runs_conceded'] / df['overs_bowled']

                # MMA and Boxing features
                if any(df['sport'].str.contains('mma|boxing')):
                    if 'significant_strikes_landed' in df.columns and 'significant_strikes_attempted' in df.columns:
                        df['striking_accuracy'] = df['significant_strikes_landed'] / df['significant_strikes_attempted'] * 100
                    if 'takedown_defense' in df.columns:
                        df['grappling_defense'] = df['takedown_defense']

            # Basic result features (if score columns exist)
            score_cols = ['home_score', 'away_score']
            if all(col in df.columns for col in score_cols):
                df['total_score'] = df['home_score'] + df['away_score']
                df['score_diff'] = df['home_score'] - df['away_score']
                df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
                df['draw'] = (df['home_score'] == df['away_score']).astype(int)
                df['away_win'] = (df['home_score'] < df['away_score']).astype(int)

                # Create result category
                df['result'] = np.select(
                    [df['home_win'] == 1, df['draw'] == 1, df['away_win'] == 1],
                    ['home_win', 'draw', 'away_win']
                )

            # Team form features (requires historical data)
            if 'date' in df.columns and len(df) > 5:
                # Sort by date
                df = df.sort_values('date')

                # Add home and away team form (last 5 matches)
                teams = pd.concat([df['home_team'], df['away_team']]).unique()

                # Initialize form columns
                df['home_team_form'] = np.nan
                df['away_team_form'] = np.nan

                for team in teams:
                    # Get all matches for this team
                    home_matches = df[df['home_team'] == team].copy()
                    away_matches = df[df['away_team'] == team].copy()

                    if not (len(home_matches) > 0 or len(away_matches) > 0):
                        continue

                    # Add result from team perspective
                    if 'result' in df.columns:
                        home_matches['team_result'] = home_matches['result'].apply(
                            lambda x: 'win' if x == 'home_win' else ('draw' if x == 'draw' else 'loss')
                        )
                        away_matches['team_result'] = away_matches['result'].apply(
                            lambda x: 'win' if x == 'away_win' else ('draw' if x == 'draw' else 'loss')
                        )

                        # Combine home and away matches
                        team_matches = pd.concat([home_matches, away_matches])
                        team_matches = team_matches.sort_values('date')

                        # Calculate form (points in last 5 matches)
                        team_matches['form_points'] = team_matches['team_result'].apply(
                            lambda x: 3 if x == 'win' else (1 if x == 'draw' else 0)
                        )

                        # Calculate rolling form
                        team_matches['rolling_form'] = team_matches['form_points'].rolling(5, min_periods=1).sum()

                        # Map form back to original DataFrame
                        for idx, row in team_matches.iterrows():
                            match_date = row['date']
                            form = row['rolling_form']

                            # Find matches where this team plays after this date
                            future_home_matches = df[(df['home_team'] == team) & (df['date'] > match_date)]
                            future_away_matches = df[(df['away_team'] == team) & (df['date'] > match_date)]

                            # Update the form for the next match
                            if len(future_home_matches) > 0:
                                next_home_match = future_home_matches.iloc[0].name
                                df.at[next_home_match, 'home_team_form'] = form

                            if len(future_away_matches) > 0:
                                next_away_match = future_away_matches.iloc[0].name
                                df.at[next_away_match, 'away_team_form'] = form

            # Add rivalry intensity feature
            if 'home_team' in df.columns and 'away_team' in df.columns:
                team_pairs = {}
                for _, row in df.iterrows():
                    pair = tuple(sorted([row['home_team'], row['away_team']]))
                    if pair in team_pairs:
                        team_pairs[pair] += 1
                    else:
                        team_pairs[pair] = 1

                df['rivalry_intensity'] = df.apply(
                    lambda row: team_pairs[tuple(sorted([row['home_team'], row['away_team']]))],
                    axis=1
                )

            # Add weather features if available
            weather_cols = ['temperature', 'is_rainy', 'wind_speed', 'humidity']
            if any(col in df.columns for col in weather_cols):
                # Convert boolean to int if needed
                if 'is_rainy' in df.columns and df['is_rainy'].dtype == bool:
                    df['is_rainy'] = df['is_rainy'].astype(int)

                # Add weather impact score (simple weighted sum)
                weather_impact = 0
                if 'temperature' in df.columns:
                    # Normalize temperature to 0-1 range (assuming 0-30C range)
                    temp_impact = 1 - abs(df['temperature'] - 20) / 20
                    weather_impact += temp_impact

                if 'is_rainy' in df.columns:
                    weather_impact -= df['is_rainy'] * 0.2  # Penalty for rain

                if 'wind_speed' in df.columns:
                    # Normalize wind speed (assuming 0-50 km/h range)
                    wind_impact = 1 - df['wind_speed'] / 50
                    weather_impact += wind_impact

                df['weather_impact'] = weather_impact

            logger.info(f"Added {len(df.columns) - len(data.columns)} new sports features")

            return df

        except Exception as e:
            logger.error(f"Error adding sports features: {e}")
            return data

    def time_series_alignment(self, data: pd.DataFrame, date_column: str = 'date',
                            freq: str = 'D') -> pd.DataFrame:
        """
        Align time series data to a regular frequency.

        Args:
            data: DataFrame with time series data
            date_column: Name of the date column
            freq: Pandas frequency string ('D' for daily, 'W' for weekly, etc.)

        Returns:
            Aligned DataFrame
        """
        if date_column not in data.columns:
            logger.warning(f"Cannot align time series. Date column '{date_column}' not found.")
            return data

        logger.info(f"Aligning time series data to {freq} frequency")

        # Create a copy to avoid modifying the original DataFrame
        df = data.copy()

        try:
            # Convert date column to datetime if it's not already
            df[date_column] = pd.to_datetime(df[date_column])

            # Set date as index
            df = df.set_index(date_column)

            # Resample to the specified frequency
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if numeric_cols:
                # For numeric columns, use mean
                df_numeric = df[numeric_cols].resample(freq).mean()

                # For non-numeric columns, use first value
                non_numeric_cols = list(set(df.columns) - set(numeric_cols))
                df_non_numeric = df[non_numeric_cols].resample(freq).first() if non_numeric_cols else pd.DataFrame()

                # Combine the results
                if len(non_numeric_cols) > 0:
                    df_aligned = pd.concat([df_numeric, df_non_numeric], axis=1)
                else:
                    df_aligned = df_numeric
            else:
                # If no numeric columns, use first value for all columns
                df_aligned = df.resample(freq).first()

            # Reset index to get the date column back
            df_aligned = df_aligned.reset_index()

            logger.info(f"Aligned data from {len(data)} rows to {len(df_aligned)} rows")

            return df_aligned

        except Exception as e:
            logger.error(f"Error aligning time series: {e}")
            return data

    def preprocess_dataset(self, data: pd.DataFrame, domain: str = 'forex',
                        config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Apply a complete preprocessing pipeline to a dataset.

        Args:
            data: Input DataFrame
            domain: Data domain ('forex' or 'sports')
            config: Preprocessing configuration

        Returns:
            Fully preprocessed DataFrame
        """
        if data.empty:
            logger.warning("Empty DataFrame provided for preprocessing")
            return data

        # Use provided config or default
        config = config or self.config or {}

        logger.info(f"Starting preprocessing pipeline for {domain} data with {len(data)} rows")

        # Create a processing copy
        processed = data.copy()

        try:
            # Step 1: Handle missing data
            if config.get('handle_missing', True):
                missing_method = config.get('missing_method', 'interpolate')
                processed = self.handle_missing_data(processed, method=missing_method)

            # Step 2: Remove outliers
            if config.get('remove_outliers', True):
                outlier_method = config.get('outlier_method', 'zscore')
                outlier_threshold = config.get('outlier_threshold', 3.0)
                processed = self.remove_outliers(processed, method=outlier_method, threshold=outlier_threshold)

            # Step 3: Add domain-specific features
            if config.get('add_features', True):
                if domain == 'forex':
                    processed = self.add_forex_features(processed)
                elif domain == 'sports':
                    processed = self.add_sports_features(processed)
                else:
                    logger.warning(f"Unknown domain: {domain}. No features added.")

            # Step 4: Time series alignment (if applicable)
            if config.get('align_time_series', False) and 'date' in processed.columns:
                time_freq = config.get('time_freq', 'D')
                processed = self.time_series_alignment(processed, freq=time_freq)

            # Step 5: Normalize data
            if config.get('normalize', True):
                scaling_method = config.get('scaling_method', 'minmax')
                # Skip normalization for certain columns
                skip_cols = config.get('skip_normalization', [])
                numeric_cols = processed.select_dtypes(include=[np.number]).columns.tolist()
                cols_to_normalize = [col for col in numeric_cols if col not in skip_cols]
                processed = self.normalize_data(processed, method=scaling_method, columns=cols_to_normalize)

            # Add preprocessing metadata
            processed['preprocessing_timestamp'] = datetime.now().isoformat()
            processed['preprocessing_version'] = '1.0'

            # Report on the preprocessing results
            logger.info(f"Preprocessing complete. Output has {len(processed)} rows and {len(processed.columns)} columns")

            return processed

        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            return data

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()

    # Create sample forex data
    sample_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100),
        'open': np.random.normal(1.2, 0.01, 100),
        'high': np.random.normal(1.21, 0.01, 100),
        'low': np.random.normal(1.19, 0.01, 100),
        'close': np.random.normal(1.2, 0.01, 100),
        'volume': np.random.randint(1000, 5000, 100)
    })

    # Add some missing values
    sample_data.loc[10:15, 'close'] = np.nan

    # Add some outliers
    sample_data.loc[50, 'high'] = 1.5

    # Process the data
    config = {
        'handle_missing': True,
        'missing_method': 'interpolate',
        'remove_outliers': True,
        'outlier_method': 'zscore',
        'add_features': True,
        'normalize': True,
        'scaling_method': 'minmax',
        'skip_normalization': ['date', 'volume']
    }

    processed_data = preprocessor.preprocess_dataset(sample_data, domain='forex', config=config)

    # Print some info about the processed data
    print(f"Original shape: {sample_data.shape}")
    print(f"Processed shape: {processed_data.shape}")
    print(f"Added features: {set(processed_data.columns) - set(sample_data.columns)}")

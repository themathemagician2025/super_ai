# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
import random
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAugmenter:
    """
    Class for implementing data augmentation techniques to enhance data quality
    and improve model robustness through synthetic data generation.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize the DataAugmenter with a random seed for reproducibility.

        Args:
            random_seed: Seed for random number generation to ensure reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        logger.info(f"DataAugmenter initialized with random seed {random_seed}")

    def add_gaussian_noise(self, data: pd.DataFrame, columns: List[str],
                          mean: float = 0.0, std_dev: float = 0.01) -> pd.DataFrame:
        """
        Add Gaussian noise to numerical columns in the DataFrame.

        Args:
            data: Input DataFrame
            columns: List of column names to add noise to
            mean: Mean of the Gaussian noise distribution
            std_dev: Standard deviation of the Gaussian noise distribution

        Returns:
            DataFrame with added noise
        """
        if data.empty:
            logger.warning("Empty DataFrame provided. Returning original data.")
            return data

        # Create a copy to avoid modifying the original data
        augmented_data = data.copy()

        for col in columns:
            if col in augmented_data.columns and pd.api.types.is_numeric_dtype(augmented_data[col]):
                # Generate noise based on the column values
                noise = np.random.normal(mean, std_dev, size=len(augmented_data)) * augmented_data[col].abs().mean()
                augmented_data[col] = augmented_data[col] + noise
                logger.debug(f"Added Gaussian noise to column {col}")
            else:
                logger.warning(f"Column {col} not found or not numeric. Skipping.")

        logger.info(f"Added Gaussian noise to {len(columns)} columns")
        return augmented_data

    def generate_synthetic_forex_data(self, template_data: pd.DataFrame,
                                      num_samples: int = 10,
                                      start_date: Optional[str] = None,
                                      volatility_factor: float = 1.0) -> pd.DataFrame:
        """
        Generate synthetic forex data based on statistical properties of existing data.

        Args:
            template_data: DataFrame with existing forex data to use as template
            num_samples: Number of synthetic samples to generate
            start_date: Starting date for synthetic data (format: 'YYYY-MM-DD')
            volatility_factor: Factor to adjust the volatility of generated data

        Returns:
            DataFrame with synthetic forex data
        """
        if template_data.empty:
            logger.error("Empty template data provided. Cannot generate synthetic data.")
            return pd.DataFrame()

        required_columns = ['date', 'pair', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in template_data.columns]

        if missing_columns:
            logger.error(f"Template data missing required columns: {missing_columns}")
            return pd.DataFrame()

        # Extract statistical properties from template data
        template_close = template_data['close'].values
        template_volume = template_data['volume'].values

        # Calculate typical price changes and volatility
        pct_changes = np.diff(template_close) / template_close[:-1]
        mean_change = np.mean(pct_changes)
        std_change = np.std(pct_changes) * volatility_factor

        # Volume statistics
        volume_mean = np.mean(template_volume)
        volume_std = np.std(template_volume)

        # Determine date range
        if start_date:
            try:
                current_date = datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                logger.warning(f"Invalid date format for {start_date}. Using current date.")
                current_date = datetime.now()
        else:
            # Get the day after the last date in template data
            last_date = pd.to_datetime(template_data['date']).max()
            current_date = last_date + timedelta(days=1)

        # Generate synthetic data
        synthetic_data = []
        pair = template_data['pair'].iloc[0]  # Use the same currency pair

        prev_close = template_data['close'].iloc[-1]

        for i in range(num_samples):
            # Generate today's price movement
            daily_return = np.random.normal(mean_change, std_change)
            close = prev_close * (1 + daily_return)

            # Generate intraday range (high and low)
            avg_range_pct = np.mean((template_data['high'] - template_data['low']) / template_data['close'])
            day_range = close * avg_range_pct * np.random.uniform(0.7, 1.3)

            # Distribute the range around close
            upside_pct = np.random.uniform(0.3, 0.7)  # How much of the range is above close
            high = close + (day_range * upside_pct)
            low = close - (day_range * (1 - upside_pct))

            # Generate open price between yesterday's close and today's close
            open_weight = np.random.uniform(0.3, 0.7)
            open_price = prev_close + (close - prev_close) * open_weight

            # Ensure OHLC relationship is maintained
            high = max(high, open_price, close)
            low = min(low, open_price, close)

            # Generate volume with some randomness
            volume = int(np.random.normal(volume_mean, volume_std))
            volume = max(100, volume)  # Ensure volume is positive

            # Add sentiment if it exists in the template
            sentiment = None
            if 'sentiment' in template_data.columns:
                sentiment_mean = template_data['sentiment'].mean()
                sentiment_std = template_data['sentiment'].std()
                sentiment = np.clip(np.random.normal(sentiment_mean, sentiment_std), 0, 1)

            # Create the record
            record = {
                'date': current_date.strftime('%Y-%m-%d'),
                'pair': pair,
                'open': round(open_price, 4),
                'high': round(high, 4),
                'low': round(low, 4),
                'close': round(close, 4),
                'volume': volume
            }

            if sentiment is not None:
                record['sentiment'] = round(sentiment, 2)

            synthetic_data.append(record)

            # Move to next day
            current_date += timedelta(days=1)
            if current_date.weekday() >= 5:  # Skip weekends for forex
                current_date += timedelta(days=8 - current_date.weekday())

            prev_close = close

        synthetic_df = pd.DataFrame(synthetic_data)
        logger.info(f"Generated {len(synthetic_df)} synthetic forex data points")
        return synthetic_df

    def generate_synthetic_sports_data(self, template_data: pd.DataFrame,
                                      num_samples: int = 5,
                                      start_date: Optional[str] = None,
                                      teams: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate synthetic sports match data based on statistical properties of existing data.

        Args:
            template_data: DataFrame with existing sports data to use as template
            num_samples: Number of synthetic matches to generate
            start_date: Starting date for synthetic data (format: 'YYYY-MM-DD')
            teams: List of team names to use (if None, uses teams from template)

        Returns:
            DataFrame with synthetic sports data
        """
        if template_data.empty:
            logger.error("Empty template data provided. Cannot generate synthetic sports data.")
            return pd.DataFrame()

        required_columns = ['date', 'league', 'home_team', 'away_team', 'home_score', 'away_score']
        missing_columns = [col for col in required_columns if col not in template_data.columns]

        if missing_columns:
            logger.error(f"Template data missing required columns: {missing_columns}")
            return pd.DataFrame()

        # Extract teams and their stats from template data
        if teams is None:
            teams = list(set(template_data['home_team'].unique()) | set(template_data['away_team'].unique()))

        # Calculate team statistics
        team_stats = {}
        for team in teams:
            # Home statistics
            home_matches = template_data[template_data['home_team'] == team]
            # Away statistics
            away_matches = template_data[template_data['away_team'] == team]

            # Calculate average scores
            avg_home_scored = home_matches['home_score'].mean() if not home_matches.empty else 1.5
            avg_home_conceded = home_matches['away_score'].mean() if not home_matches.empty else 1.0
            avg_away_scored = away_matches['away_score'].mean() if not away_matches.empty else 1.0
            avg_away_conceded = away_matches['home_score'].mean() if not away_matches.empty else 1.5

            # Additional stats if available
            possession_stats = {}
            if 'home_possession' in template_data.columns:
                possession_stats['home_possession'] = home_matches['home_possession'].mean() if not home_matches.empty else 50
            if 'away_possession' in template_data.columns:
                possession_stats['away_possession'] = away_matches['away_possession'].mean() if not away_matches.empty else 50

            shots_stats = {}
            if 'home_shots' in template_data.columns:
                shots_stats['home_shots'] = home_matches['home_shots'].mean() if not home_matches.empty else 10
            if 'away_shots' in template_data.columns:
                shots_stats['away_shots'] = away_matches['away_shots'].mean() if not away_matches.empty else 10

            corners_stats = {}
            if 'home_corners' in template_data.columns:
                corners_stats['home_corners'] = home_matches['home_corners'].mean() if not home_matches.empty else 5
            if 'away_corners' in template_data.columns:
                corners_stats['away_corners'] = away_matches['away_corners'].mean() if not away_matches.empty else 5

            # Store team stats
            team_stats[team] = {
                'avg_home_scored': avg_home_scored,
                'avg_home_conceded': avg_home_conceded,
                'avg_away_scored': avg_away_scored,
                'avg_away_conceded': avg_away_conceded,
                **possession_stats,
                **shots_stats,
                **corners_stats
            }

        # Determine start date
        if start_date:
            try:
                current_date = datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                logger.warning(f"Invalid date format for {start_date}. Using current date.")
                current_date = datetime.now()
        else:
            # Get the day after the last date in template data
            last_date = pd.to_datetime(template_data['date']).max()
            current_date = last_date + timedelta(days=1)

        # Get league from template
        league = template_data['league'].iloc[0]

        # Generate synthetic matches
        synthetic_data = []

        for i in range(num_samples):
            # Select random home and away teams
            home_team, away_team = random.sample(teams, 2)

            # Calculate expected scores based on team stats
            home_expected_goals = (team_stats[home_team]['avg_home_scored'] + team_stats[away_team]['avg_away_conceded']) / 2
            away_expected_goals = (team_stats[away_team]['avg_away_scored'] + team_stats[home_team]['avg_home_conceded']) / 2

            # Generate actual scores with Poisson distribution
            home_score = np.random.poisson(home_expected_goals)
            away_score = np.random.poisson(away_expected_goals)

            # Prepare match record
            match = {
                'date': current_date.strftime('%Y-%m-%d'),
                'league': league,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score
            }

            # Add additional stats if they exist in template data
            if 'home_possession' in template_data.columns:
                home_possession = max(30, min(70, int(np.random.normal(
                    team_stats[home_team].get('home_possession', 50),
                    5))))
                match['home_possession'] = home_possession
                match['away_possession'] = 100 - home_possession

            if 'home_shots' in template_data.columns:
                # Shots usually correlate with goals and possession
                possession_factor = match.get('home_possession', 50) / 50
                home_shots = max(3, int(np.random.normal(
                    team_stats[home_team].get('home_shots', 10) * possession_factor,
                    3)))
                away_shots = max(3, int(np.random.normal(
                    team_stats[away_team].get('away_shots', 10) * (2 - possession_factor),
                    3)))
                match['home_shots'] = home_shots
                match['away_shots'] = away_shots

            if 'home_corners' in template_data.columns:
                # Corners often correlate with shots
                match['home_corners'] = max(0, int(np.random.normal(
                    team_stats[home_team].get('home_corners', 5),
                    2)))
                match['away_corners'] = max(0, int(np.random.normal(
                    team_stats[away_team].get('away_corners', 5),
                    2)))

            if 'attendance' in template_data.columns:
                # Generate attendance with some variation
                avg_attendance = template_data['attendance'].mean()
                std_attendance = template_data['attendance'].std()
                match['attendance'] = max(5000, int(np.random.normal(avg_attendance, std_attendance)))

            if 'weather_condition' in template_data.columns:
                weather_conditions = template_data['weather_condition'].unique()
                match['weather_condition'] = random.choice(weather_conditions)

            synthetic_data.append(match)

            # Move to next weekend for next match (sports games typically on weekends)
            days_to_add = 7 if current_date.weekday() == 5 else 6 - current_date.weekday()
            current_date += timedelta(days=days_to_add)

        synthetic_df = pd.DataFrame(synthetic_data)
        logger.info(f"Generated {len(synthetic_df)} synthetic sports matches")
        return synthetic_df

    def time_series_transformation(self, data: pd.DataFrame, date_column: str = 'date',
                                 transformations: List[str] = ['shift', 'scale']) -> pd.DataFrame:
        """
        Apply time series transformations to generate new variants of the data.

        Args:
            data: Input DataFrame with time series data
            date_column: Name of the column containing dates
            transformations: List of transformations to apply ('shift', 'scale', 'window_warp')

        Returns:
            DataFrame with transformed time series data
        """
        if data.empty:
            logger.warning("Empty DataFrame provided. Returning original data.")
            return data

        if date_column not in data.columns:
            logger.error(f"Date column '{date_column}' not found in data.")
            return data

        # Ensure data is sorted by date
        data = data.sort_values(by=date_column).reset_index(drop=True)

        # Create a copy for transformation
        transformed_data = data.copy()
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col != date_column]

        for transformation in transformations:
            if transformation == 'shift':
                # Shift the time series data by a random number of days
                shift_days = random.randint(1, 5)
                dates = pd.to_datetime(data[date_column])
                transformed_data[date_column] = dates + pd.Timedelta(days=shift_days)

                logger.info(f"Shifted time series data by {shift_days} days")

            elif transformation == 'scale':
                # Apply scaling to numeric columns
                scale_factor = np.random.uniform(0.9, 1.1)  # Random scaling between 90% and 110%

                for col in numeric_columns:
                    transformed_data[col] = data[col] * scale_factor

                logger.info(f"Scaled numeric data by factor {scale_factor:.2f}")

            elif transformation == 'window_warp':
                # Perform window warping (stretching or compressing segments)
                if len(data) > 10:
                    # Select a random window to warp
                    window_size = random.randint(5, min(10, len(data) // 2))
                    start_idx = random.randint(0, len(data) - window_size)

                    # Determine whether to stretch or compress
                    stretch_factor = np.random.uniform(0.7, 1.3)

                    for col in numeric_columns:
                        window_data = data.iloc[start_idx:start_idx + window_size][col].values

                        # Interpolate to new size
                        new_size = int(window_size * stretch_factor)
                        new_size = max(2, new_size)  # Ensure at least 2 points for interpolation

                        # Create original and target indices for interpolation
                        orig_indices = np.linspace(0, 1, window_size)
                        new_indices = np.linspace(0, 1, new_size)

                        # Interpolate the window data
                        warped_window = np.interp(new_indices, orig_indices, window_data)

                        # Replace the original window with the warped version
                        # This requires adjusting the dataframe size
                        if new_size <= window_size:
                            # If compressing, we need to remove rows
                            transformed_data = pd.concat([
                                transformed_data.iloc[:start_idx],
                                pd.DataFrame({col: warped_window}),
                                transformed_data.iloc[start_idx + window_size:]
                            ], ignore_index=True)
                        else:
                            # If stretching, we need to add rows
                            # For simplicity, we'll just replace the window and accept the size change
                            transformed_data = pd.concat([
                                transformed_data.iloc[:start_idx],
                                pd.DataFrame({col: warped_window}),
                                transformed_data.iloc[start_idx + window_size:]
                            ], ignore_index=True)

                    logger.info(f"Applied window warping with stretch factor {stretch_factor:.2f}")
                else:
                    logger.warning("Data too short for window warping. Skipping.")

        return transformed_data

    def balance_classes(self, data: pd.DataFrame, target_column: str,
                       method: str = 'oversample',
                       minority_class: Optional[Union[str, int, float]] = None) -> pd.DataFrame:
        """
        Balance class distribution in the dataset by oversampling minority class or
        undersampling majority class.

        Args:
            data: Input DataFrame
            target_column: Name of the target column (class labels)
            method: Balancing method ('oversample' or 'undersample')
            minority_class: Value of the minority class (if None, automatically determined)

        Returns:
            DataFrame with balanced class distribution
        """
        if data.empty or target_column not in data.columns:
            logger.warning(f"Empty DataFrame or target column '{target_column}' not found. Returning original data.")
            return data

        # Get class distribution
        class_counts = data[target_column].value_counts()

        # If only one class exists, return original data
        if len(class_counts) <= 1:
            logger.warning("Only one class found in the dataset. Cannot balance classes.")
            return data

        # Determine minority and majority classes
        if minority_class is None:
            minority_class = class_counts.idxmin()

        majority_class = class_counts.idxmax() if class_counts.idxmax() != minority_class else class_counts.index[1]

        minority_samples = data[data[target_column] == minority_class]
        majority_samples = data[data[target_column] == majority_class]
        other_samples = data[~data[target_column].isin([minority_class, majority_class])]

        if method == 'oversample':
            # Oversample minority class to match majority class
            n_to_sample = len(majority_samples) - len(minority_samples)

            if n_to_sample <= 0:
                logger.warning("Minority class already matches or exceeds majority class size. No oversampling needed.")
                return data

            # Sample with replacement from minority class
            oversampled = minority_samples.sample(n=n_to_sample, replace=True, random_state=self.random_seed)
            balanced_data = pd.concat([data, oversampled])

            logger.info(f"Oversampled minority class {minority_class} by adding {n_to_sample} samples")

        elif method == 'undersample':
            # Undersample majority class to match minority class
            undersampled = majority_samples.sample(n=len(minority_samples), random_state=self.random_seed)
            balanced_data = pd.concat([minority_samples, undersampled, other_samples])

            logger.info(f"Undersampled majority class {majority_class} to match minority class size of {len(minority_samples)}")

        else:
            logger.error(f"Unknown balancing method: {method}. Valid options are 'oversample' or 'undersample'.")
            return data

        return balanced_data.reset_index(drop=True)

    def add_synthetic_anomalies(self, data: pd.DataFrame, columns: List[str],
                               anomaly_percentage: float = 0.05,
                               anomaly_magnitude: float = 3.0) -> Tuple[pd.DataFrame, List[int]]:
        """
        Add synthetic anomalies to the dataset for anomaly detection training.

        Args:
            data: Input DataFrame
            columns: List of columns to add anomalies to
            anomaly_percentage: Percentage of rows to contain anomalies
            anomaly_magnitude: Magnitude of anomalies as a factor of standard deviation

        Returns:
            Tuple of (DataFrame with anomalies, List of anomaly row indices)
        """
        if data.empty:
            logger.warning("Empty DataFrame provided. Returning original data.")
            return data, []

        # Create a copy to avoid modifying the original data
        anomalous_data = data.copy()

        # Calculate number of anomalies to add
        n_anomalies = max(1, int(len(data) * anomaly_percentage))

        # Randomly select rows to add anomalies
        anomaly_indices = np.random.choice(len(data), size=n_anomalies, replace=False)

        # Add anomalies to selected rows
        for idx in anomaly_indices:
            # Select a random column from the provided list to modify
            if not columns:
                logger.warning("No columns provided for adding anomalies.")
                return data, []

            target_col = np.random.choice(columns)

            if target_col not in anomalous_data.columns or not pd.api.types.is_numeric_dtype(anomalous_data[target_col]):
                logger.warning(f"Column {target_col} not found or not numeric. Skipping.")
                continue

            # Calculate standard deviation of the column
            col_std = anomalous_data[target_col].std()

            # Add anomaly: either very high or very low value
            if np.random.random() > 0.5:
                # High anomaly
                anomalous_data.loc[idx, target_col] += col_std * anomaly_magnitude
            else:
                # Low anomaly
                anomalous_data.loc[idx, target_col] -= col_std * anomaly_magnitude

        logger.info(f"Added {n_anomalies} synthetic anomalies to the dataset")
        return anomalous_data, anomaly_indices.tolist()

    def elastic_scaling(self, data: pd.DataFrame, numeric_columns: Optional[List[str]] = None,
                       stretch_factor_range: Tuple[float, float] = (0.8, 1.2)) -> pd.DataFrame:
        """
        Apply elastic scaling to time series data, randomly stretching or compressing segments.

        Args:
            data: Input DataFrame with time series data
            numeric_columns: List of numeric columns to transform (if None, all numeric columns are used)
            stretch_factor_range: Range for random stretch factors

        Returns:
            DataFrame with elastically scaled data
        """
        if data.empty:
            logger.warning("Empty DataFrame provided. Returning original data.")
            return data

        # If no columns specified, use all numeric columns
        if numeric_columns is None:
            numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

        if not numeric_columns:
            logger.warning("No numeric columns found for elastic scaling.")
            return data

        # Create a copy for transformation
        scaled_data = data.copy()

        # Apply elastic scaling to each segment
        n_segments = random.randint(2, 5)  # Divide data into 2-5 segments
        segment_size = len(data) // n_segments

        for i in range(n_segments):
            # Determine segment bounds
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < n_segments - 1 else len(data)

            # Randomly determine stretch factor for this segment
            stretch_factor = np.random.uniform(stretch_factor_range[0], stretch_factor_range[1])

            # Apply to each numeric column
            for col in numeric_columns:
                if col in scaled_data.columns:
                    scaled_data.loc[start_idx:end_idx, col] *= stretch_factor

        logger.info(f"Applied elastic scaling with {n_segments} segments")
        return scaled_data

    def generate_diverse_weather_conditions(self, sports_data: pd.DataFrame,
                                          weather_column: str = 'weather_condition') -> pd.DataFrame:
        """
        Generate more diverse weather conditions for sports data to improve model robustness.

        Args:
            sports_data: Input DataFrame with sports data
            weather_column: Name of the weather condition column

        Returns:
            DataFrame with more diverse weather conditions
        """
        if sports_data.empty or weather_column not in sports_data.columns:
            logger.warning(f"Empty DataFrame or weather column '{weather_column}' not found. Returning original data.")
            return sports_data

        # Create a copy to avoid modifying the original data
        diverse_data = sports_data.copy()

        # Define a more diverse set of weather conditions and their weights
        # Higher weight = more common condition
        weather_conditions = {
            'Clear': 0.3,
            'Cloudy': 0.25,
            'Partly Cloudy': 0.15,
            'Rainy': 0.1,
            'Light Rain': 0.08,
            'Heavy Rain': 0.05,
            'Windy': 0.04,
            'Foggy': 0.02,
            'Snowy': 0.01
        }

        # Create weighted list for random selection
        conditions = list(weather_conditions.keys())
        weights = list(weather_conditions.values())

        # Generate new weather conditions
        diverse_data[weather_column] = np.random.choice(
            conditions,
            size=len(diverse_data),
            p=weights
        )

        logger.info(f"Generated diverse weather conditions for {len(diverse_data)} sports events")
        return diverse_data


# Example usage
if __name__ == "__main__":
    # Example: Load forex data and apply augmentation
    try:
        forex_data = pd.read_csv("Development/data/historical_forex.csv")

        augmenter = DataAugmenter(random_seed=42)

        # Add noise to price columns
        noisy_data = augmenter.add_gaussian_noise(
            forex_data,
            columns=['open', 'high', 'low', 'close'],
            std_dev=0.005
        )

        # Generate synthetic forex data
        synthetic_forex = augmenter.generate_synthetic_forex_data(
            forex_data,
            num_samples=10,
            volatility_factor=1.2
        )

        # Combine original and synthetic data
        combined_data = pd.concat([forex_data, synthetic_forex]).reset_index(drop=True)

        print(f"Original data size: {len(forex_data)}")
        print(f"Combined data size: {len(combined_data)}")

        # Also demonstrate sports data augmentation if available
        try:
            sports_data = pd.read_csv("Development/data/historical_sports.csv")

            # Generate synthetic sports matches
            synthetic_sports = augmenter.generate_synthetic_sports_data(
                sports_data,
                num_samples=5
            )

            # Add diverse weather conditions
            diverse_weather = augmenter.generate_diverse_weather_conditions(synthetic_sports)

            print(f"Synthetic sports matches generated: {len(synthetic_sports)}")

        except Exception as e:
            print(f"Sports data example error: {e}")

    except Exception as e:
        print(f"Example execution error: {e}")

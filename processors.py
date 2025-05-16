# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Data Processing Module for Super AI Prediction System
Implements data cleaning, transformation, and feature extraction
"""

import os
import re
import json
import logging
import numpy as np
import pandas as pd
import nltk
from typing import Dict, List, Union, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

# NLP processing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    """Base class for all data processors"""

    def __init__(self, output_dir: str = "processed_data"):
        """
        Initialize the base processor.

        Args:
            output_dir: Directory to save processed data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """
        Process the input data

        Args:
            data: Input data
            **kwargs: Additional arguments

        Returns:
            Processed data
        """
        pass

    def save_results(self, data: Any, filename: str, format: str = "csv") -> str:
        """
        Save processed results to file

        Args:
            data: Data to save
            filename: Name of the file (without extension)
            format: File format (csv, json, parquet)

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}"

        if format == "csv":
            filepath = os.path.join(self.output_dir, f"{filename}.csv")
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            else:
                pd.DataFrame(data).to_csv(filepath, index=False)

        elif format == "json":
            filepath = os.path.join(self.output_dir, f"{filename}.json")
            if isinstance(data, pd.DataFrame):
                data.to_json(filepath, orient="records", indent=2)
            else:
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)

        elif format == "parquet":
            filepath = os.path.join(self.output_dir, f"{filename}.parquet")
            if isinstance(data, pd.DataFrame):
                data.to_parquet(filepath, index=False)
            else:
                pd.DataFrame(data).to_parquet(filepath, index=False)

        else:
            logger.error(f"Unsupported format: {format}")
            return ""

        logger.info(f"Results saved to {filepath}")
        return filepath


class DataFrameProcessor(BaseProcessor):
    """Processor for pandas DataFrames"""

    def process(self, data: Union[pd.DataFrame, List[Dict], str],
                operations: List[Dict] = None,
                input_format: str = "dataframe", **kwargs) -> pd.DataFrame:
        """
        Process the input data

        Args:
            data: Input data (DataFrame, list of dicts, or path to file)
            operations: List of operations to perform
            input_format: Format of input data (dataframe, list, json, csv, parquet)
            **kwargs: Additional arguments

        Returns:
            Processed DataFrame
        """
        # Load data into DataFrame if not already
        df = self._load_data(data, input_format)

        # No operations, return as is
        if not operations:
            return df

        # Apply operations in sequence
        for operation in operations:
            op_type = operation.get("type")

            if op_type == "drop_columns":
                columns = operation.get("columns", [])
                df = df.drop(columns=columns, errors='ignore')

            elif op_type == "rename_columns":
                rename_dict = operation.get("mapping", {})
                df = df.rename(columns=rename_dict)

            elif op_type == "filter_rows":
                query = operation.get("query")
                if query:
                    df = df.query(query)

            elif op_type == "select_columns":
                columns = operation.get("columns", [])
                if columns:
                    df = df[columns]

            elif op_type == "sort":
                by = operation.get("by", [])
                ascending = operation.get("ascending", True)
                df = df.sort_values(by=by, ascending=ascending)

            elif op_type == "fillna":
                value = operation.get("value")
                columns = operation.get("columns")
                if columns:
                    for col in columns:
                        if col in df.columns:
                            df[col] = df[col].fillna(value)
                else:
                    df = df.fillna(value)

            elif op_type == "dropna":
                subset = operation.get("subset")
                how = operation.get("how", "any")
                df = df.dropna(subset=subset, how=how)

            elif op_type == "apply_function":
                func_name = operation.get("function")
                columns = operation.get("columns", [])
                new_column = operation.get("new_column")
                func_args = operation.get("args", {})

                if hasattr(self, func_name):
                    func = getattr(self, func_name)

                    if new_column:
                        if len(columns) == 1:
                            # Apply to single column
                            df[new_column] = df[columns[0]].apply(
                                lambda x: func(x, **func_args)
                            )
                        else:
                            # Apply to multiple columns (row-wise)
                            df[new_column] = df.apply(
                                lambda row: func(*[row[col] for col in columns], **func_args),
                                axis=1
                            )
                    else:
                        # In-place modification
                        for col in columns:
                            df[col] = df[col].apply(lambda x: func(x, **func_args))

            elif op_type == "create_column":
                new_column = operation.get("new_column")
                expression = operation.get("expression")
                if new_column and expression:
                    df[new_column] = df.eval(expression)

            elif op_type == "group_by":
                by = operation.get("by", [])
                agg = operation.get("agg", {})
                if by and agg:
                    df = df.groupby(by).agg(agg).reset_index()

            elif op_type == "pivot":
                index = operation.get("index")
                columns = operation.get("columns")
                values = operation.get("values")
                if index and columns and values:
                    df = df.pivot(index=index, columns=columns, values=values).reset_index()

            elif op_type == "one_hot_encode":
                columns = operation.get("columns", [])
                for col in columns:
                    if col in df.columns:
                        dummies = pd.get_dummies(df[col], prefix=col)
                        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

            elif op_type == "normalize":
                columns = operation.get("columns", [])
                method = operation.get("method", "min_max")

                for col in columns:
                    if col in df.columns:
                        if method == "min_max":
                            min_val = df[col].min()
                            max_val = df[col].max()
                            if max_val > min_val:
                                df[col] = (df[col] - min_val) / (max_val - min_val)
                        elif method == "z_score":
                            mean = df[col].mean()
                            std = df[col].std()
                            if std > 0:
                                df[col] = (df[col] - mean) / std

            elif op_type == "custom":
                # Execute custom Python code (be careful!)
                code = operation.get("code")
                if code:
                    # Add df to local variables
                    local_vars = {"df": df}
                    try:
                        exec(code, globals(), local_vars)
                        df = local_vars["df"]
                    except Exception as e:
                        logger.error(f"Error executing custom code: {str(e)}")

        return df

    def _load_data(self, data: Any, input_format: str) -> pd.DataFrame:
        """Load data into DataFrame from various formats"""
        if input_format == "dataframe" and isinstance(data, pd.DataFrame):
            return data

        elif input_format == "list" and isinstance(data, list):
            return pd.DataFrame(data)

        elif input_format in ["json", "csv", "parquet"] and isinstance(data, str):
            try:
                if input_format == "json":
                    return pd.read_json(data)
                elif input_format == "csv":
                    return pd.read_csv(data)
                elif input_format == "parquet":
                    return pd.read_parquet(data)
            except Exception as e:
                logger.error(f"Error loading data from {data}: {str(e)}")
                return pd.DataFrame()

        # If we get here, something went wrong
        if not isinstance(data, pd.DataFrame):
            logger.warning(f"Input is not a DataFrame. Attempting conversion.")
            try:
                return pd.DataFrame(data)
            except:
                logger.error("Failed to convert input to DataFrame")
                return pd.DataFrame()

        return data

    # Helper functions for apply_function
    def extract_numbers(self, text: str) -> float:
        """Extract numbers from text"""
        if not text or not isinstance(text, str):
            return np.nan

        matches = re.findall(r'[-+]?\d*\.\d+|\d+', text)
        if matches:
            return float(matches[0])
        return np.nan

    def clean_text(self, text: str) -> str:
        """Clean text: remove special chars, lowercase, etc."""
        if not text or not isinstance(text, str):
            return ""

        # Remove special characters and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    def extract_date(self, text: str, formats: List[str] = None) -> pd.Timestamp:
        """Extract date from text"""
        if not text or not isinstance(text, str):
            return pd.NaT

        formats = formats or [
            '%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y',
            '%b %d, %Y', '%d %b %Y', '%B %d, %Y', '%d %B %Y'
        ]

        for fmt in formats:
            try:
                return pd.to_datetime(text, format=fmt)
            except:
                continue

        # Try pandas default parser as last resort
        try:
            return pd.to_datetime(text)
        except:
            return pd.NaT


class TextProcessor(BaseProcessor):
    """Processor for text data"""

    def __init__(self, output_dir: str = "processed_data", language: str = "english"):
        """
        Initialize the text processor.

        Args:
            output_dir: Directory to save processed data
            language: Language for stopwords
        """
        super().__init__(output_dir)
        self.stop_words = set(stopwords.words(language))

    def process(self, data: Union[List[str], List[Dict], pd.DataFrame],
               text_column: str = None, operations: List[Dict] = None,
               **kwargs) -> pd.DataFrame:
        """
        Process text data

        Args:
            data: Input text data
            text_column: Column name for text (if DataFrame or dict)
            operations: List of operations to perform
            **kwargs: Additional arguments

        Returns:
            DataFrame with processed text
        """
        # Convert data to DataFrame if needed
        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                df = pd.DataFrame({"text": data})
                text_column = "text"
            else:
                df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Ensure text column exists
        if text_column and text_column not in df.columns:
            logger.error(f"Text column '{text_column}' not found in data")
            return df

        # No operations, return as is
        if not operations:
            return df

        # Apply operations in sequence
        for operation in operations:
            op_type = operation.get("type")

            if op_type == "clean_text":
                new_column = operation.get("new_column", text_column + "_clean")
                df[new_column] = df[text_column].apply(self.clean_text)

            elif op_type == "remove_stopwords":
                input_col = operation.get("input_column", text_column)
                new_column = operation.get("new_column", input_col + "_nostop")
                df[new_column] = df[input_col].apply(self.remove_stopwords)

            elif op_type == "tokenize":
                input_col = operation.get("input_column", text_column)
                new_column = operation.get("new_column", input_col + "_tokens")
                df[new_column] = df[input_col].apply(self.tokenize)

            elif op_type == "sentiment_analysis":
                input_col = operation.get("input_column", text_column)
                polarity_col = operation.get("polarity_column", input_col + "_polarity")
                subjectivity_col = operation.get("subjectivity_column", input_col + "_subjectivity")

                sentiments = df[input_col].apply(self.get_sentiment)
                df[polarity_col] = sentiments.apply(lambda x: x[0])
                df[subjectivity_col] = sentiments.apply(lambda x: x[1])

            elif op_type == "extract_entities":
                input_col = operation.get("input_column", text_column)
                new_column = operation.get("new_column", input_col + "_entities")

                # Note: we'll do simple pattern matching as a placeholder
                # For production, you might want to use spaCy or other NER tools
                df[new_column] = df[input_col].apply(self.extract_simple_entities)

        return df

    def clean_text(self, text: str) -> str:
        """Clean text by removing special characters, extra spaces, etc."""
        if not text or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if not text or not isinstance(text, str):
            return []

        return word_tokenize(text)

    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text"""
        if not text or not isinstance(text, str):
            return ""

        tokens = self.tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        return ' '.join(filtered_tokens)

    def get_sentiment(self, text: str) -> tuple:
        """
        Get sentiment analysis scores using TextBlob

        Returns:
            Tuple of (polarity, subjectivity)
        """
        if not text or not isinstance(text, str):
            return (0.0, 0.0)

        analysis = TextBlob(text)
        return (analysis.sentiment.polarity, analysis.sentiment.subjectivity)

    def extract_simple_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract simple entities based on patterns

        Returns:
            Dictionary of entity types and values
        """
        if not text or not isinstance(text, str):
            return {}

        entities = {
            "dates": [],
            "amounts": [],
            "percentages": [],
            "emails": [],
            "urls": []
        }

        # Extract dates (simple patterns)
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # 01/01/2020, 01-01-2020
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}'      # 2020/01/01, 2020-01-01
        ]
        for pattern in date_patterns:
            entities["dates"].extend(re.findall(pattern, text))

        # Extract monetary amounts
        entities["amounts"] = re.findall(r'[$€£¥]\s?\d+(?:\.\d+)?|\d+(?:\.\d+)?\s?[$€£¥]', text)

        # Extract percentages
        entities["percentages"] = re.findall(r'\d+(?:\.\d+)?%', text)

        # Extract emails
        entities["emails"] = re.findall(r'\S+@\S+\.\S+', text)

        # Extract URLs
        entities["urls"] = re.findall(r'https?://\S+|www\.\S+\.\S+', text)

        # Remove empty lists
        return {k: v for k, v in entities.items() if v}


class ForexDataProcessor(DataFrameProcessor):
    """Specialized processor for forex data"""

    def process(self, data: Any, input_format: str = "dataframe",
               operations: List[Dict] = None, calculate_indicators: bool = True,
               **kwargs) -> pd.DataFrame:
        """
        Process forex data with specialized operations

        Args:
            data: Input data
            input_format: Format of input data
            operations: List of operations to perform
            calculate_indicators: Whether to calculate technical indicators
            **kwargs: Additional arguments

        Returns:
            Processed DataFrame
        """
        # First apply standard operations
        df = super().process(data, operations, input_format, **kwargs)

        # Calculate technical indicators if requested
        if calculate_indicators:
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close']
            lowercase_columns = {col.lower(): col for col in df.columns}

            # Map column names if they exist in different case
            column_mapping = {}
            for req_col in required_columns:
                if req_col in lowercase_columns:
                    column_mapping[lowercase_columns[req_col]] = req_col

            if column_mapping:
                df = df.rename(columns=column_mapping)

            # Check if we have the necessary columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing columns for indicator calculation: {missing_columns}")
                return df

            # Calculate indicators
            df = self._calculate_indicators(df)

        return df

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for forex data"""
        try:
            # Moving Averages
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()

            # Exponential Moving Averages
            df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
            df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

            # MACD
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

            # RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)

            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()

            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()

            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=14).mean()

            # Volatility
            df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(20)

            # Price change
            df['daily_change'] = df['close'].pct_change()
            df['daily_range'] = (df['high'] - df['low']) / df['close']

            # Volume indicators (if volume column exists)
            if 'volume' in df.columns:
                df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
                df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")

        return df


class SportsDataProcessor(DataFrameProcessor):
    """Specialized processor for sports betting data"""

    def process(self, data: Any, input_format: str = "dataframe",
               operations: List[Dict] = None, sport_type: str = None,
               **kwargs) -> pd.DataFrame:
        """
        Process sports data with specialized operations

        Args:
            data: Input data
            input_format: Format of input data
            operations: List of operations to perform
            sport_type: Type of sport (football, basketball, etc.)
            **kwargs: Additional arguments

        Returns:
            Processed DataFrame
        """
        # First apply standard operations
        df = super().process(data, operations, input_format, **kwargs)

        # Apply sport-specific processing
        if sport_type:
            method_name = f"_process_{sport_type.lower()}"
            if hasattr(self, method_name):
                process_method = getattr(self, method_name)
                df = process_method(df)
            else:
                logger.warning(f"No specialized processing for sport type: {sport_type}")

        return df

    def _process_football(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process football (soccer) data"""
        try:
            # Check if we have essential columns
            required_columns = ['home_team', 'away_team', 'home_score', 'away_score', 'date']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing columns for football processing: {missing_columns}")
                return df

            # Calculate match outcomes
            df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
            df['away_win'] = (df['home_score'] < df['away_score']).astype(int)
            df['draw'] = (df['home_score'] == df['away_score']).astype(int)

            # Total goals
            df['total_goals'] = df['home_score'] + df['away_score']
            df['goal_difference'] = df['home_score'] - df['away_score']

            # Over/Under indicators
            df['over_2_5'] = (df['total_goals'] > 2.5).astype(int)
            df['under_2_5'] = (df['total_goals'] < 2.5).astype(int)

            # BTTS (Both Teams To Score)
            df['btts'] = ((df['home_score'] > 0) & (df['away_score'] > 0)).astype(int)

            # Create team-specific features if date column exists
            if pd.api.types.is_datetime64_dtype(df['date']) or pd.api.types.is_datetime64_any_dtype(df['date']):
                # Sort by date
                df = df.sort_values('date')

                # Create team-based aggregated stats
                home_team_stats = {}
                away_team_stats = {}

                # Process data sequentially to create team form
                for idx, row in df.iterrows():
                    home_team = row['home_team']
                    away_team = row['away_team']

                    # Initialize team stats if not exists
                    for team in [home_team, away_team]:
                        if team not in home_team_stats:
                            home_team_stats[team] = {
                                'last_5_home_wins': [],
                                'last_5_home_goals_scored': [],
                                'last_5_home_goals_conceded': [],
                            }
                        if team not in away_team_stats:
                            away_team_stats[team] = {
                                'last_5_away_wins': [],
                                'last_5_away_goals_scored': [],
                                'last_5_away_goals_conceded': [],
                            }

                    # Update current match with previous form
                    df.at[idx, 'home_team_last_5_home_wins'] = sum(home_team_stats[home_team]['last_5_home_wins'][-5:]) if home_team_stats[home_team]['last_5_home_wins'] else None
                    df.at[idx, 'home_team_last_5_home_goals_scored'] = sum(home_team_stats[home_team]['last_5_home_goals_scored'][-5:]) if home_team_stats[home_team]['last_5_home_goals_scored'] else None
                    df.at[idx, 'home_team_last_5_home_goals_conceded'] = sum(home_team_stats[home_team]['last_5_home_goals_conceded'][-5:]) if home_team_stats[home_team]['last_5_home_goals_conceded'] else None

                    df.at[idx, 'away_team_last_5_away_wins'] = sum(away_team_stats[away_team]['last_5_away_wins'][-5:]) if away_team_stats[away_team]['last_5_away_wins'] else None
                    df.at[idx, 'away_team_last_5_away_goals_scored'] = sum(away_team_stats[away_team]['last_5_away_goals_scored'][-5:]) if away_team_stats[away_team]['last_5_away_goals_scored'] else None
                    df.at[idx, 'away_team_last_5_away_goals_conceded'] = sum(away_team_stats[away_team]['last_5_away_goals_conceded'][-5:]) if away_team_stats[away_team]['last_5_away_goals_conceded'] else None

                    # Update team stats with current match
                    home_team_stats[home_team]['last_5_home_wins'].append(1 if row['home_score'] > row['away_score'] else 0)
                    home_team_stats[home_team]['last_5_home_goals_scored'].append(row['home_score'])
                    home_team_stats[home_team]['last_5_home_goals_conceded'].append(row['away_score'])

                    away_team_stats[away_team]['last_5_away_wins'].append(1 if row['away_score'] > row['home_score'] else 0)
                    away_team_stats[away_team]['last_5_away_goals_scored'].append(row['away_score'])
                    away_team_stats[away_team]['last_5_away_goals_conceded'].append(row['home_score'])

        except Exception as e:
            logger.error(f"Error processing football data: {str(e)}")

        return df

    def _process_basketball(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process basketball data"""
        try:
            # Check if we have essential columns
            required_columns = ['home_team', 'away_team', 'home_score', 'away_score']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing columns for basketball processing: {missing_columns}")
                return df

            # Calculate match outcomes
            df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
            df['away_win'] = (df['home_score'] < df['away_score']).astype(int)

            # Total points
            df['total_points'] = df['home_score'] + df['away_score']
            df['point_difference'] = df['home_score'] - df['away_score']

            # Over/Under indicators (adjust threshold as needed)
            df['over_200'] = (df['total_points'] > 200).astype(int)
            df['under_200'] = (df['total_points'] < 200).astype(int)

            # Additional stats if available
            for stat in ['rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'fouls']:
                home_stat = f'home_{stat}'
                away_stat = f'away_{stat}'
                if home_stat in df.columns and away_stat in df.columns:
                    df[f'total_{stat}'] = df[home_stat] + df[away_stat]
                    df[f'{stat}_difference'] = df[home_stat] - df[away_stat]

        except Exception as e:
            logger.error(f"Error processing basketball data: {str(e)}")

        return df


class ProcessorFactory:
    """Factory for creating data processors"""

    @staticmethod
    def create_processor(processor_type: str, **kwargs) -> BaseProcessor:
        """
        Create a processor of the specified type

        Args:
            processor_type: Type of processor (dataframe, text, forex, sports)
            **kwargs: Additional arguments for the processor

        Returns:
            Initialized processor
        """
        if processor_type.lower() == "dataframe":
            return DataFrameProcessor(**kwargs)
        elif processor_type.lower() == "text":
            return TextProcessor(**kwargs)
        elif processor_type.lower() == "forex":
            return ForexDataProcessor(**kwargs)
        elif processor_type.lower() == "sports":
            return SportsDataProcessor(**kwargs)
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")

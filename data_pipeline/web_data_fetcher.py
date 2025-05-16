# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import os
import re
import json
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Union, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants and configuration
DATA_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data'))
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
SCRAPED_DATA_DIR = DATA_DIR / 'scraped'

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SCRAPED_DATA_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# API Keys (should be stored in environment variables or config files in production)
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"  # Replace with actual key
NEWS_API_KEY = "YOUR_NEWS_API_KEY"  # Replace with actual key

class WebDataFetcher:
    """
    Handles fetching and preprocessing of data from web sources for the Super AI Prediction System.
    Implements comprehensive data quality checks and preprocessing steps.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the WebDataFetcher with optional configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scalers = {}
        self.sentiment_analyzer = None
        self.selenium_driver = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize necessary components like NLP tools."""
        try:
            import nltk
            nltk.download('vader_lexicon', quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("NLTK not available. Sentiment analysis will be limited.")

    def fetch_forex_data(self, symbol: str, interval: str = 'daily',
                         output_size: str = 'full') -> pd.DataFrame:
        """
        Fetch forex data from Alpha Vantage API.

        Args:
            symbol: Currency pair (e.g., "EUR/USD")
            interval: Time interval ('daily', 'weekly', 'monthly', 'intraday')
            output_size: 'compact' for 100 data points, 'full' for all available

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Format currency pair for API
            from_currency, to_currency = symbol.split('/')

            # Build API URL
            base_url = "https://www.alphavantage.co/query"
            function = "FX_DAILY"
            if interval == 'weekly':
                function = "FX_WEEKLY"
            elif interval == 'monthly':
                function = "FX_MONTHLY"

            params = {
                "function": function,
                "from_symbol": from_currency,
                "to_symbol": to_currency,
                "outputsize": output_size,
                "apikey": ALPHA_VANTAGE_API_KEY
            }

            # Make API request
            logger.info(f"Fetching {interval} forex data for {symbol}")
            response = requests.get(base_url, params=params)

            if response.status_code != 200:
                logger.error(f"API request failed with status {response.status_code}")
                return pd.DataFrame()

            data = response.json()

            # Check for error messages
            if "Error Message" in data:
                logger.error(f"API error: {data['Error Message']}")
                return pd.DataFrame()

            # Parse the response
            time_series_key = f"Time Series FX ({interval.capitalize()})"
            if time_series_key not in data:
                logger.error(f"Expected key {time_series_key} not found in response")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data[time_series_key]).T
            df.columns = [col.split('. ')[1] for col in df.columns]
            df.columns = ['open', 'high', 'low', 'close']

            # Convert string values to float
            for col in df.columns:
                df[col] = df[col].astype(float)

            # Reset index and rename to date
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'date'}, inplace=True)

            # Add symbol and interval info
            df['symbol'] = symbol
            df['interval'] = interval

            # Add timestamp for tracking
            df['timestamp'] = datetime.now().isoformat()

            # Save raw data
            self._save_raw_data(df, f"forex_{symbol.replace('/', '')}_raw.csv")

            return df

        except Exception as e:
            logger.error(f"Error fetching forex data: {e}")
            return pd.DataFrame()

    def fetch_sports_data(self, sport: str, league: str,
                         days_back: int = 30) -> pd.DataFrame:
        """
        Fetch sports data from APIs or web scraping.

        Args:
            sport: Sport type (e.g., "soccer", "basketball")
            league: League name (e.g., "premier_league", "nba")
            days_back: How many days of data to fetch

        Returns:
            DataFrame with match data
        """
        try:
            logger.info(f"Fetching {sport} data for {league}")

            # This is a placeholder - in a real implementation, you would:
            # 1. Connect to a sports data API (e.g., SportRadar, BetFair, etc.)
            # 2. Fetch historical match data
            # 3. Process and structure the data

            # For this example, we'll create mock data
            mock_data = self._generate_mock_sports_data(sport, league, days_back)

            # Save raw data
            self._save_raw_data(mock_data, f"sports_{sport}_{league}_raw.csv")

            return mock_data

        except Exception as e:
            logger.error(f"Error fetching sports data: {e}")
            return pd.DataFrame()

    def _generate_mock_sports_data(self, sport: str, league: str, days_back: int) -> pd.DataFrame:
        """Generate mock sports data for demonstration purposes."""
        np.random.seed(42)  # For reproducibility

        dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(days_back)]

        if sport.lower() == "soccer":
            teams = ["Manchester United", "Liverpool", "Chelsea", "Arsenal",
                    "Manchester City", "Tottenham", "Leicester", "Everton"]

            matches = []
            for i in range(len(dates)):
                # Randomly select home and away teams
                home_idx = np.random.randint(0, len(teams))
                away_idx = np.random.randint(0, len(teams))
                while away_idx == home_idx:
                    away_idx = np.random.randint(0, len(teams))

                home_team = teams[home_idx]
                away_team = teams[away_idx]

                # Generate scores
                home_score = np.random.randint(0, 5)
                away_score = np.random.randint(0, 4)

                # Other stats
                home_possession = np.random.randint(30, 71)
                away_possession = 100 - home_possession

                home_shots = np.random.randint(5, 25)
                away_shots = np.random.randint(5, 20)

                home_shots_on_target = np.random.randint(1, home_shots + 1)
                away_shots_on_target = np.random.randint(1, away_shots + 1)

                # Weather and other factors
                temperature = np.random.randint(5, 30)
                is_rainy = np.random.random() < 0.3

                matches.append({
                    'date': dates[i],
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'home_possession': home_possession,
                    'away_possession': away_possession,
                    'home_shots': home_shots,
                    'away_shots': away_shots,
                    'home_shots_on_target': home_shots_on_target,
                    'away_shots_on_target': away_shots_on_target,
                    'temperature': temperature,
                    'is_rainy': is_rainy,
                    'league': league,
                    'sport': sport
                })

            return pd.DataFrame(matches)

        elif sport.lower() == "basketball":
            teams = ["Lakers", "Celtics", "Bulls", "Warriors", "Nets",
                    "Clippers", "Heat", "Mavericks"]

            matches = []
            for i in range(len(dates)):
                home_idx = np.random.randint(0, len(teams))
                away_idx = np.random.randint(0, len(teams))
                while away_idx == home_idx:
                    away_idx = np.random.randint(0, len(teams))

                home_team = teams[home_idx]
                away_team = teams[away_idx]

                # Generate scores
                home_score = np.random.randint(70, 120)
                away_score = np.random.randint(70, 115)

                matches.append({
                    'date': dates[i],
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'league': league,
                    'sport': sport
                })

            return pd.DataFrame(matches)

        else:
            # Return empty DataFrame for unsupported sports
            return pd.DataFrame()

    def scrape_daily_news(self,
                          keywords: List[str] = ["forex", "currency", "market", "stock", "trade", "finance", "betting", "sports"],
                          max_articles: int = 50,
                          store_format: str = "csv") -> pd.DataFrame:
        """
        Scrape daily financial and sports news from various sources.

        Args:
            keywords: List of keywords to filter news
            max_articles: Maximum number of articles to scrape
            store_format: Format to store data ('csv' or 'ai_notation')

        Returns:
            DataFrame containing scraped news data
        """
        try:
            logger.info(f"Scraping daily news for keywords: {keywords}")

            # Define news sources to scrape
            sources = [
                {
                    "name": "Yahoo Finance",
                    "url": "https://finance.yahoo.com/news/",
                    "article_selector": "div.Ov\\(h\\).Pend\\(44px\\).Pstart\\(25px\\)",
                    "title_selector": "h3",
                    "content_selector": "p",
                    "date_selector": "span.C\\(\\#959595\\)"
                },
                {
                    "name": "CNBC",
                    "url": "https://www.cnbc.com/world-markets/",
                    "article_selector": "div.Card-titleContainer",
                    "title_selector": "a",
                    "content_selector": "div.Card-standardBreakerContent",
                    "date_selector": "span.Card-time"
                },
                {
                    "name": "ESPN",
                    "url": "https://www.espn.com/",
                    "article_selector": "div.contentItem__content",
                    "title_selector": "h1",
                    "content_selector": "p",
                    "date_selector": "span.contentMeta__timestamp"
                }
            ]

            all_articles = []

            # Initialize Selenium if available
            if self.selenium_driver is None:
                self._initialize_selenium()

            # Scrape each source
            for source in sources:
                try:
                    logger.info(f"Scraping news from {source['name']}")

                    # Choose scraping method based on availability
                    if self.selenium_driver is not None:
                        articles = self._scrape_with_selenium(source)
                    else:
                        articles = self._scrape_with_requests(source)

                    # Add source information
                    for article in articles:
                        article['source'] = source['name']

                    all_articles.extend(articles)

                    # Limit the number of articles
                    if len(all_articles) >= max_articles:
                        all_articles = all_articles[:max_articles]
                        break

                except Exception as e:
                    logger.error(f"Error scraping {source['name']}: {str(e)}")
                    continue

            # Filter articles by keywords
            if keywords:
                filtered_articles = []
                for article in all_articles:
                    text = (article.get('title', '') + ' ' + article.get('content', '')).lower()
                    if any(keyword.lower() in text for keyword in keywords):
                        filtered_articles.append(article)
                all_articles = filtered_articles

            # Convert to DataFrame
            df = pd.DataFrame(all_articles)

            # Add timestamp and additional metadata
            df['scrape_timestamp'] = datetime.now().isoformat()
            df['keywords_matched'] = df.apply(
                lambda row: [kw for kw in keywords if kw.lower() in (row.get('title', '') + ' ' + row.get('content', '')).lower()],
                axis=1
            )

            # Apply sentiment analysis if available
            if self.sentiment_analyzer:
                df['sentiment'] = df.apply(
                    lambda row: self._analyze_sentiment(row.get('title', '') + ' ' + row.get('content', '')),
                    axis=1
                )

            # Store in specified format
            if store_format.lower() == 'csv':
                self._store_as_csv(df, "scraped_data.csv")
            elif store_format.lower() == 'ai_notation':
                self._store_as_ai_notation(df)

            return df

        except Exception as e:
            logger.error(f"Error in daily news scraping: {str(e)}")
            return pd.DataFrame()

    def _scrape_with_selenium(self, source_config):
        """Scrape news using Selenium for JavaScript-heavy sites."""
        if self.selenium_driver is None:
            raise ValueError("Selenium driver not initialized")

        articles = []
        try:
            # Navigate to the source URL
            self.selenium_driver.get(source_config['url'])

            # Wait for articles to load
            WebDriverWait(self.selenium_driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, source_config['article_selector']))
            )

            # Find all article elements
            article_elements = self.selenium_driver.find_elements(By.CSS_SELECTOR, source_config['article_selector'])

            for article_elem in article_elements[:10]:  # Limit to first 10 articles per source
                try:
                    # Extract article data
                    title_elem = article_elem.find_element(By.CSS_SELECTOR, source_config['title_selector'])
                    title = title_elem.text

                    # Try to get content
                    try:
                        content_elem = article_elem.find_element(By.CSS_SELECTOR, source_config['content_selector'])
                        content = content_elem.text
                    except:
                        content = ""

                    # Try to get date
                    try:
                        date_elem = article_elem.find_element(By.CSS_SELECTOR, source_config['date_selector'])
                        date = date_elem.text
                    except:
                        date = datetime.now().strftime("%Y-%m-%d")

                    # Try to get URL
                    try:
                        link_elem = title_elem if title_elem.tag_name == 'a' else title_elem.find_element(By.TAG_NAME, 'a')
                        url = link_elem.get_attribute('href')
                    except:
                        url = source_config['url']

                    articles.append({
                        'title': title,
                        'content': content,
                        'date': date,
                        'url': url
                    })

                except Exception as e:
                    logger.warning(f"Error extracting article data: {str(e)}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"Selenium scraping error: {str(e)}")
            return []

    def _scrape_with_requests(self, source_config):
        """Scrape news using requests and BeautifulSoup for simpler sites."""
        articles = []
        try:
            # Send HTTP request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(source_config['url'], headers=headers, timeout=10)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all article elements
            article_elements = soup.select(source_config['article_selector'])

            for article_elem in article_elements[:10]:  # Limit to first 10 articles per source
                try:
                    # Extract article data
                    title_elem = article_elem.select_one(source_config['title_selector'])
                    title = title_elem.text.strip() if title_elem else "No title"

                    # Try to get content
                    content_elem = article_elem.select_one(source_config['content_selector'])
                    content = content_elem.text.strip() if content_elem else ""

                    # Try to get date
                    date_elem = article_elem.select_one(source_config['date_selector'])
                    date = date_elem.text.strip() if date_elem else datetime.now().strftime("%Y-%m-%d")

                    # Try to get URL
                    url = ""
                    if title_elem and title_elem.name == 'a' and title_elem.has_attr('href'):
                        url = title_elem['href']
                    elif title_elem:
                        link_elem = title_elem.find('a')
                        if link_elem and link_elem.has_attr('href'):
                            url = link_elem['href']

                    # Make URL absolute if it's relative
                    if url and not url.startswith('http'):
                        url = f"{response.url.rstrip('/')}/{url.lstrip('/')}"

                    articles.append({
                        'title': title,
                        'content': content,
                        'date': date,
                        'url': url
                    })

                except Exception as e:
                    logger.warning(f"Error extracting article data: {str(e)}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"Requests scraping error: {str(e)}")
            return []

    def _analyze_sentiment(self, text):
        """Analyze sentiment of text using VADER."""
        if self.sentiment_analyzer:
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            return {
                'compound': sentiment['compound'],
                'positive': sentiment['pos'],
                'negative': sentiment['neg'],
                'neutral': sentiment['neu'],
                'label': 'positive' if sentiment['compound'] >= 0.05 else 'negative' if sentiment['compound'] <= -0.05 else 'neutral'
            }
        return None

    def _store_as_csv(self, df, filename):
        """Store scraped data as CSV."""
        filepath = SCRAPED_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved scraped data to {filepath}")
        return filepath

    def _store_as_ai_notation(self, df):
        """Store scraped data in AI-readable format with compressed keys."""
        # Create AI notation directory if it doesn't exist
        ai_notation_dir = DATA_DIR / 'ai_notation'
        ai_notation_dir.mkdir(exist_ok=True)

        # Define compressed keys mapping
        key_mapping = {
            'title': 't',
            'content': 'c',
            'date': 'd',
            'url': 'u',
            'source': 's',
            'scrape_timestamp': 'ts',
            'keywords_matched': 'kw',
            'sentiment': 'sent'
        }

        # Convert to dictionary with compressed keys
        data_dict = df.to_dict(orient='records')
        compressed_data = []

        for item in data_dict:
            compressed_item = {}
            for key, value in item.items():
                compressed_key = key_mapping.get(key, key)
                compressed_item[compressed_key] = value
            compressed_data.append(compressed_item)

        # Add metadata
        ai_notation = {
            'meta': {
                'format_version': '1.0',
                'type': 'news_data',
                'timestamp': datetime.now().isoformat(),
                'item_count': len(compressed_data),
                'key_mapping': key_mapping
            },
            'data': compressed_data
        }

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = ai_notation_dir / f"news_{timestamp}.json"

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(ai_notation, f, indent=2)

        logger.info(f"Saved AI notation data to {filepath}")
        return filepath

    def run_daily_scraping(self):
        """Run daily scraping job and store in both formats."""
        logger.info("Starting daily scraping job")

        # Scrape news
        news_df = self.scrape_daily_news(store_format='csv')

        # Also store in AI notation format
        if not news_df.empty:
            self._store_as_ai_notation(news_df)

        # Scrape forex data for major pairs
        currency_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
        for pair in currency_pairs:
            try:
                forex_df = self.fetch_forex_data(pair)
                if not forex_df.empty:
                    # Store in CSV
                    filename = f"forex_{pair.replace('/', '')}.csv"
                    self._store_as_csv(forex_df, filename)

                    # Store in AI notation
                    self._store_as_ai_notation(forex_df)
            except Exception as e:
                logger.error(f"Error scraping forex data for {pair}: {str(e)}")

        # Scrape sports data
        sports = [("soccer", "premier_league"), ("basketball", "nba")]
        for sport, league in sports:
            try:
                sports_df = self.fetch_sports_data(sport, league)
                if not sports_df.empty:
                    # Store in CSV
                    filename = f"sports_{sport}_{league}.csv"
                    self._store_as_csv(sports_df, filename)

                    # Store in AI notation
                    self._store_as_ai_notation(sports_df)
            except Exception as e:
                logger.error(f"Error scraping sports data for {sport} {league}: {str(e)}")

        logger.info("Daily scraping job completed")
        return True

    def validate_data(self, data: pd.DataFrame, validation_rules: Dict[str, Dict]) -> Tuple[bool, pd.DataFrame, Dict]:
        """
        Validate data using specified rules.

        Args:
            data: DataFrame to validate
            validation_rules: Dictionary of validation rules

        Returns:
            Tuple of (is_valid, filtered_data, validation_results)
        """
        if data.empty:
            logger.warning("Empty DataFrame provided for validation")
            return False, data, {"error": "Empty DataFrame"}

        logger.info(f"Validating DataFrame with {len(data)} rows")

        # Track validation results
        results = {
            "total_rows": len(data),
            "valid_rows": 0,
            "filtered_rows": 0,
            "column_results": {}
        }

        # Create a mask for valid rows (initially all True)
        valid_mask = pd.Series(True, index=data.index)

        # Apply validation rules for each column
        for column, rules in validation_rules.items():
            if column not in data.columns:
                logger.warning(f"Column '{column}' not found in DataFrame")
                results["column_results"][column] = {"error": "Column not found"}
                continue

            column_result = {
                "invalid_values": 0,
                "rules_applied": []
            }

            # Check for missing values
            if rules.get("allow_missing", False) is False:
                missing_mask = data[column].isna()
                valid_mask = valid_mask & ~missing_mask
                missing_count = missing_mask.sum()
                column_result["missing_values"] = int(missing_count)
                column_result["rules_applied"].append("no_missing")

            # Check for minimum value
            if "min_value" in rules:
                min_mask = data[column] >= rules["min_value"]
                valid_mask = valid_mask & min_mask
                column_result["min_value_failures"] = int((~min_mask).sum())
                column_result["rules_applied"].append(f"min_value={rules['min_value']}")

            # Check for maximum value
            if "max_value" in rules:
                max_mask = data[column] <= rules["max_value"]
                valid_mask = valid_mask & max_mask
                column_result["max_value_failures"] = int((~max_mask).sum())
                column_result["rules_applied"].append(f"max_value={rules['max_value']}")

            # Check for regex pattern
            if "pattern" in rules:
                pattern = rules["pattern"]
                # Convert column to string if not already
                str_col = data[column].astype(str)
                pattern_mask = str_col.str.match(pattern)
                valid_mask = valid_mask & pattern_mask
                column_result["pattern_failures"] = int((~pattern_mask).sum())
                column_result["rules_applied"].append(f"pattern={pattern}")

            # Check for allowed values
            if "allowed_values" in rules:
                allowed = rules["allowed_values"]
                allowed_mask = data[column].isin(allowed)
                valid_mask = valid_mask & allowed_mask
                column_result["allowed_values_failures"] = int((~allowed_mask).sum())
                column_result["rules_applied"].append(f"allowed_values={allowed}")

            # Check outliers using Z-score
            if "max_zscore" in rules:
                max_zscore = rules["max_zscore"]
                zscore = np.abs((data[column] - data[column].mean()) / data[column].std())
                outlier_mask = zscore <= max_zscore
                valid_mask = valid_mask & outlier_mask
                column_result["outlier_failures"] = int((~outlier_mask).sum())
                column_result["rules_applied"].append(f"max_zscore={max_zscore}")

            # Custom validation function
            if "custom_func" in rules and callable(rules["custom_func"]):
                func = rules["custom_func"]
                custom_mask = data[column].apply(func)
                valid_mask = valid_mask & custom_mask
                column_result["custom_failures"] = int((~custom_mask).sum())
                column_result["rules_applied"].append("custom_function")

            # Update column results
            column_result["invalid_values"] = int((~valid_mask).sum())
            results["column_results"][column] = column_result

        # Filter the DataFrame
        filtered_data = data[valid_mask].copy()

        # Update results
        results["valid_rows"] = len(filtered_data)
        results["filtered_rows"] = len(data) - len(filtered_data)
        results["is_valid"] = results["filtered_rows"] == 0

        logger.info(f"Validation complete: {results['valid_rows']} valid rows, "
                  f"{results['filtered_rows']} filtered rows")

        return results["is_valid"], filtered_data, results

    def preprocess_data(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Preprocess data based on configuration.

        Args:
            data: DataFrame to preprocess
            config: Preprocessing configuration

        Returns:
            Preprocessed DataFrame
        """
        if data.empty:
            logger.warning("Empty DataFrame provided for preprocessing")
            return data

        logger.info(f"Preprocessing DataFrame with {len(data)} rows")

        processed_data = data.copy()

        # Handle missing values
        if config.get("handle_missing", False):
            missing_strategy = config.get("missing_strategy", "interpolate")

            if missing_strategy == "drop":
                processed_data.dropna(inplace=True)
                logger.info(f"Dropped rows with missing values. {len(processed_data)} rows remaining.")

            elif missing_strategy == "interpolate":
                processed_data = processed_data.interpolate(method='linear', limit_direction='both')
                logger.info("Interpolated missing values")

            elif missing_strategy == "forward_fill":
                processed_data = processed_data.ffill().bfill()
                logger.info("Forward-filled missing values")

            elif missing_strategy == "mean":
                for col in processed_data.select_dtypes(include=[np.number]).columns:
                    processed_data[col].fillna(processed_data[col].mean(), inplace=True)
                logger.info("Filled missing values with column means")

        # Normalize/standardize data
        if config.get("normalize", False):
            scaling_method = config.get("scaling_method", "minmax")
            columns_to_scale = config.get("columns_to_scale", [])

            # If no specific columns provided, use all numeric columns
            if not columns_to_scale:
                columns_to_scale = processed_data.select_dtypes(include=[np.number]).columns.tolist()

            # Only attempt scaling if we have numeric columns
            if columns_to_scale:
                # Create a unique identifier for this scaling operation
                scaling_id = f"{','.join(columns_to_scale)}_{scaling_method}"

                if scaling_method == "minmax":
                    scaler = MinMaxScaler()

                elif scaling_method == "standard":
                    scaler = StandardScaler()

                else:
                    logger.warning(f"Unknown scaling method: {scaling_method}. Using MinMaxScaler.")
                    scaler = MinMaxScaler()

                # Fit and transform the data
                processed_data[columns_to_scale] = scaler.fit_transform(processed_data[columns_to_scale])

                # Store the scaler for later use
                self.scalers[scaling_id] = scaler

                logger.info(f"Applied {scaling_method} scaling to {len(columns_to_scale)} columns")

        # Handle outliers
        if config.get("remove_outliers", False):
            outlier_method = config.get("outlier_method", "zscore")
            threshold = config.get("outlier_threshold", 3.0)

            if outlier_method == "zscore":
                # Calculate z-scores for numeric columns
                numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()

                for col in numeric_cols:
                    if processed_data[col].std() == 0:
                        continue  # Skip constant columns

                    zscore = np.abs((processed_data[col] - processed_data[col].mean()) / processed_data[col].std())
                    processed_data = processed_data[zscore < threshold]

                logger.info(f"Removed outliers using Z-score method. {len(processed_data)} rows remaining.")

            elif outlier_method == "iqr":
                # Use interquartile range to identify outliers
                numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()

                for col in numeric_cols:
                    Q1 = processed_data[col].quantile(0.25)
                    Q3 = processed_data[col].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR

                    processed_data = processed_data[(processed_data[col] >= lower_bound) &
                                               (processed_data[col] <= upper_bound)]

                logger.info(f"Removed outliers using IQR method. {len(processed_data)} rows remaining.")

        # Feature engineering
        if config.get("feature_engineering", False):
            domain = config.get("domain", "")

            if domain == "forex":
                processed_data = self._add_forex_features(processed_data)

            elif domain == "sports":
                processed_data = self._add_sports_features(processed_data)

            logger.info(f"Added engineered features for domain: {domain}")

        # Time-series alignment
        if config.get("align_time_series", False) and 'date' in processed_data.columns:
            freq = config.get("time_freq", "D")  # Daily by default

            # Convert date to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(processed_data['date']):
                processed_data['date'] = pd.to_datetime(processed_data['date'])

            # Set date as index and resample
            processed_data.set_index('date', inplace=True)
            processed_data = processed_data.resample(freq).mean().reset_index()

            logger.info(f"Aligned time series with frequency {freq}")

        # Add a timestamp for tracking
        processed_data['processing_timestamp'] = datetime.now().isoformat()

        return processed_data

    def _add_forex_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add forex-specific features."""
        df = data.copy()

        # Ensure we have the necessary columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.warning("Cannot add forex features. Missing required columns.")
            return df

        # Calculate price indicators only if we have numeric data
        if df[required_cols].dtypes.all() == np.dtype('float64'):
            # Add Moving Averages
            windows = [5, 10, 20, 50, 200]
            for window in windows:
                df[f'ma_{window}'] = df['close'].rolling(window=window).mean()

            # Add exponential moving averages
            for window in [12, 26]:
                df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()

            # Calculate MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # Calculate RSI (14-period)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # Bollinger Bands (20-period, 2 standard deviations)
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

            # Volatility
            df['volatility'] = df['bb_std'] / df['bb_middle']

            # Price momentum
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

            logger.info("Added forex technical indicators")

        return df

    def _add_sports_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add sports-specific features."""
        df = data.copy()

        # Check if we have the necessary columns
        if not all(col in df.columns for col in ['home_team', 'away_team', 'home_score', 'away_score']):
            logger.warning("Cannot add sports features. Missing required columns.")
            return df

        # Add basic features
        df['total_score'] = df['home_score'] + df['away_score']
        df['score_difference'] = df['home_score'] - df['away_score']

        # Add result column
        df['result'] = np.where(df['home_score'] > df['away_score'], 'home_win',
                              np.where(df['home_score'] < df['away_score'], 'away_win', 'draw'))

        # Add features specific to the sport type
        if 'sport' in df.columns:
            if 'soccer' in df['sport'].values:
                soccer_df = df[df['sport'] == 'soccer'].copy()

                # Check if we have the additional soccer stats
                if all(col in soccer_df.columns for col in ['home_possession', 'home_shots', 'home_shots_on_target']):
                    # Add shot conversion rate
                    soccer_df['home_conversion'] = np.where(soccer_df['home_shots'] > 0,
                                                      soccer_df['home_score'] / soccer_df['home_shots'], 0)
                    soccer_df['away_conversion'] = np.where(soccer_df['away_shots'] > 0,
                                                      soccer_df['away_score'] / soccer_df['away_shots'], 0)

                    # Add shot accuracy
                    soccer_df['home_accuracy'] = np.where(soccer_df['home_shots'] > 0,
                                                    soccer_df['home_shots_on_target'] / soccer_df['home_shots'], 0)
                    soccer_df['away_accuracy'] = np.where(soccer_df['away_shots'] > 0,
                                                    soccer_df['away_shots_on_target'] / soccer_df['away_shots'], 0)

                    # Merge back with the main DataFrame
                    soccer_cols = ['home_conversion', 'away_conversion', 'home_accuracy', 'away_accuracy']
                    df.loc[df['sport'] == 'soccer', soccer_cols] = soccer_df[soccer_cols]

                logger.info("Added soccer-specific features")

            if 'basketball' in df['sport'].values:
                # For basketball, we might add different types of features
                # This is a placeholder for basketball-specific features
                logger.info("Basketball feature engineering would go here")

        return df

    def _save_raw_data(self, data: pd.DataFrame, filename: str) -> None:
        """Save raw data to the raw data directory."""
        if data.empty:
            return

        try:
            filepath = RAW_DATA_DIR / filename
            data.to_csv(filepath, index=False)
            logger.info(f"Saved raw data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving raw data: {e}")

    def generate_synthetic_data(self, template_data: pd.DataFrame, num_samples: int = 100,
                             noise_level: float = 0.1) -> pd.DataFrame:
        """
        Generate synthetic data points based on template data.

        Args:
            template_data: DataFrame to use as a template
            num_samples: Number of synthetic samples to generate
            noise_level: Amount of noise to add (0.0 to 1.0)

        Returns:
            DataFrame with synthetic data
        """
        if template_data.empty:
            logger.warning("Cannot generate synthetic data from empty template")
            return pd.DataFrame()

        logger.info(f"Generating {num_samples} synthetic data points")

        # Make a copy of the template
        synthetic_data = pd.DataFrame(columns=template_data.columns)

        # Get numeric columns for adding noise
        numeric_cols = template_data.select_dtypes(include=[np.number]).columns.tolist()

        # Generate synthetic samples
        for _ in range(num_samples):
            # Randomly select a row from the template
            template_row = template_data.sample(n=1).iloc[0]

            # Create a new row with the same values
            new_row = template_row.copy()

            # Add noise to numeric columns
            for col in numeric_cols:
                base_value = new_row[col]
                if base_value == 0:
                    continue  # Skip zero values to avoid multiplication issues

                # Add random noise proportional to the original value
                noise = np.random.normal(0, noise_level * abs(base_value))
                new_row[col] = base_value + noise

            # Add to synthetic dataset
            synthetic_data = pd.concat([synthetic_data, pd.DataFrame([new_row])], ignore_index=True)

        logger.info(f"Generated {len(synthetic_data)} synthetic samples")

        return synthetic_data

    def close(self):
        """Close any open resources."""
        if self.selenium_driver is not None:
            try:
                self.selenium_driver.quit()
                logger.info("Closed Selenium WebDriver")
            except Exception as e:
                logger.error(f"Error closing Selenium WebDriver: {e}")
            finally:
                self.selenium_driver = None

if __name__ == "__main__":
    # Example usage
    fetcher = WebDataFetcher()

    try:
        # Example 1: Fetch forex data
        forex_data = fetcher.fetch_forex_data("EUR/USD", "daily", "compact")

        # Example 2: Validate data
        validation_rules = {
            "close": {
                "allow_missing": False,
                "min_value": 0.5,
                "max_value": 2.0,
                "max_zscore": 3.0
            },
            "open": {
                "allow_missing": False,
                "min_value": 0.5
            }
        }

        is_valid, valid_data, results = fetcher.validate_data(forex_data, validation_rules)

        # Example 3: Preprocess data
        preprocessing_config = {
            "handle_missing": True,
            "missing_strategy": "interpolate",
            "normalize": True,
            "scaling_method": "minmax",
            "feature_engineering": True,
            "domain": "forex"
        }

        processed_data = fetcher.preprocess_data(valid_data, preprocessing_config)

        # Save processed data
        processed_data.to_csv(PROCESSED_DATA_DIR / "eur_usd_processed.csv", index=False)

        print(f"Data processing complete. Processed {len(processed_data)} rows.")

    finally:
        # Always close resources properly
        fetcher.close()


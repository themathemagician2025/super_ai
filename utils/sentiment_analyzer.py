# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import re
import random
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import datetime

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Class for analyzing sentiment in news, social media, and other text sources
    related to forex and betting markets.
    """

    def __init__(self, config_path: Path = None):
        # Sentiment analysis configuration
        self.use_external_api = False
        self.api_key = None
        self.api_endpoint = None
        self.cache_duration = 3600  # 1 hour in seconds
        self.sentiment_cache = {}  # Cache for sentiment results
        self.positive_words = set(['bullish', 'growth', 'profit', 'boom', 'success', 'gain', 'positive',
                                  'rally', 'recovery', 'strong', 'upbeat', 'uptrend', 'win'])
        self.negative_words = set(['bearish', 'recession', 'loss', 'crash', 'crisis', 'decline', 'negative',
                                  'fall', 'weak', 'downtrend', 'risk', 'worried', 'fear'])
        self.sources = ['twitter', 'news', 'reddit', 'blogs']

        # Load configuration if provided
        if config_path:
            self._load_config(config_path)

        logger.info("Sentiment Analyzer initialized")

    def _load_config(self, config_path: Path):
        """Load sentiment analyzer configuration"""
        try:
            import yaml
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            if config:
                if 'sentiment' in config:
                    cfg = config['sentiment']
                    self.use_external_api = cfg.get('use_external_api', self.use_external_api)
                    self.api_key = cfg.get('api_key', self.api_key)
                    self.api_endpoint = cfg.get('api_endpoint', self.api_endpoint)
                    self.cache_duration = cfg.get('cache_duration', self.cache_duration)

                    # Load custom word lists if provided
                    if 'positive_words' in cfg:
                        self.positive_words.update(cfg['positive_words'])
                    if 'negative_words' in cfg:
                        self.negative_words.update(cfg['negative_words'])
                    if 'sources' in cfg:
                        self.sources = cfg['sources']

                    logger.info(f"Loaded sentiment analyzer configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading sentiment analyzer configuration: {str(e)}")

    def analyze_sentiment(self, target: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Analyze sentiment for a specific target (e.g., currency pair, sports team).
        Returns a dictionary containing sentiment scores and related data.
        """
        # Check cache first
        if not force_refresh and target in self.sentiment_cache:
            cache_time, cache_data = self.sentiment_cache[target]
            current_time = datetime.datetime.now().timestamp()

            # If cache is still valid, return cached data
            if current_time - cache_time < self.cache_duration:
                logger.debug(f"Using cached sentiment for {target}")
                return cache_data

        try:
            # If external API is configured, use it
            if self.use_external_api and self.api_key and self.api_endpoint:
                sentiment_data = self._analyze_with_external_api(target)
            else:
                # Otherwise use basic internal analysis
                sentiment_data = self._analyze_internally(target)

            # Cache the result
            self.sentiment_cache[target] = (datetime.datetime.now().timestamp(), sentiment_data)

            return sentiment_data
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {target}: {str(e)}")
            return {
                "score": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "sources": [],
                "error": str(e)
            }

    def _analyze_with_external_api(self, target: str) -> Dict[str, Any]:
        """Use external API for sentiment analysis"""
        try:
            # In a real implementation, we would make an HTTP request to the API
            # For this example, we'll simulate a response
            logger.info(f"Using external API to analyze sentiment for {target}")

            # Simulate API call delay
            import time
            time.sleep(0.5)

            # Simulate API response
            # In reality, we would parse the actual API response
            is_forex = any(pair in target for pair in ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"])
            is_sports = any(sport in target.lower() for sport in ["football", "soccer", "tennis", "basketball", "baseball"])

            if is_forex:
                # Generate slightly different sentiment for forex
                sentiment = {
                    "score": round(random.uniform(-1.0, 1.0), 2),
                    "magnitude": round(random.uniform(0.1, 5.0), 2),
                    "articles": random.randint(50, 200),
                    "tweets": random.randint(500, 5000),
                    "timeframe": "24h",
                }
            elif is_sports:
                # Generate slightly different sentiment for sports
                sentiment = {
                    "score": round(random.uniform(-0.5, 0.5), 2),
                    "magnitude": round(random.uniform(0.5, 3.0), 2),
                    "tweets": random.randint(1000, 10000),
                    "reddit_posts": random.randint(50, 500),
                    "news_articles": random.randint(10, 50),
                    "timeframe": "24h",
                }
            else:
                # Generic sentiment
                sentiment = {
                    "score": round(random.uniform(-0.3, 0.3), 2),
                    "magnitude": round(random.uniform(0.1, 2.0), 2),
                    "sources": random.randint(10, 100),
                    "timeframe": "24h",
                }

            # Convert to our standard format
            positive = max(0, sentiment["score"])
            negative = max(0, -sentiment["score"])
            neutral = 1.0 - (positive + negative)

            return {
                "score": sentiment["score"],
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "magnitude": sentiment.get("magnitude", 1.0),
                "sources": sentiment.get("sources", ["api"]),
                "count": sentiment.get("articles", 0) + sentiment.get("tweets", 0) + sentiment.get("reddit_posts", 0) + sentiment.get("news_articles", 0),
                "timeframe": sentiment.get("timeframe", "24h")
            }
        except Exception as e:
            logger.error(f"Error using external API for sentiment analysis: {str(e)}")
            raise

    def _analyze_internally(self, target: str) -> Dict[str, Any]:
        """
        Simple rule-based sentiment analysis without external API.
        In a real system, this would use NLP libraries like NLTK or spaCy.
        """
        # For this example, we'll generate pseudo-random but consistent sentiment
        # based on the target string

        # Create a seed from the target string for consistent randomness
        seed = sum(ord(c) for c in target)
        random.seed(seed)

        # Generate base sentiment score (-1.0 to 1.0)
        base_score = random.uniform(-0.5, 0.5)

        # Adjust score based on keywords in target
        for word in self.positive_words:
            if word.lower() in target.lower():
                base_score += 0.1

        for word in self.negative_words:
            if word.lower() in target.lower():
                base_score -= 0.1

        # Clamp to -1.0 to 1.0 range
        base_score = max(-1.0, min(1.0, base_score))

        # Calculate positive, negative, and neutral components
        positive = max(0, base_score)
        negative = max(0, -base_score)
        neutral = 1.0 - (positive + negative)

        # Generate sample sources
        num_sources = random.randint(3, 10)
        selected_sources = random.sample(self.sources * 3, num_sources)  # * 3 to ensure enough items

        # Generate a count of documents analyzed
        doc_count = random.randint(50, 500)

        return {
            "score": base_score,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "magnitude": random.uniform(0.5, 3.0),
            "sources": selected_sources,
            "count": doc_count,
            "timeframe": "24h"
        }

    def analyze_multiple(self, targets: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze sentiment for multiple targets at once.
        Returns a dictionary mapping each target to its sentiment data.
        """
        results = {}
        for target in targets:
            results[target] = self.analyze_sentiment(target)
        return results

    def analyze_for_forex(self, currency_pair: str) -> Dict[str, Any]:
        """
        Specialized sentiment analysis for forex markets.
        Includes currency-specific sentiment and news impact.
        """
        # Parse the currency pair
        try:
            base_currency, quote_currency = currency_pair.split('/')
        except ValueError:
            if len(currency_pair) == 6:
                base_currency, quote_currency = currency_pair[:3], currency_pair[3:]
            else:
                logger.warning(f"Could not parse currency pair: {currency_pair}")
                base_currency, quote_currency = "Unknown", "Unknown"

        # Get general sentiment for the pair
        pair_sentiment = self.analyze_sentiment(currency_pair)

        # Get sentiment for each currency individually
        base_sentiment = self.analyze_sentiment(base_currency)
        quote_sentiment = self.analyze_sentiment(quote_currency)

        # Calculate relative currency strength
        base_strength = base_sentiment["score"]
        quote_strength = quote_sentiment["score"]
        relative_strength = base_strength - quote_strength

        # Generate economic news sentiment
        economic_news = self._generate_economic_news(base_currency, quote_currency)

        # Combine all data
        result = {
            "pair_sentiment": pair_sentiment,
            "base_currency": {
                "currency": base_currency,
                "sentiment": base_sentiment,
                "economic_news": economic_news.get(base_currency, [])
            },
            "quote_currency": {
                "currency": quote_currency,
                "sentiment": quote_sentiment,
                "economic_news": economic_news.get(quote_currency, [])
            },
            "relative_strength": relative_strength,
            "forecast": {
                "direction": "bullish" if relative_strength > 0 else "bearish",
                "strength": abs(relative_strength) * 5  # Scale to 0-5
            },
            "momentum": {
                "short_term": random.uniform(-1.0, 1.0),
                "medium_term": random.uniform(-0.8, 0.8),
                "long_term": random.uniform(-0.6, 0.6)
            }
        }

        # Add USD impact for non-USD pairs
        if "USD" not in currency_pair:
            result["usd_impact"] = self.analyze_sentiment("USD")

        return result

    def _generate_economic_news(self, *currencies) -> Dict[str, List[Dict[str, Any]]]:
        """Generate simulated economic news for currencies"""
        news_types = [
            "Interest Rate Decision", "GDP Report", "Inflation Data",
            "Employment Report", "Trade Balance", "Retail Sales",
            "Manufacturing PMI", "Consumer Confidence", "Housing Data"
        ]

        impacts = ["High", "Medium", "Low"]
        sentiments = ["Positive", "Negative", "Neutral"]

        result = {}
        for currency in currencies:
            # Generate 1-3 news items per currency
            num_news = random.randint(1, 3)
            currency_news = []

            for _ in range(num_news):
                news_type = random.choice(news_types)
                impact = random.choice(impacts)
                sentiment = random.choice(sentiments)

                # Generate a date within the last week
                days_ago = random.randint(0, 7)
                date = (datetime.datetime.now() - datetime.timedelta(days=days_ago)).strftime("%Y-%m-%d")

                currency_news.append({
                    "title": f"{currency} {news_type}",
                    "date": date,
                    "impact": impact,
                    "sentiment": sentiment,
                    "summary": f"The {news_type} for {currency} was released, showing {'better' if sentiment == 'Positive' else 'worse' if sentiment == 'Negative' else 'as expected'} than expected results."
                })

            result[currency] = currency_news

        return result

    def analyze_for_sports(self, team_or_match: str) -> Dict[str, Any]:
        """
        Specialized sentiment analysis for sports betting.
        Includes team sentiment, match hype, and fan confidence.
        """
        # Check if it's a match (contains 'vs' or 'v' or '-')
        is_match = any(sep in team_or_match for sep in [" vs ", " v ", " - "])

        if is_match:
            # Parse team names
            for separator in [" vs ", " v ", " - "]:
                if separator in team_or_match:
                    team1, team2 = team_or_match.split(separator, 1)
                    team1 = team1.strip()
                    team2 = team2.strip()
                    break
            else:
                team1, team2 = "Unknown", "Unknown"

            # Get sentiment for each team
            team1_sentiment = self.analyze_sentiment(team1)
            team2_sentiment = self.analyze_sentiment(team2)
            match_sentiment = self.analyze_sentiment(team_or_match)

            # Calculate match hype (how much buzz/discussion)
            match_hype = (team1_sentiment.get("count", 0) + team2_sentiment.get("count", 0)) / 1000
            match_hype = min(1.0, match_hype)  # Cap at 1.0

            # Analyze fan confidence based on sentiment
            team1_confidence = (team1_sentiment["positive"] - team1_sentiment["negative"] + 1) / 2
            team2_confidence = (team2_sentiment["positive"] - team2_sentiment["negative"] + 1) / 2

            # Generate expert predictions
            expert_predictions = self._generate_expert_predictions(team1, team2)

            # Combine all data
            result = {
                "match": team_or_match,
                "teams": [
                    {
                        "name": team1,
                        "sentiment": team1_sentiment,
                        "fan_confidence": team1_confidence
                    },
                    {
                        "name": team2,
                        "sentiment": team2_sentiment,
                        "fan_confidence": team2_confidence
                    }
                ],
                "match_sentiment": match_sentiment,
                "match_hype": match_hype,
                "expert_predictions": expert_predictions,
                "betting_trends": {
                    "team1_percentage": round(team1_confidence * 100),
                    "team2_percentage": round(team2_confidence * 100),
                    "volume_trend": "Increasing" if match_hype > 0.7 else "Stable" if match_hype > 0.4 else "Low"
                }
            }
        else:
            # Single team analysis
            team_sentiment = self.analyze_sentiment(team_or_match)

            # Calculate fan confidence
            fan_confidence = (team_sentiment["positive"] - team_sentiment["negative"] + 1) / 2

            # Recent form (fictional)
            seed = sum(ord(c) for c in team_or_match)
            random.seed(seed)
            recent_results = []
            for _ in range(5):
                result = random.choice(["W", "L", "D"])
                recent_results.append(result)

            # Combine data
            result = {
                "team": team_or_match,
                "sentiment": team_sentiment,
                "fan_confidence": fan_confidence,
                "recent_form": recent_results,
                "momentum": "Positive" if recent_results.count("W") > recent_results.count("L") else "Negative",
                "buzz_level": team_sentiment.get("count", 0) / 500  # Normalized buzz level
            }

        return result

    def _generate_expert_predictions(self, team1: str, team2: str) -> List[Dict[str, Any]]:
        """Generate simulated expert predictions for a match"""
        expert_names = [
            "Sports Analytics Pro", "Betting Expert", "Former Coach",
            "Statistics Guru", "Industry Insider", "Fan Favorite Analyst"
        ]

        # Create a seed from the team names for consistency
        seed = sum(ord(c) for c in team1 + team2)
        random.seed(seed)

        # Shuffle expert names to get different experts each time
        shuffled_experts = expert_names.copy()
        random.shuffle(shuffled_experts)

        # Generate 2-4 expert predictions
        num_predictions = random.randint(2, 4)
        predictions = []

        team1_win_prob = random.uniform(0.3, 0.7)
        draw_prob = random.uniform(0.1, 0.3)
        team2_win_prob = 1.0 - team1_win_prob - draw_prob

        for i in range(num_predictions):
            # Add some variance to each expert's prediction
            variance = random.uniform(-0.1, 0.1)
            expert_team1_prob = max(0.1, min(0.9, team1_win_prob + variance))
            expert_draw_prob = max(0.05, min(0.3, draw_prob + variance/2))
            expert_team2_prob = 1.0 - expert_team1_prob - expert_draw_prob

            # Pick a winner based on these probabilities
            winner = random.choices(
                [team1, "Draw", team2],
                weights=[expert_team1_prob, expert_draw_prob, expert_team2_prob]
            )[0]

            confidence = random.uniform(0.6, 0.9)

            # Generate a rationale
            if winner == team1:
                rationale = f"{team1} has shown better form recently."
            elif winner == team2:
                rationale = f"{team2} has the edge in key matchups."
            else:
                rationale = "The teams are evenly matched, expect a draw."

            predictions.append({
                "expert": shuffled_experts[i],
                "prediction": winner,
                "confidence": confidence,
                "rationale": rationale,
                "probabilities": {
                    team1: round(expert_team1_prob, 2),
                    "Draw": round(expert_draw_prob, 2),
                    team2: round(expert_team2_prob, 2)
                }
            })

        return predictions

    def get_trending_topics(self) -> List[Dict[str, Any]]:
        """Get trending topics related to forex or sports betting"""
        # In a real implementation, this would retrieve actual trending topics
        # For this example, we'll generate some fictional trending topics

        forex_topics = [
            {"topic": "Fed Interest Rate Decision", "sentiment": 0.2, "momentum": "rising"},
            {"topic": "EUR/USD Technical Analysis", "sentiment": -0.1, "momentum": "stable"},
            {"topic": "JPY Weakness Concerns", "sentiment": -0.3, "momentum": "rising"},
            {"topic": "GBP Brexit Developments", "sentiment": 0.1, "momentum": "falling"},
            {"topic": "USD Inflation Impact", "sentiment": -0.2, "momentum": "rising"}
        ]

        sports_topics = [
            {"topic": "Premier League Title Race", "sentiment": 0.4, "momentum": "rising"},
            {"topic": "NBA Playoff Predictions", "sentiment": 0.3, "momentum": "stable"},
            {"topic": "Tennis Grand Slam Preview", "sentiment": 0.2, "momentum": "rising"},
            {"topic": "Major Upset in Boxing", "sentiment": 0.5, "momentum": "rising"},
            {"topic": "Horse Racing Derby Favorites", "sentiment": 0.1, "momentum": "stable"}
        ]

        # Select a random mix of topics
        num_forex = random.randint(1, 3)
        num_sports = random.randint(1, 3)

        selected_forex = random.sample(forex_topics, num_forex)
        selected_sports = random.sample(sports_topics, num_sports)

        # Combine and add category
        for topic in selected_forex:
            topic["category"] = "forex"

        for topic in selected_sports:
            topic["category"] = "sports"

        return selected_forex + selected_sports

    def clear_cache(self):
        """Clear the sentiment cache"""
        self.sentiment_cache = {}
        logger.info("Sentiment cache cleared")

    def update_wordlists(self, positive_words: List[str] = None, negative_words: List[str] = None):
        """Update the positive and negative word lists"""
        if positive_words:
            self.positive_words.update(positive_words)
            logger.info(f"Added {len(positive_words)} positive words to sentiment analyzer")

        if negative_words:
            self.negative_words.update(negative_words)
            logger.info(f"Added {len(negative_words)} negative words to sentiment analyzer")

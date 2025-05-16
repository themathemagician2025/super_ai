# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import random
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import datetime
import re

logger = logging.getLogger(__name__)

class ForexPredictor:
    """
    Predictor for forex market trends and currency pair movements
    using advanced algorithms and technical analysis.
    """

    def __init__(self, config_path: Path = None):
        # Forex prediction configuration
        self.api_endpoint = None
        self.api_key = None
        self.cache_duration = 3600  # 1 hour in seconds
        self.prediction_cache = {}  # Cache for predictions

        # Supported currency pairs
        self.supported_pairs = [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
            "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP",
            "EUR/JPY", "GBP/JPY"
        ]

        # Historical performance tracking
        self.performance_history = []

        # Technical indicators
        self.indicators = [
            "MA", "EMA", "RSI", "MACD", "Bollinger",
            "Stochastic", "Fibonacci", "Ichimoku"
        ]

        # Load configuration if provided
        if config_path:
            self._load_config(config_path)

        logger.info("Forex Predictor initialized")

    def _load_config(self, config_path: Path):
        """Load forex predictor configuration"""
        try:
            import yaml
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            if config and 'forex_predictor' in config:
                cfg = config['forex_predictor']
                self.api_endpoint = cfg.get('api_endpoint', self.api_endpoint)
                self.api_key = cfg.get('api_key', self.api_key)
                self.cache_duration = cfg.get('cache_duration', self.cache_duration)

                logger.info(f"Loaded forex predictor configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading forex predictor configuration: {str(e)}")

    def predict(self, pair: str, timeframe: str = "1d", lot_size: str = "1.0") -> Dict[str, Any]:
        """
        Generate predictions for a specific currency pair.
        Returns a dictionary with trend direction, strength, targets, and analysis.
        """
        pair = self._normalize_pair(pair)
        cache_key = f"{pair}:{timeframe}:{lot_size}"

        # Check cache first
        if cache_key in self.prediction_cache:
            cache_time, cache_data = self.prediction_cache[cache_key]
            current_time = datetime.datetime.now().timestamp()

            # If cache is still valid, return cached data
            if current_time - cache_time < self.cache_duration:
                logger.debug(f"Using cached prediction for {pair} ({timeframe})")
                return cache_data

        try:
            # Parse the pair
            if "/" in pair:
                base_currency, quote_currency = pair.split("/")
            else:
                base_currency, quote_currency = pair[:3], pair[3:]

            # Generate base prediction using technical analysis
            prediction = self._generate_technical_prediction(pair, timeframe)

            # Add fundamental analysis
            prediction["analysis"] = self._generate_fundamental_analysis(base_currency, quote_currency)

            # Add risk assessment based on lot size
            prediction["risk"] = self._assess_risk(pair, timeframe, lot_size)

            # Add metadata
            prediction.update({
                "pair": pair,
                "timeframe": timeframe,
                "lot_size": lot_size,
                "timestamp": datetime.datetime.now().isoformat()
            })

            # Cache the result
            self.prediction_cache[cache_key] = (datetime.datetime.now().timestamp(), prediction)

            # Log successful prediction
            logger.info(f"Generated forex prediction for {pair} ({timeframe})")

            return prediction
        except Exception as e:
            logger.error(f"Error generating prediction for {pair}: {str(e)}")
            # Return a minimal prediction with error information
            return {
                "pair": pair,
                "timeframe": timeframe,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }

    def _normalize_pair(self, pair: str) -> str:
        """Normalize currency pair format (e.g., EURUSD -> EUR/USD)"""
        if "/" in pair:
            return pair.upper()

        if len(pair) == 6:
            return f"{pair[:3]}/{pair[3:]}".upper()

        return pair.upper()

    def _generate_technical_prediction(self, pair: str, timeframe: str) -> Dict[str, Any]:
        """Generate prediction based on technical analysis"""
        # Create a seed from the pair and timeframe for consistent randomness
        seed = sum(ord(c) for c in pair + timeframe)
        random.seed(seed)

        # Generate a trend prediction (up or down)
        trend_value = random.uniform(-1.0, 1.0)
        trend_direction = "up" if trend_value > 0 else "down"
        trend_strength = abs(trend_value) * 10  # Scale to 0-10

        # Current price (fictional)
        base_price = random.uniform(0.5, 2.0)
        if "JPY" in pair:
            # JPY pairs typically have higher values
            base_price = random.uniform(80, 150)

        # Calculate potential targets based on trend
        pip_value = 0.0001 if "JPY" not in pair else 0.01
        potential_movement = random.uniform(30, 100) * pip_value

        if trend_direction == "up":
            target = base_price + potential_movement
            stop_loss = base_price - (potential_movement * 0.5)
        else:
            target = base_price - potential_movement
            stop_loss = base_price + (potential_movement * 0.5)

        # Format prices to appropriate decimal places
        if "JPY" in pair:
            base_price = round(base_price, 3)
            target = round(target, 3)
            stop_loss = round(stop_loss, 3)
        else:
            base_price = round(base_price, 5)
            target = round(target, 5)
            stop_loss = round(stop_loss, 5)

        # Technical indicators with random signals
        indicators = {}
        for indicator in random.sample(self.indicators, 4):  # Pick 4 random indicators
            signal = random.choice(["buy", "sell", "neutral"])
            strength = random.uniform(0.1, 1.0)
            indicators[indicator] = {"signal": signal, "strength": strength}

        # Overall signal strength based on indicator consensus
        buy_signals = sum(1 for ind in indicators.values() if ind["signal"] == "buy")
        sell_signals = sum(1 for ind in indicators.values() if ind["signal"] == "sell")

        signal_strength = 0.5  # Neutral by default
        if buy_signals > sell_signals:
            signal_strength = 0.5 + (buy_signals - sell_signals) / (2 * len(indicators))
        elif sell_signals > buy_signals:
            signal_strength = 0.5 - (sell_signals - buy_signals) / (2 * len(indicators))

        # Compile prediction
        prediction = {
            "trend": {
                "direction": trend_direction,
                "strength": round(trend_strength, 1),
                "current_price": base_price,
                "target": target,
                "stop_loss": stop_loss
            },
            "technical": {
                "indicators": indicators,
                "signal_strength": signal_strength,
                "confidence": random.uniform(0.6, 0.9)
            },
            "timeframes": self._generate_multi_timeframe_outlook(pair, timeframe)
        }

        return prediction

    def _generate_multi_timeframe_outlook(self, pair: str, base_timeframe: str) -> Dict[str, str]:
        """Generate outlook across multiple timeframes"""
        # Map timeframes to their relative strength
        timeframe_map = {
            "1m": 1, "5m": 2, "15m": 3, "30m": 4, "1h": 5,
            "4h": 6, "1d": 7, "1w": 8, "1M": 9
        }

        # Get index of base timeframe
        base_index = timeframe_map.get(base_timeframe, 5)  # Default to 1h if unknown

        # Create a seed for consistent randomness
        seed = sum(ord(c) for c in pair) + base_index
        random.seed(seed)

        # Base trend bias
        trend_bias = random.uniform(-1.0, 1.0)

        # Generate outlooks for different timeframes
        outlook = {}

        for tf, tf_index in timeframe_map.items():
            # Adjust trend based on timeframe relation to base
            # Higher timeframes have more weight and less variation
            adjustment = (tf_index - base_index) * 0.1
            timeframe_trend = trend_bias + adjustment + random.uniform(-0.3, 0.3)

            if timeframe_trend > 0.2:
                outlook[tf] = "bullish"
            elif timeframe_trend < -0.2:
                outlook[tf] = "bearish"
            else:
                outlook[tf] = "neutral"

        return outlook

    def _generate_fundamental_analysis(self, base_currency: str, quote_currency: str) -> Dict[str, str]:
        """Generate fundamental analysis for currency pair"""
        # Create a seed for consistent randomness
        seed = sum(ord(c) for c in base_currency + quote_currency)
        random.seed(seed)

        # Generate analysis for each major factor

        # 1. USD Dynamics
        usd_factors = [
            "Federal Reserve policies",
            "interest rate expectations",
            "inflation data",
            "employment figures",
            "GDP growth",
            "trade balance"
        ]

        usd_dynamics = ""
        if "USD" in [base_currency, quote_currency]:
            usd_strength = random.uniform(-1.0, 1.0)
            usd_factor = random.choice(usd_factors)

            if usd_strength > 0.3:
                usd_dynamics = f"The USD is showing strength due to {usd_factor}, which may {'boost' if quote_currency == 'USD' else 'pressure'} {base_currency}/{quote_currency}."
            elif usd_strength < -0.3:
                usd_dynamics = f"The USD is showing weakness due to {usd_factor}, which may {'pressure' if quote_currency == 'USD' else 'boost'} {base_currency}/{quote_currency}."
            else:
                usd_dynamics = f"The USD is relatively stable with {usd_factor} having minimal impact on {base_currency}/{quote_currency}."

        # 2. Geopolitical factors
        geo_factors = [
            "trade tensions",
            "political uncertainty",
            "diplomatic relations",
            "regional conflicts",
            "economic sanctions",
            "Brexit developments"
        ]

        geo_factor = random.choice(geo_factors)
        geo_impact = random.uniform(-1.0, 1.0)

        if abs(geo_impact) > 0.5:
            geopolitical = f"Current {geo_factor} are {'positively' if geo_impact > 0 else 'negatively'} affecting {base_currency} relative to {quote_currency}."
        else:
            geopolitical = f"{geo_factor.capitalize()} have limited impact on {base_currency}/{quote_currency} at this time."

        # 3. Technical trend analysis
        tech_factors = [
            "moving averages",
            "support/resistance levels",
            "chart patterns",
            "momentum indicators",
            "volume analysis",
            "Fibonacci retracements"
        ]

        tech_factor = random.choice(tech_factors)
        tech_signal = random.choice(["bullish", "bearish", "neutral"])
        tech_strength = random.uniform(0.3, 0.9)

        if tech_signal == "bullish":
            technical = f"{base_currency}/{quote_currency} shows bullish signals based on {tech_factor}, with {'strong' if tech_strength > 0.6 else 'moderate'} upward momentum."
        elif tech_signal == "bearish":
            technical = f"{base_currency}/{quote_currency} shows bearish signals based on {tech_factor}, with {'strong' if tech_strength > 0.6 else 'moderate'} downward pressure."
        else:
            technical = f"{base_currency}/{quote_currency} is consolidating according to {tech_factor}, with no clear directional bias."

        # 4. Market sentiment
        sentiment_factors = [
            "institutional positioning",
            "retail sentiment",
            "social media buzz",
            "futures market positioning",
            "options market activity",
            "analyst consensus"
        ]

        sentiment_factor = random.choice(sentiment_factors)
        sentiment_signal = random.uniform(-1.0, 1.0)

        if sentiment_signal > 0.3:
            sentiment = f"Market sentiment from {sentiment_factor} is bullish on {base_currency}/{quote_currency}, indicating potential upward movement."
        elif sentiment_signal < -0.3:
            sentiment = f"Market sentiment from {sentiment_factor} is bearish on {base_currency}/{quote_currency}, suggesting potential downward pressure."
        else:
            sentiment = f"Market sentiment from {sentiment_factor} is mixed on {base_currency}/{quote_currency}, with no clear consensus."

        # Compile analysis
        analysis = {
            "usd_dynamics": usd_dynamics,
            "geopolitical": geopolitical,
            "technical": technical,
            "sentiment": sentiment
        }

        return analysis

    def _assess_risk(self, pair: str, timeframe: str, lot_size: str) -> Dict[str, Any]:
        """Assess risk based on pair volatility and lot size"""
        try:
            # Convert lot size to float
            lot_value = float(lot_size)

            # Create a seed for consistent randomness
            seed = sum(ord(c) for c in pair + timeframe)
            random.seed(seed)

            # Baseline volatility by pair
            pair_volatility = {
                "EUR/USD": 0.5, "GBP/USD": 0.7, "USD/JPY": 0.6, "USD/CHF": 0.6,
                "AUD/USD": 0.8, "USD/CAD": 0.7, "NZD/USD": 0.8, "EUR/GBP": 0.6,
                "EUR/JPY": 0.8, "GBP/JPY": 0.9
            }

            # Get volatility with fallback
            volatility = pair_volatility.get(pair, 0.7)

            # Adjust volatility based on timeframe
            timeframe_volatility = {
                "1m": 1.5, "5m": 1.3, "15m": 1.2, "30m": 1.1, "1h": 1.0,
                "4h": 0.9, "1d": 0.8, "1w": 0.7, "1M": 0.6
            }

            volatility *= timeframe_volatility.get(timeframe, 1.0)

            # Add some random variation
            volatility *= random.uniform(0.9, 1.1)

            # Calculate risk level based on volatility and lot size
            # Higher volatility or lot size = higher risk
            risk_level = min(10, volatility * lot_value * 5)

            # Position size recommendation based on risk
            if risk_level > 7:
                position_rec = "Consider reducing position size"
            elif risk_level > 4:
                position_rec = "Current position size is moderate risk"
            else:
                position_rec = "Position size is conservative"

            # Calculate potential pip exposure
            pip_value = 10 if "JPY" in pair else 1
            pip_exposure = lot_value * pip_value

            # Compile risk assessment
            risk = {
                "level": round(risk_level, 1),
                "volatility": round(volatility, 2),
                "position_size": lot_size,
                "pip_exposure": pip_exposure,
                "recommendation": position_rec
            }

            return risk
        except Exception as e:
            logger.error(f"Error assessing risk: {str(e)}")
            return {
                "level": 5.0,
                "error": str(e),
                "recommendation": "Unable to accurately assess risk"
            }

    def update(self):
        """Update the forex predictor models"""
        # In a real implementation, this would update the models with new data
        # For this example, we'll just log a message
        logger.info("Forex predictor models would be updated with new data here")
        return True

    def get_strategy(self, pair: str, timeframe: str) -> Dict[str, Any]:
        """Generate a trading strategy for a currency pair"""
        # Normalize pair format
        pair = self._normalize_pair(pair)

        # Get the base prediction
        prediction = self.predict(pair, timeframe)

        # Create a seed for consistent randomness
        seed = sum(ord(c) for c in pair + timeframe)
        random.seed(seed)

        # Define strategy types
        strategy_types = [
            "Trend Following", "Breakout", "Range Trading",
            "Swing Trading", "Scalping", "Position Trading"
        ]

        # Define entry/exit rules
        entry_rules = [
            "Moving Average Crossover", "RSI Oversold/Overbought",
            "MACD Signal Line Crossover", "Support/Resistance Breakout",
            "Bollinger Band Squeeze", "Fibonacci Retracement"
        ]

        exit_rules = [
            "Take Profit at Key Level", "Stop Loss on Reversal Signal",
            "Trailing Stop Loss", "Time-Based Exit", "Opposite Signal"
        ]

        # Select strategy components based on trend prediction
        trending = prediction["trend"]["strength"] > 5.0
        trend_direction = prediction["trend"]["direction"]

        if trending:
            # For strong trends, use trend following or breakout
            strategy_type = random.choice(["Trend Following", "Breakout"])
        else:
            # For weaker trends, use range or swing trading
            strategy_type = random.choice(["Range Trading", "Swing Trading"])

        # Select entry rule
        entry_rule = random.choice(entry_rules)

        # Select exit rule
        exit_rule = random.choice(exit_rules)

        # Generate risk management parameters
        risk_reward = random.choice([1.5, 2.0, 2.5, 3.0])
        max_risk_percent = random.uniform(1.0, 3.0)

        # Compile strategy
        strategy = {
            "pair": pair,
            "timeframe": timeframe,
            "strategy_type": strategy_type,
            "bias": trend_direction,
            "entry": {
                "rule": entry_rule,
                "description": f"Enter {trend_direction} positions when {entry_rule} occurs",
                "confirmation": random.choice(entry_rules)
            },
            "exit": {
                "rule": exit_rule,
                "description": f"Exit positions when {exit_rule} occurs",
                "take_profit": prediction["trend"]["target"],
                "stop_loss": prediction["trend"]["stop_loss"]
            },
            "risk_management": {
                "risk_reward_ratio": risk_reward,
                "max_risk_percent": round(max_risk_percent, 1),
                "position_sizing": "Based on stop loss distance"
            },
            "prediction": prediction
        }

        return strategy

    def analyze_correlations(self, pairs: List[str]) -> Dict[str, Any]:
        """Analyze correlations between currency pairs"""
        # Normalize all pairs
        pairs = [self._normalize_pair(pair) for pair in pairs]

        # Create a seed for consistent randomness
        seed = sum(ord(c) for c in "".join(pairs))
        random.seed(seed)

        # Generate correlation matrix
        correlation_matrix = {}
        for i, pair1 in enumerate(pairs):
            correlation_matrix[pair1] = {}
            for j, pair2 in enumerate(pairs):
                if i == j:
                    # Perfect correlation with self
                    correlation_matrix[pair1][pair2] = 1.0
                else:
                    # Generate pseudo-random but consistent correlation
                    # Pairs with same currencies tend to be more correlated
                    base_correlation = random.uniform(-0.9, 0.9)

                    # Adjust correlation based on common currencies
                    base1, quote1 = pair1.split("/")
                    base2, quote2 = pair2.split("/")

                    if base1 == base2 or quote1 == quote2:
                        # Same base or quote currency
                        base_correlation = (base_correlation + 0.7) / 2
                    elif base1 == quote2 or quote1 == base2:
                        # Inverse relationship
                        base_correlation = (base_correlation - 0.7) / 2

                    correlation_matrix[pair1][pair2] = round(base_correlation, 2)

        # Generate insights based on correlations
        strong_positive = []
        strong_negative = []

        for pair1 in correlation_matrix:
            for pair2, corr in correlation_matrix[pair1].items():
                if pair1 != pair2:
                    if corr > 0.7:
                        if f"{pair2} and {pair1}" not in strong_positive:
                            strong_positive.append(f"{pair1} and {pair2}")
                    elif corr < -0.7:
                        if f"{pair2} and {pair1}" not in strong_negative:
                            strong_negative.append(f"{pair1} and {pair2}")

        # Generate diversification advice
        if strong_positive:
            diversification = f"Consider avoiding simultaneous positions in {', '.join(strong_positive)} due to strong positive correlation."
        else:
            diversification = "No strong positive correlations found, suggesting good diversification potential."

        # Generate trading opportunities
        if strong_negative:
            hedge_opportunity = f"Potential hedging opportunity exists between {', '.join(strong_negative)} due to negative correlation."
        else:
            hedge_opportunity = "No strong negative correlations found for effective hedging."

        # Compile correlation analysis
        result = {
            "correlation_matrix": correlation_matrix,
            "insights": {
                "strong_positive_correlations": strong_positive,
                "strong_negative_correlations": strong_negative,
                "diversification_advice": diversification,
                "hedge_opportunities": hedge_opportunity
            },
            "timestamp": datetime.datetime.now().isoformat()
        }

        return result

    def save_state(self, file_path: Path):
        """Save the current state of the predictor"""
        try:
            import pickle
            state = {
                "performance_history": self.performance_history,
                "prediction_cache": self.prediction_cache,
                "timestamp": datetime.datetime.now().isoformat()
            }

            with open(file_path, 'wb') as f:
                pickle.dump(state, f)

            logger.info(f"Forex predictor state saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving predictor state: {str(e)}")
            return False

    def load_state(self, file_path: Path):
        """Load a previously saved state"""
        try:
            import pickle
            with open(file_path, 'rb') as f:
                state = pickle.load(f)

            if "performance_history" in state:
                self.performance_history = state["performance_history"]

            if "prediction_cache" in state:
                self.prediction_cache = state["prediction_cache"]

            logger.info(f"Forex predictor state loaded from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading predictor state: {str(e)}")
            return False

# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from ai.prediction_engine import PredictionEngine
from ai.lstm_decision import LSTMDecisionMaker
from ai.minmax_search import MinMaxSearcher
from ai.rational_agent import RationalAgent
from ai.neat_algorithm import NEATAlgorithm
from ai.deap_framework import DEAPFramework
from ai.random_forest import RandomForestPredictor
from data.data_manager import DataManager
from utils.web_scraper import WebScraper
from utils.sentiment_analyzer import SentimentAnalyzer
from self_modification.engine import SelfModificationEngine
from betting.sports_predictor import SportsPredictor
from forecasting.forex_predictor import ForexPredictor
from core.system_integrator import SystemConfiguration

logger = logging.getLogger(__name__)

class SystemIntegrator:
    """Integrates all AI components for unified predictions with interactive capabilities"""

    def __init__(self, config_path: Path = Path("config/system.yaml")):
        # Load system configuration
        self.system_config = SystemConfiguration(config_path)

        # Initialize core components
        self.data_manager = DataManager()
        self.web_scraper = WebScraper()
        self.prediction_engine = PredictionEngine()
        self.lstm_decision = LSTMDecisionMaker(input_shape=(50, 10))
        self.minmax = MinMaxSearcher(max_depth=5)
        self.rational_agent = RationalAgent()
        self.self_modifier = SelfModificationEngine()

        # Initialize advanced AI components
        self.neat_algorithm = NEATAlgorithm()
        self.deap_framework = DEAPFramework()
        self.random_forest = RandomForestPredictor()
        self.sentiment_analyzer = SentimentAnalyzer()

        # Domain-specific predictors
        self.sports_predictor = SportsPredictor()
        self.forex_predictor = ForexPredictor()

        # Apply configuration settings
        self._apply_config()
        logger.info("Enhanced system integrator initialized")

        # System state
        self.current_mode = None
        self.available_sports = [
            "Soccer", "Basketball", "Tennis", "Rugby", "Cricket",
            "Volleyball", "MMA", "Boxing", "Horse Racing", "Golf",
            "Cycling", "Virtual Sports"
        ]
        self.available_forex_pairs = [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
            "AUD/USD", "USD/CAD", "NZD/USD"
        ]

    def _apply_config(self):
        """Apply configuration settings to components"""
        try:
            # Set debug mode
            if self.system_config.debug_mode:
                logging.getLogger().setLevel(logging.DEBUG)
                logger.debug("Debug mode enabled")

            # Configure components based on settings
            if self.system_config.use_external_apis:
                self.web_scraper.enable_external_apis()
                self.sentiment_analyzer.use_external_api = True
                logger.info("External APIs enabled")

            # Performance settings
            if self.system_config.use_parallel:
                logger.info(f"Parallel processing enabled with {self.system_config.max_threads} threads")

            # Apply AI-specific configurations
            if self.system_config.ai_config:
                logger.info("Applying AI-specific configurations")
        except Exception as e:
            logger.error(f"Error applying configurations: {str(e)}")

    def _load_config(self, config_path: Path):
        """Load system configuration from YAML file (legacy method, use system_config instead)"""
        logger.warning("_load_config is deprecated, use system_config instead")
        # This method is kept for backward compatibility
        pass

    def interactive_startup(self) -> None:
        """Interactive startup menu for prediction type selection"""
        print("\n===== Super AI Prediction System =====")
        print("1. Betting Predictions")
        print("2. Market/Forex Trends")

        while True:
            try:
                choice = input("\nDo you want predictions for betting or market/forex trends? (1/2): ")
                if choice == "1":
                    self.current_mode = "betting"
                    self.show_betting_menu()
                    break
                elif choice == "2":
                    self.current_mode = "forex"
                    self.show_forex_menu()
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except Exception as e:
                logger.error(f"Error in interactive startup: {str(e)}")
                print("An error occurred. Please try again.")

    def show_betting_menu(self) -> None:
        """Show betting sports selection menu"""
        print("\n===== Sports Selection =====")
        for i, sport in enumerate(self.available_sports, 1):
            print(f"{i}. {sport}")

        while True:
            try:
                choice = input("\nSelect a sport (1-12): ")
                idx = int(choice) - 1
                if 0 <= idx < len(self.available_sports):
                    sport = self.available_sports[idx]
                    self.generate_sports_prediction(sport)
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 12.")
            except ValueError:
                print("Please enter a valid number.")
            except Exception as e:
                logger.error(f"Error in betting menu: {str(e)}")
                print("An error occurred. Please try again.")

    def show_forex_menu(self) -> None:
        """Show forex currency pair selection menu"""
        print("\n===== Currency Pair Selection =====")
        for i, pair in enumerate(self.available_forex_pairs, 1):
            print(f"{i}. {pair}")

        while True:
            try:
                choice = input("\nSelect a currency pair (1-7): ")
                idx = int(choice) - 1
                if 0 <= idx < len(self.available_forex_pairs):
                    pair = self.available_forex_pairs[idx]
                    timeframe = input("Enter timeframe (e.g., 1h, 4h, 1d): ")
                    lot_size = input("Enter lot size (e.g., 0.01, 0.1, 1.0): ")
                    self.generate_forex_prediction(pair, timeframe, lot_size)
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 7.")
            except ValueError:
                print("Please enter a valid number.")
            except Exception as e:
                logger.error(f"Error in forex menu: {str(e)}")
                print("An error occurred. Please try again.")

    def generate_prediction(self, request_type: str, target: str) -> Dict[str, Any]:
        """Generate unified prediction using all AI components."""
        historical_data = self.data_manager.get_historical_data(request_type, target)
        online_data = self.web_scraper.get_realtime_data(request_type, target)
        sentiment_data = self.sentiment_analyzer.analyze_sentiment(target)

        # Combine all data sources
        input_data = {
            "historical": historical_data,
            "online": online_data,
            "sentiment": sentiment_data
        }

        # Generate predictions using different algorithms
        predictions = {
            "base": self.prediction_engine.predict(input_data),
            "lstm": self.lstm_decision.predict(input_data),
            "neat": self.neat_algorithm.predict(input_data),
            "random_forest": self.random_forest.predict(input_data)
        }

        # Use rational agent to combine predictions
        final_prediction = self.rational_agent.make_decision(predictions)

        return final_prediction

    def generate_sports_prediction(self, sport: str) -> Dict[str, Any]:
        """Generate sports betting predictions"""
        print(f"\n===== Generating {sport} Predictions =====")

        # Get user input for specific bet types
        match = input("Enter match or event (e.g., 'Manchester United vs Chelsea'): ")

        # Generate prediction using domain-specific predictor
        prediction = self.sports_predictor.predict(sport, match)

        # Display the prediction
        print("\n===== Prediction Results =====")
        print(f"Sport: {sport}")
        print(f"Match: {match}")

        if "match_result" in prediction:
            print(f"\nMatch Result (1X2):")
            print(f"Home Win (1): {prediction['match_result']['home_win']:.2f}")
            print(f"Draw (X): {prediction['match_result']['draw']:.2f}")
            print(f"Away Win (2): {prediction['match_result']['away_win']:.2f}")

        if "over_under" in prediction:
            print(f"\nOver/Under 2.5 Goals:")
            print(f"Over: {prediction['over_under']['over']:.2f}")
            print(f"Under: {prediction['over_under']['under']:.2f}")

        if "btts" in prediction:
            print(f"\nBoth Teams to Score:")
            print(f"Yes: {prediction['btts']['yes']:.2f}")
            print(f"No: {prediction['btts']['no']:.2f}")

        return prediction

    def generate_forex_prediction(self, pair: str, timeframe: str, lot_size: str) -> Dict[str, Any]:
        """Generate forex market predictions with verbose output"""
        print(f"\n===== Generating {pair} Predictions ({timeframe}) =====")
        print("Analyzing market data, news sentiment, and technical indicators...")

        # Generate prediction using domain-specific predictor
        prediction = self.forex_predictor.predict(pair, timeframe, lot_size)

        # Display the verbose prediction
        print("\n===== Forex Prediction Results =====")
        print(f"Currency Pair: {pair}")
        print(f"Timeframe: {timeframe}")
        print(f"Lot Size: {lot_size}")

        if "trend" in prediction:
            print(f"\nTrend Direction: {prediction['trend']['direction']}")
            print(f"Strength: {prediction['trend']['strength']}/10")
            print(f"Target: {prediction['trend']['target']}")
            print(f"Stop Loss: {prediction['trend']['stop_loss']}")

        if "analysis" in prediction:
            print("\nMarket Analysis:")
            if "usd_dynamics" in prediction["analysis"]:
                print(f"\nUSD Dynamics:")
                print(f"- {prediction['analysis']['usd_dynamics']}")

            if "geopolitical" in prediction["analysis"]:
                print(f"\nGeopolitical Factors:")
                print(f"- {prediction['analysis']['geopolitical']}")

            if "technical" in prediction["analysis"]:
                print(f"\nTechnical Analysis:")
                print(f"- {prediction['analysis']['technical']}")

            if "sentiment" in prediction["analysis"]:
                print(f"\nMarket Sentiment:")
                print(f"- {prediction['analysis']['sentiment']}")

        if "risk" in prediction:
            print(f"\nRisk Assessment:")
            print(f"Risk Level: {prediction['risk']['level']}/10")
            print(f"Recommended Position Size: {prediction['risk']['position_size']}")

        return prediction

    def update_models(self):
        """Update all AI models with latest data"""
        try:
            # Create checkpoint before update
            self.self_modifier.create_checkpoint("pre_model_update")

            # Update all models
            self.prediction_engine.update()
            self.lstm_decision.update()
            self.neat_algorithm.update()
            self.deap_framework.update()
            self.random_forest.update()

            # Update domain-specific predictors
            self.sports_predictor.update()
            self.forex_predictor.update()

            logger.info("All models updated successfully")
            return True
        except Exception as e:
            logger.error(f"Model update failed: {str(e)}")
            # Rollback to checkpoint in case of failure
            self.self_modifier.rollback_to_checkpoint()
            return False

    def self_modify(self, target_module: str = None):
        """Trigger self-modification of AI components"""
        try:
            logger.info(f"Starting self-modification process for {target_module or 'all modules'}")
            self.self_modifier.create_checkpoint("pre_self_modification")

            # Measure performance before modification
            performance_before = self.evaluate_performance()

            # Perform self-modification (in a real system, this would modify actual code)
            if target_module:
                # Modify specific module
                self.self_modifier.modify_code(target_module, "new_code_content")
            else:
                # Modify multiple modules based on performance analysis
                modules_to_modify = self.identify_modules_for_improvement()
                for module in modules_to_modify:
                    self.self_modifier.modify_code(module, "new_code_content")

            # Measure performance after modification
            performance_after = self.evaluate_performance()

            # Rollback if performance decreased
            if performance_after["overall"] < performance_before["overall"]:
                logger.warning("Performance decreased after self-modification, rolling back")
                self.self_modifier.rollback_to_checkpoint()
                return False

            # Adapt learning rate based on performance change
            performance_change = {
                key: performance_after.get(key, 0) - performance_before.get(key, 0)
                for key in performance_after
            }
            self.self_modifier.adapt_learning_rate(performance_change)

            logger.info("Self-modification completed successfully")
            return True
        except Exception as e:
            logger.error(f"Self-modification failed: {str(e)}")
            self.self_modifier.rollback_to_checkpoint()
            return False

    def evaluate_performance(self) -> Dict[str, float]:
        """Evaluate current system performance"""
        # In a real system, this would use actual metrics
        return {
            "accuracy": 0.85,
            "response_time": 0.95,
            "adaptation": 0.78,
            "overall": 0.86
        }

    def identify_modules_for_improvement(self) -> List[str]:
        """Identify modules that need improvement"""
        # In a real system, this would analyze actual performance data
        return [
            "ai.prediction_engine",
            "ai.lstm_decision",
            "betting.sports_predictor"
        ]

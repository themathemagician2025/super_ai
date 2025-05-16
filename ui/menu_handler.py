# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
from typing import Optional, Dict, Any, List
import re

logger = logging.getLogger(__name__)

class MenuHandler:
    """Handles all interactive menus with advanced prediction explanation and counterfactual reasoning"""

    def __init__(self):
        self.sports = [
            "Soccer", "Basketball", "Tennis", "Rugby",
            "Cricket", "Volleyball", "MMA", "Boxing",
            "Horse Racing", "Golf", "Cycling",
            "Virtual Soccer", "Virtual Horse Racing",
            "Virtual Cycling", "Virtual Tennis"
        ]

        self.forex_pairs = [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
            "AUD/USD", "USD/CAD", "NZD/USD"
        ]

        self.weather_locations = [
            "New York", "London", "Tokyo", "Sydney",
            "Berlin", "Paris", "Moscow", "Dubai"
        ]

        self.explanation_levels = ["simple", "detailed", "technical"]
        self.current_explanation_level = "detailed"
        self.show_counterfactuals = True
        self.sentiment_analysis = True

        self.math_models = {
            "riemann": "Riemann Hypothesis",
            "birch_swinnerton": "Birch and Swinnerton-Dyer Conjecture",
            "navier_stokes": "Navier–Stokes Equations",
            "yang_mills": "Yang–Mills Theory",
            "hodge": "Hodge Conjecture",
            "p_vs_np": "P vs NP Problem",
            "collatz": "Collatz Conjecture",
            "twin_prime": "Twin Prime Conjecture",
            "goldbach": "Goldbach Conjecture",
            "erdos_straus": "Erdős–Straus Conjecture"
        }
        self.active_math_models = ["riemann", "navier_stokes", "p_vs_np"]  # Default active models

        logger.info("Advanced menu handler initialized")

    def display_main_menu(self) -> str:
        """Display main prediction menu"""
        while True:
            print("\n=== Super AI Prediction System ===")
            print("1. Sports Betting Predictions")
            print("2. Forex Market Trends")
            print("3. Weather Forecasting")
            print("4. Cross-Domain Analysis")
            print("5. System Settings")
            print("6. View Prediction History")
            print("7. Model Information")
            print("8. Mathematical Models Configuration")
            print("9. Quit")

            choice = input("\nEnter your choice (1-9): ").strip()

            if choice == "1":
                return "betting"
            elif choice == "2":
                return "forex"
            elif choice == "3":
                return "weather"
            elif choice == "4":
                return "cross_domain"
            elif choice == "5":
                return "settings"
            elif choice == "6":
                return "history"
            elif choice == "7":
                return "model_info"
            elif choice == "8":
                self._configure_math_models()
                continue
            elif choice == "9":
                return "quit"
            else:
                print("Invalid choice. Please try again.")

    def display_sports_menu(self) -> Optional[str]:
        """Display sports selection menu"""
        while True:
            print("\n=== Available Sports ===")
            for i, sport in enumerate(self.sports, 1):
                print(f"{i}. {sport}")
            print(f"{len(self.sports)+1}. Back to main menu")

            try:
                choice = int(input(f"\nSelect sport (1-{len(self.sports)+1}): "))
                if 1 <= choice <= len(self.sports):
                    sport = self.sports[choice-1]
                    logger.info(f"Selected sport: {sport}")
                    return sport.lower()
                elif choice == len(self.sports)+1:
                    return None
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def display_forex_menu(self) -> Optional[str]:
        """Display forex pair selection menu"""
        while True:
            print("\n=== Available Currency Pairs ===")
            for i, pair in enumerate(self.forex_pairs, 1):
                print(f"{i}. {pair}")
            print(f"{len(self.forex_pairs)+1}. Back to main menu")

            try:
                choice = int(input(f"\nSelect pair (1-{len(self.forex_pairs)+1}): "))
                if 1 <= choice <= len(self.forex_pairs):
                    pair = self.forex_pairs[choice-1]
                    logger.info(f"Selected forex pair: {pair}")
                    return pair
                elif choice == len(self.forex_pairs)+1:
                    return None
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def display_weather_menu(self) -> Optional[str]:
        """Display weather location selection menu"""
        while True:
            print("\n=== Available Locations ===")
            for i, location in enumerate(self.weather_locations, 1):
                print(f"{i}. {location}")
            print(f"{len(self.weather_locations)+1}. Back to main menu")

            try:
                choice = int(input(f"\nSelect location (1-{len(self.weather_locations)+1}): "))
                if 1 <= choice <= len(self.weather_locations):
                    location = self.weather_locations[choice-1]
                    logger.info(f"Selected weather location: {location}")
                    return location.lower()
                elif choice == len(self.weather_locations)+1:
                    return None
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def display_settings_menu(self) -> Dict[str, Any]:
        """Display and update system settings"""
        while True:
            print("\n=== System Settings ===")
            print(f"1. Explanation Level: {self.current_explanation_level}")
            print(f"2. Show Counterfactuals: {'Yes' if self.show_counterfactuals else 'No'}")
            print(f"3. Sentiment Analysis: {'Enabled' if self.sentiment_analysis else 'Disabled'}")
            print("4. Update Model Version")
            print("5. Toggle API/Local Model")
            print("6. Save Settings")
            print("7. Back to Main Menu")

            choice = input("\nEnter your choice (1-7): ").strip()

            if choice == "1":
                self._change_explanation_level()
            elif choice == "2":
                self.show_counterfactuals = not self.show_counterfactuals
                print(f"Counterfactuals {'enabled' if self.show_counterfactuals else 'disabled'}")
            elif choice == "3":
                self.sentiment_analysis = not self.sentiment_analysis
                print(f"Sentiment analysis {'enabled' if self.sentiment_analysis else 'disabled'}")
            elif choice == "4":
                self._update_model_version()
            elif choice == "5":
                self._toggle_api_local()
            elif choice == "6":
                self._save_settings()
                print("Settings saved")
            elif choice == "7":
                return {
                    "explanation_level": self.current_explanation_level,
                    "show_counterfactuals": self.show_counterfactuals,
                    "sentiment_analysis": self.sentiment_analysis
                }
            else:
                print("Invalid choice. Please try again.")

    def _change_explanation_level(self):
        """Change the explanation detail level"""
        print("\n=== Explanation Levels ===")
        print("1. Simple - Basic prediction information")
        print("2. Detailed - Includes causal factors and confidence")
        print("3. Technical - Complete model information and analysis")

        try:
            choice = int(input("\nSelect level (1-3): "))
            if 1 <= choice <= 3:
                self.current_explanation_level = self.explanation_levels[choice-1]
                print(f"Explanation level set to: {self.current_explanation_level}")
            else:
                print("Invalid selection. Keeping current level.")
        except ValueError:
            print("Please enter a valid number.")

    def _update_model_version(self):
        """Update the model version"""
        print("\n=== Update Model Version ===")
        print("Current model version will be displayed from system integrator")
        print("Enter new version (format: x.y.z) or leave blank to cancel:")

        new_version = input("> ").strip()

        if not new_version:
            print("Version update cancelled")
            return

        # Basic validation - can be handled by the system integrator as well
        if not re.match(r'^\d+\.\d+\.\d+$', new_version):
            print("Invalid version format. Please use format: x.y.z (e.g., 1.2.3)")
            return

        # Return the new version for system integrator to handle
        return new_version

    def _toggle_api_local(self):
        """Toggle between API and local model"""
        print("\n=== Data Source Settings ===")
        print("1. Use API (External Data)")
        print("2. Use Local Model Only")
        print("3. Use API with Local Fallback")
        print("4. Cancel")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == "1":
            return {"use_real_data": True, "fallback_enabled": False}
        elif choice == "2":
            return {"use_real_data": False, "fallback_enabled": True}
        elif choice == "3":
            return {"use_real_data": True, "fallback_enabled": True}
        else:
            print("Operation cancelled")
            return None

    def _save_settings(self):
        """Save current settings to configuration"""
        # In a real implementation, this would write to a config file
        logger.info(f"Saving settings: explanation_level={self.current_explanation_level}, "
                   f"show_counterfactuals={self.show_counterfactuals}, "
                   f"sentiment_analysis={self.sentiment_analysis}")

    def _configure_math_models(self):
        """Configure which mathematical models to use in predictions"""
        while True:
            print("\n=== Mathematical Models Configuration ===")
            print("Currently active models:")
            for model_key in self.active_math_models:
                print(f"- {self.math_models[model_key]}")

            print("\nAvailable models:")
            for i, (key, name) in enumerate(self.math_models.items(), 1):
                status = "✓" if key in self.active_math_models else " "
                print(f"{i}. [{status}] {name}")
            print(f"{len(self.math_models) + 1}. Back to main menu")

            try:
                choice = int(input(f"\nToggle model (1-{len(self.math_models) + 1}): "))
                if choice == len(self.math_models) + 1:
                    break
                if 1 <= choice <= len(self.math_models):
                    model_key = list(self.math_models.keys())[choice - 1]
                    if model_key in self.active_math_models:
                        self.active_math_models.remove(model_key)
                    else:
                        self.active_math_models.append(model_key)
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Please enter a valid number.")

    def display_model_info(self, model_info: Dict[str, Any]):
        """Display model information"""
        print("\n=== Model Information ===")
        print(f"Model Version: {model_info.get('model_version', 'Unknown')}")
        print(f"Accuracy: {model_info.get('accuracy', 0)*100:.1f}%")

        # Display data source information
        data_source = "API (External Data)" if model_info.get('use_real_data', False) else "Local Model"
        fallback = "Enabled" if model_info.get('fallback_enabled', False) else "Disabled"
        api_status = "Available" if model_info.get('api_available', False) else "Not Available"

        print(f"Data Source: {data_source}")
        print(f"Fallback to Local: {fallback}")
        print(f"API Status: {api_status}")

        # Display supported sports
        if 'supported_sports' in model_info:
            print("\nSupported Sports:")
            for sport in model_info['supported_sports']:
                print(f"- {sport}")

        # Add error information if present
        if 'error' in model_info:
            print(f"\nWarning: {model_info['error']}")

        input("\nPress Enter to continue...")

    def display_predictions(self, predictions: Dict[str, Any]):
        """Display prediction results with explanations and counterfactuals"""
        print("\n=== Prediction Results ===")

        # Check for errors first
        if 'error' in predictions:
            print(f"\nError: {predictions['error']}")
            print("\nPrediction could not be completed. Please try again later.")
            input("\nPress Enter to continue...")
            return

        # Model information
        if 'model_version' in predictions:
            print(f"Model: v{predictions['model_version']}")
        if 'accuracy' in predictions:
            print(f"Model Accuracy: {predictions['accuracy']*100:.1f}%")
        if 'source' in predictions:
            print(f"Data Source: {predictions['source']}")

        # Main prediction
        print(f"\nOutcome: {predictions.get('outcome', 'Unknown')}")
        print(f"Confidence: {predictions.get('confidence', 0)*100:.1f}%")

        # Explanation based on current detail level
        if 'explanation' in predictions:
            print("\n--- Explanation ---")
            if self.current_explanation_level == 'simple':
                print(predictions['explanation'].get('simple', 'No simple explanation available'))
            elif self.current_explanation_level == 'detailed':
                print(predictions['explanation'].get('detailed', 'No detailed explanation available'))
                # Show causal factors in detailed view
                if 'causal_factors' in predictions:
                    print("\nCausal Factors:")
                    for factor, impact in predictions['causal_factors'].items():
                        print(f"- {factor}: {'↑↑' if impact > 0.5 else '↑' if impact > 0 else '↓' if impact > -0.5 else '↓↓'} ({impact:.2f})")
            else:  # technical
                print(predictions['explanation'].get('technical', 'No technical explanation available'))

        # Sentiment analysis if enabled
        if self.sentiment_analysis and 'sentiment' in predictions:
            print("\n--- Sentiment Analysis ---")
            print(f"Market Sentiment: {predictions['sentiment'].get('market', 'Neutral')}")
            print(f"News Sentiment: {predictions['sentiment'].get('news', 'Neutral')}")
            print(f"Social Media: {predictions['sentiment'].get('social', 'Neutral')}")

        # Counterfactuals if enabled
        if self.show_counterfactuals and 'counterfactuals' in predictions:
            print("\n--- What-If Scenarios ---")
            for cf in predictions['counterfactuals']:
                print(f"If {cf['condition']}: {cf['outcome']} (Confidence: {cf['confidence']*100:.1f}%)")

        # Mathematical model insights
        if 'math_insights' in predictions and self.current_explanation_level != 'simple':
            print("\n--- Mathematical Model Insights ---")
            for model_key in self.active_math_models:
                if model_key in predictions['math_insights']:
                    insight = predictions['math_insights'][model_key]
                    print(f"\n{self.math_models[model_key]}:")
                    print(f"- Confidence: {insight.get('confidence', 0)*100:.1f}%")
                    print(f"- Impact: {insight.get('impact', 'Unknown')}")
                    if self.current_explanation_level == 'technical':
                        print(f"- Technical Details: {insight.get('technical_details', 'Not available')}")

        # Alternative approaches
        if 'alternatives' in predictions:
            print("\n--- Alternative Approaches ---")
            for alt in predictions['alternatives']:
                print(f"- {alt['description']}: {alt['outcome']} (Confidence: {alt['confidence']*100:.1f}%)")

        input("\nPress Enter to continue...")

    def display_history(self, history_items: List[Dict[str, Any]]):
        """Display prediction history with performance tracking"""
        if not history_items:
            print("\nNo prediction history available.")
            input("\nPress Enter to continue...")
            return

        print("\n=== Prediction History ===")
        print(f"Total predictions: {len(history_items)}")

        # Calculate success rate
        if any('actual_outcome' in item for item in history_items):
            correct = sum(1 for item in history_items if item.get('outcome') == item.get('actual_outcome'))
            success_rate = correct / len(history_items) * 100
            print(f"Success rate: {success_rate:.1f}%")

        # Show recent predictions
        print("\nRecent predictions:")
        for i, item in enumerate(history_items[-5:], 1):
            print(f"{i}. {item.get('timestamp', 'Unknown date')}: {item.get('domain', 'Unknown')} - "
                 f"Predicted: {item.get('outcome', 'Unknown')}, "
                 f"Actual: {item.get('actual_outcome', 'Pending')}")

        # Allow viewing details of a specific prediction
        while True:
            try:
                choice = input("\nEnter number to view details (or 0 to return): ")
                if choice == "0":
                    break

                idx = int(choice) - 1
                if 0 <= idx < len(history_items[-5:]):
                    self.display_predictions(history_items[-(5-idx)])
                    break
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Please enter a valid number.")

        input("\nPress Enter to continue...")

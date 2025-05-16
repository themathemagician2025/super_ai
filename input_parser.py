# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Input Parser Module

Handles intent recognition and entity parsing from user input.
"""

import re
import json
from typing import Dict, List, Any, Tuple, Optional

class InputParser:
    def __init__(self):
        self.intents = {
            "predict": ["predict", "forecast", "will", "chance", "probability", "odds"],
            "analyze": ["analyze", "compare", "stats", "statistics", "performance"],
            "search": ["search", "find", "look up", "get"],
            "help": ["help", "how to", "guide", "documentation"]
        }

    def parse(self, user_input: str) -> Dict[str, Any]:
        """
        Parse user input to extract intent and entities.

        Args:
            user_input: The raw user input string

        Returns:
            Dict containing intent, entities, and other parsed information
        """
        # Convert to lowercase for easier matching
        text = user_input.lower()

        # Detect intent
        intent = self._detect_intent(text)

        # Extract entities
        entities = self._extract_entities(text)

        return {
            "raw_input": user_input,
            "intent": intent,
            "entities": entities,
            "timestamp": None  # Will be filled by orchestrator
        }

    def _detect_intent(self, text: str) -> str:
        """Detect the primary intent of the user query."""
        max_score = 0
        detected_intent = "unknown"

        for intent, keywords in self.intents.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > max_score:
                max_score = score
                detected_intent = intent

        return detected_intent

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract various entities from the text."""
        entities = {
            "sports_teams": self._extract_sports_teams(text),
            "players": self._extract_players(text),
            "dates": self._extract_dates(text),
            "currency_pairs": self._extract_currency_pairs(text)
        }

        return entities

    def _extract_sports_teams(self, text: str) -> List[str]:
        """Extract sports team names - stub implementation."""
        # To be replaced with real entity extraction
        return []

    def _extract_players(self, text: str) -> List[str]:
        """Extract player names - stub implementation."""
        # To be replaced with real entity extraction
        return []

    def _extract_dates(self, text: str) -> List[str]:
        """Extract date references - stub implementation."""
        # To be replaced with real entity extraction
        return []

    def _extract_currency_pairs(self, text: str) -> List[str]:
        """Extract currency pairs like EUR/USD."""
        # Simple regex for currency pairs
        pairs = re.findall(r'[A-Z]{3}/[A-Z]{3}', text.upper())
        return pairs

# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
API Processor Module

This module is responsible for processing natural language API requests and extracting
key information needed for predictions. It parses user queries to identify athletes,
sports, surfaces, tournaments, and the type of prediction request.
"""

import re
import logging
from typing import Dict, List, Any, Tuple, Optional

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_processor")

class APIProcessor:
    """
    Process natural language requests and convert them to structured queries
    for the Super AI prediction system.
    """

    def __init__(self):
        """Initialize the API processor."""
        # Patterns for matching different types of prediction requests
        self.patterns = {
            'match_prediction': [
                r"(?:who|which|predict).+(?:win|defeat|beat).+(?:between|vs\.?|against)\s+([a-zA-Z\s\.]+)\s+(?:and|vs\.?)\s+([a-zA-Z\s\.]+)",
                r"(?:will|can|could)\s+([a-zA-Z\s\.]+)\s+(?:win|defeat|beat|overcome)\s+([a-zA-Z\s\.]+)",
                r"([a-zA-Z\s\.]+)\s+(?:vs\.?|versus|against)\s+([a-zA-Z\s\.]+)\s+(?:who|which|predict|will).+(?:win)",
                r"(?:compare|matchup)\s+(?:between)?\s*([a-zA-Z\s\.]+)\s+(?:and|vs\.?)\s+([a-zA-Z\s\.]+)"
            ]
        }

        # Surface types to detect
        self.surfaces = {
            'tennis': ['clay', 'grass', 'hard', 'indoor', 'carpet'],
            'general': ['home', 'away', 'neutral']
        }

        # Sport types to detect
        self.sports = [
            'tennis', 'football', 'soccer', 'basketball', 'baseball',
            'hockey', 'golf', 'cricket', 'rugby'
        ]

        # Tournament keywords
        self.tournaments = {
            'tennis': ['australian open', 'french open', 'wimbledon', 'us open', 'atp', 'wta', 'masters'],
            'football': ['super bowl', 'nfl', 'playoff'],
            'soccer': ['world cup', 'champions league', 'premier league', 'la liga', 'bundesliga', 'serie a'],
            'basketball': ['nba', 'finals', 'playoff', 'ncaa']
        }

    def process_request(self, request_text: str) -> Dict[str, Any]:
        """
        Process a natural language request and extract structured information.

        Args:
            request_text: Natural language request text

        Returns:
            Dict with extracted parameters for API calls
        """
        request_text = request_text.lower().strip()
        logger.info(f"Processing request: {request_text}")

        # Check if it's a match prediction request
        match_data = self._extract_match_prediction(request_text)
        if match_data:
            return match_data

        # If no pattern matches, return a generic response
        return {
            'request_type': 'unknown',
            'original_text': request_text,
            'error': 'Could not determine the type of prediction request'
        }

    def _extract_match_prediction(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract information for a match prediction request.

        Args:
            text: Request text

        Returns:
            Dict with athlete names and context, or None if not a match prediction
        """
        for pattern in self.patterns['match_prediction']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                athlete1 = match.group(1).strip()
                athlete2 = match.group(2).strip()

                # Extract additional context
                sport = self._extract_sport(text)
                surface = self._extract_surface(text, sport)
                tournament = self._extract_tournament(text, sport)

                return {
                    'request_type': 'match_prediction',
                    'athlete1': athlete1,
                    'athlete2': athlete2,
                    'sport': sport,
                    'surface': surface,
                    'tournament': tournament,
                    'original_text': text
                }

        return None

    def _extract_sport(self, text: str) -> Optional[str]:
        """Extract the sport from the request text."""
        for sport in self.sports:
            if sport in text.lower():
                return sport

        # Try to infer from context
        if any(term in text.lower() for term in ['court', 'set', 'serve', 'deuce']):
            return 'tennis'
        elif any(term in text.lower() for term in ['touchdown', 'quarterback', 'nfl']):
            return 'football'
        elif any(term in text.lower() for term in ['goal', 'penalty', 'fifa']):
            return 'soccer'
        elif any(term in text.lower() for term in ['basket', 'dunk', 'nba']):
            return 'basketball'

        return None

    def _extract_surface(self, text: str, sport: Optional[str] = None) -> Optional[str]:
        """Extract the playing surface from the request text."""
        text_lower = text.lower()

        # Tennis-specific surfaces
        if sport == 'tennis':
            for surface in self.surfaces['tennis']:
                if surface in text_lower:
                    return surface

        # General surfaces
        for surface in self.surfaces['general']:
            if surface in text_lower:
                return surface

        return None

    def _extract_tournament(self, text: str, sport: Optional[str] = None) -> Optional[str]:
        """Extract the tournament from the request text."""
        text_lower = text.lower()

        if sport:
            tournament_list = self.tournaments.get(sport, [])
            for tournament in tournament_list:
                if tournament in text_lower:
                    return tournament

        # Try all tournaments if sport-specific search fails
        for sport_tournaments in self.tournaments.values():
            for tournament in sport_tournaments:
                if tournament in text_lower:
                    return tournament

        return None

def process_prediction_request(request_text: str) -> Dict[str, Any]:
    """
    Process a natural language prediction request.

    Args:
        request_text: Natural language request

    Returns:
        Dict with structured parameters for prediction
    """
    processor = APIProcessor()
    return processor.process_request(request_text)

# Example usage
if __name__ == "__main__":
    # Test with some example queries
    test_queries = [
        "Who will win between Djokovic and Alcaraz?",
        "Can Nadal beat Federer on clay?",
        "Predict the winner of LeBron James vs Kevin Durant",
        "Chiefs against 49ers, who's going to win?",
        "Will Manchester United defeat Arsenal at home?"
    ]

    processor = APIProcessor()

    for query in test_queries:
        result = processor.process_request(query)
        print(f"Query: {query}")
        print(f"Result: {result}")
        print("-" * 50)

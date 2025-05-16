# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Player Intelligence Engine (PIE)

This module is responsible for processing athlete profiles, analyzing trends over time,
and generating insights such as player readiness, matchup ratings, and injury probability.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("player_intelligence")

# Define constants
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_DIR = BASE_DIR / 'data'
PLAYERS_DIR = DATA_DIR / 'players'

# Ensure directories exist
PLAYERS_DIR.mkdir(parents=True, exist_ok=True)


class PlayerIntelligence:
    """
    Player Intelligence Engine that processes athlete profiles, tracks metrics over time,
    and generates insights for prediction models.
    """

    def __init__(self):
        """Initialize the Player Intelligence Engine."""
        self.sport_specific_weights = self._load_sport_weights()
        self.cached_profiles = {}  # Cache for loaded profiles
        self.last_cache_refresh = datetime.now()
        self.cache_ttl = timedelta(minutes=15)  # Refresh cache every 15 minutes

    def _load_sport_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Load sport-specific weightings for metrics.

        Returns:
            Dictionary of sports with their metric weights
        """
        # This would normally load from a configuration file
        # For now, we'll hardcode some example weights
        return {
            "soccer": {
                "speed_kmh": 0.7,
                "acceleration": 0.6,
                "xG": 0.9,
                "xA": 0.8,
                "pass_completion": 0.7,
                "work_rate_km_per_game": 0.5,
                "reaction_time_ms": 0.4,
                "injury_prone": 0.6
            },
            "basketball": {
                "PER": 0.9,
                "true_shooting": 0.8,
                "win_shares": 0.9,
                "defensive_rating": 0.7,
                "offensive_rating": 0.8,
                "VORP": 0.9,
                "vertical_jump": 0.6,
                "speed_kmh": 0.5
            },
            "tennis": {
                "serve_speed": 0.8,
                "first_serve_percentage": 0.7,
                "return_points_won": 0.8,
                "break_points_converted": 0.9,
                "aces_per_match": 0.6,
                "winners_per_match": 0.7,
                "unforced_errors": 0.7,
                "VO2_max": 0.8
            },
            "golf": {
                "driving_distance": 0.7,
                "driving_accuracy": 0.8,
                "greens_in_regulation": 0.8,
                "strokes_gained_total": 0.9,
                "strokes_gained_putting": 0.8,
                "strokes_gained_approach": 0.8,
                "sand_save_percentage": 0.6,
                "scrambling": 0.7
            },
            # Default weights for any sport
            "default": {
                "speed_kmh": 0.5,
                "reaction_time_ms": 0.5,
                "strength_rating": 0.5,
                "agility_score": 0.5,
                "injury_prone": 0.5
            }
        }

    def get_athlete_profile(self, athlete_id: str, refresh: bool = False) -> Dict[str, Any]:
        """
        Get a complete athlete profile with all metrics and historical data.

        Args:
            athlete_id: Unique identifier for the athlete
            refresh: Force refresh from disk if True

        Returns:
            Complete athlete profile dictionary
        """
        # Check cache first if not forcing refresh
        if not refresh and athlete_id in self.cached_profiles:
            cache_age = datetime.now() - self.last_cache_refresh
            if cache_age < self.cache_ttl:
                return self.cached_profiles[athlete_id]

        # Load from file
        profile_path = PLAYERS_DIR / f"{athlete_id}.json"

        if profile_path.exists():
            try:
                with open(profile_path, 'r') as f:
                    profile = json.load(f)

                # Update cache
                self.cached_profiles[athlete_id] = profile

                logger.info(f"Loaded athlete profile for {profile.get('name', athlete_id)}")
                return profile
            except Exception as e:
                logger.error(f"Error loading athlete profile: {e}")
                return {}
        else:
            logger.warning(f"Athlete profile not found for ID: {athlete_id}")
            return {}

    def save_athlete_profile(self, athlete_id: str, profile: Dict[str, Any]) -> bool:
        """
        Save an athlete profile to disk.

        Args:
            athlete_id: Unique identifier for the athlete
            profile: Complete athlete profile dictionary

        Returns:
            True if successful, False otherwise
        """
        # Add last_updated timestamp
        profile['last_updated'] = datetime.now().isoformat()

        # Save to file
        profile_path = PLAYERS_DIR / f"{athlete_id}.json"

        try:
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)

            # Update cache
            self.cached_profiles[athlete_id] = profile
            self.last_cache_refresh = datetime.now()

            logger.info(f"Saved athlete profile for {profile.get('name', athlete_id)}")
            return True
        except Exception as e:
            logger.error(f"Error saving athlete profile: {e}")
            return False

    def update_athlete_metrics(self, athlete_id: str, new_metrics: Dict[str, Any],
                              source: str = "scraper") -> bool:
        """
        Update specific metrics for an athlete.

        Args:
            athlete_id: Unique identifier for the athlete
            new_metrics: New metrics to add or update
            source: Source of the new metrics

        Returns:
            True if successful, False otherwise
        """
        profile = self.get_athlete_profile(athlete_id)

        if not profile:
            logger.warning(f"Cannot update metrics for non-existent profile: {athlete_id}")
            return False

        # Get current metrics
        metrics = profile.get('metrics', {})
        history = profile.get('metrics_history', [])

        # Store current metrics to history
        if metrics:
            history_entry = {
                'date': profile.get('last_updated', datetime.now().isoformat()),
                'metrics': metrics.copy(),
                'source': profile.get('last_source', 'unknown')
            }
            history.append(history_entry)

            # Keep only the last 20 history entries
            if len(history) > 20:
                history = history[-20:]

        # Update with new metrics
        metrics.update(new_metrics)
        profile['metrics'] = metrics
        profile['metrics_history'] = history
        profile['last_source'] = source

        # Save updated profile
        return self.save_athlete_profile(athlete_id, profile)

    def get_metric_trends(self, athlete_id: str, metric_names: List[str],
                         period: int = 5) -> Dict[str, List[float]]:
        """
        Calculate trends for specific metrics over a period.

        Args:
            athlete_id: Unique identifier for the athlete
            metric_names: List of metric names to analyze
            period: Number of historical entries to consider

        Returns:
            Dictionary of metric names with their trend values
        """
        profile = self.get_athlete_profile(athlete_id)

        if not profile:
            return {}

        history = profile.get('metrics_history', [])
        current_metrics = profile.get('metrics', {})

        # Include current metrics in analysis
        history_with_current = history[-period:] if period < len(history) else history
        if current_metrics:
            current_entry = {
                'date': profile.get('last_updated', datetime.now().isoformat()),
                'metrics': current_metrics
            }
            history_with_current = history_with_current + [current_entry]

        trends = {}

        for metric in metric_names:
            values = []
            for entry in history_with_current:
                entry_metrics = entry.get('metrics', {})
                if metric in entry_metrics:
                    values.append(entry_metrics[metric])

            if len(values) > 1:
                # Calculate trend as percentage change from first to last
                first_val = values[0]
                last_val = values[-1]

                if first_val != 0:
                    trend = (last_val - first_val) / first_val * 100
                else:
                    trend = 0

                # Also calculate slope for more nuanced trend
                if len(values) >= 3:
                    x = np.arange(len(values))
                    y = np.array(values)
                    slope, _ = np.polyfit(x, y, 1)
                else:
                    slope = 0

                trends[metric] = {
                    'percent_change': trend,
                    'slope': slope,
                    'values': values
                }

        return trends

    def calculate_readiness_score(self, athlete_id: str) -> float:
        """
        Calculate a player readiness score based on recent metrics and trends.

        Args:
            athlete_id: Unique identifier for the athlete

        Returns:
            Readiness score between 0 and 100
        """
        profile = self.get_athlete_profile(athlete_id)

        if not profile:
            return 0.0

        # Get key metrics for readiness
        metrics = profile.get('metrics', {})
        sport = profile.get('sport', 'default').lower()

        # Default readiness factors
        fatigue = metrics.get('fatigue_level', 50)
        injury_status = metrics.get('injury_status', 'healthy')
        recent_form = metrics.get('recent_form', 0.5) * 100  # Convert to 0-100 scale

        # Convert injury status to numeric value
        injury_impact = 0
        if injury_status.lower() == 'healthy':
            injury_impact = 0
        elif injury_status.lower() == 'minor':
            injury_impact = 30
        elif injury_status.lower() == 'moderate':
            injury_impact = 60
        elif injury_status.lower() == 'severe':
            injury_impact = 90

        # Calculate raw score (higher is better)
        # Fatigue is 0-100 where lower is better
        raw_score = (100 - fatigue) * 0.4 + (100 - injury_impact) * 0.4 + recent_form * 0.2

        # Adjust based on sport-specific factors
        if sport == 'soccer':
            # Check for key soccer metrics
            work_rate = metrics.get('work_rate_km_per_game', 0)
            if 'work_rate_km_per_game' in metrics:
                ideal_workrate = 10.5  # Example ideal workrate
                workrate_factor = min(1, work_rate / ideal_workrate) * 10
                raw_score += workrate_factor

        elif sport == 'basketball':
            # Basketball-specific adjustments
            minutes_per_game = metrics.get('minutes_per_game', 0)
            if minutes_per_game > 35:  # High minutes may indicate fatigue
                raw_score -= 5

        # Normalize to 0-100 scale
        final_score = max(0, min(100, raw_score))

        return round(final_score, 1)

    def calculate_injury_probability(self, athlete_id: str) -> float:
        """
        Calculate probability of injury based on metrics and history.

        Args:
            athlete_id: Unique identifier for the athlete

        Returns:
            Injury probability as percentage (0-100)
        """
        profile = self.get_athlete_profile(athlete_id)

        if not profile:
            return 0.0

        metrics = profile.get('metrics', {})
        injury_history = profile.get('injury_history', [])
        sport = profile.get('sport', 'default').lower()

        # Base risk factors
        age = metrics.get('age', 25)
        fatigue = metrics.get('fatigue_level', 50) / 100  # Normalize to 0-1
        injury_prone = metrics.get('injury_prone', 0.5)  # 0-1 scale

        # Age risk factor (peaks around 33-35)
        age_factor = 0.3 * (1 - abs(age - 34) / 14)  # 0.3 at age 34, decreases away from peak

        # Recent injury history factor
        recent_injuries = len(injury_history)
        recency_factor = 0

        for injury in injury_history:
            # Check if injury was in last year
            injury_date = datetime.fromisoformat(injury.get('date', '2000-01-01'))
            days_since = (datetime.now() - injury_date).days

            if days_since < 365:
                severity = injury.get('severity', 'minor')

                if severity == 'severe':
                    recency_factor += 0.2
                elif severity == 'moderate':
                    recency_factor += 0.1
                else:
                    recency_factor += 0.05

        # Cap recency factor
        recency_factor = min(0.5, recency_factor)

        # Sport-specific factors
        sport_factor = 0

        if sport == 'football' or sport == 'rugby':
            sport_factor = 0.15  # Higher risk for contact sports
        elif sport == 'soccer':
            sport_factor = 0.1
        elif sport == 'tennis':
            sport_factor = 0.08

        # Calculate base probability
        base_probability = (
            fatigue * 0.3 +
            injury_prone * 0.25 +
            age_factor +
            recency_factor +
            sport_factor
        )

        # Convert to percentage and cap at 95%
        final_probability = min(0.95, base_probability) * 100

        return round(final_probability, 1)

    def calculate_matchup_rating(self, athlete1_id: str, athlete2_id: str,
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate a matchup rating between two athletes based on their metrics.

        Args:
            athlete1_id: ID of first athlete
            athlete2_id: ID of second athlete
            context: Optional context like surface, venue, etc.

        Returns:
            Dictionary with matchup analysis
        """
        athlete1 = self.get_athlete_profile(athlete1_id)
        athlete2 = self.get_athlete_profile(athlete2_id)

        if not athlete1 or not athlete2:
            logger.warning("One or both athlete profiles not found")
            return {}

        # Ensure athletes are from the same sport
        if athlete1.get('sport', '').lower() != athlete2.get('sport', '').lower():
            logger.warning(f"Athletes are from different sports: {athlete1.get('sport')} vs {athlete2.get('sport')}")
            return {
                'error': 'Cannot compare athletes from different sports',
                'athlete1_sport': athlete1.get('sport'),
                'athlete2_sport': athlete2.get('sport')
            }

        sport = athlete1.get('sport', 'default').lower()
        metrics1 = athlete1.get('metrics', {})
        metrics2 = athlete2.get('metrics', {})

        # Get sport-specific weights
        weights = self.sport_specific_weights.get(sport, self.sport_specific_weights['default'])

        # Calculate weighted scores for comparable metrics
        score1 = 0
        score2 = 0
        total_weight = 0

        comparisons = {}

        for metric, weight in weights.items():
            if metric in metrics1 and metric in metrics2:
                val1 = metrics1[metric]
                val2 = metrics2[metric]

                # Higher is better for most metrics
                better_higher = True

                # Exception list for metrics where lower is better
                if metric in ['reaction_time_ms', 'error_rate', 'injury_prone', 'fatigue_level']:
                    better_higher = False

                # Calculate normalized advantage
                if val1 == val2:
                    advantage = 0
                elif better_higher:
                    # If metric is "better" when higher
                    max_val = max(val1, val2)
                    min_val = min(val1, val2)
                    if max_val == 0:
                        advantage = 0
                    else:
                        advantage = (val1 - val2) / max_val
                else:
                    # If metric is "better" when lower
                    max_val = max(val1, val2)
                    min_val = min(val1, val2)
                    if max_val == 0:
                        advantage = 0
                    else:
                        advantage = (val2 - val1) / max_val

                # Add to weighted scores
                score1 += advantage * weight if advantage > 0 else 0
                score2 += -advantage * weight if advantage < 0 else 0
                total_weight += weight

                # Store comparison details
                comparisons[metric] = {
                    'athlete1_value': val1,
                    'athlete2_value': val2,
                    'advantage': advantage,
                    'weight': weight
                }

        # Normalize scores based on total weight
        if total_weight > 0:
            score1 = score1 / total_weight
            score2 = score2 / total_weight

        # Calculate overall advantage
        if score1 > score2:
            advantage = 'athlete1'
            advantage_score = score1
        elif score2 > score1:
            advantage = 'athlete2'
            advantage_score = score2
        else:
            advantage = 'neutral'
            advantage_score = 0

        # Context specific adjustments
        context_factors = []

        if context:
            surface = context.get('surface')
            if surface and sport == 'tennis':
                surface_pref1 = athlete1.get('surface_preference', {}).get(surface, 0.5)
                surface_pref2 = athlete2.get('surface_preference', {}).get(surface, 0.5)

                if abs(surface_pref1 - surface_pref2) > 0.1:
                    if surface_pref1 > surface_pref2:
                        context_factors.append({
                            'factor': 'surface',
                            'advantage': 'athlete1',
                            'description': f"{athlete1.get('name')} performs better on {surface}"
                        })
                    else:
                        context_factors.append({
                            'factor': 'surface',
                            'advantage': 'athlete2',
                            'description': f"{athlete2.get('name')} performs better on {surface}"
                        })

        # Readiness comparison
        readiness1 = self.calculate_readiness_score(athlete1_id)
        readiness2 = self.calculate_readiness_score(athlete2_id)

        if abs(readiness1 - readiness2) > 10:
            if readiness1 > readiness2:
                context_factors.append({
                    'factor': 'readiness',
                    'advantage': 'athlete1',
                    'description': f"{athlete1.get('name')} has better current form/readiness"
                })
            else:
                context_factors.append({
                    'factor': 'readiness',
                    'advantage': 'athlete2',
                    'description': f"{athlete2.get('name')} has better current form/readiness"
                })

        # Final matchup summary
        return {
            'athlete1': {
                'id': athlete1_id,
                'name': athlete1.get('name'),
                'score': round(score1, 2),
                'readiness': readiness1
            },
            'athlete2': {
                'id': athlete2_id,
                'name': athlete2.get('name'),
                'score': round(score2, 2),
                'readiness': readiness2
            },
            'advantage': advantage,
            'advantage_score': round(advantage_score, 2),
            'metric_comparisons': comparisons,
            'context_factors': context_factors,
            'sport': sport
        }

    def merge_athlete_data(self, athlete_id: str, data_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge data from multiple sources into a single athlete profile.

        Args:
            athlete_id: Athlete's unique identifier
            data_sources: List of data dictionaries from different sources

        Returns:
            Complete merged athlete profile
        """
        # Get existing profile or create new one
        profile = self.get_athlete_profile(athlete_id)
        if not profile:
            profile = {
                'id': athlete_id,
                'metrics': {},
                'metrics_history': [],
                'data_sources': []
            }

        current_metrics = profile.get('metrics', {})
        current_sources = profile.get('data_sources', [])

        # Process each data source
        for source_data in data_sources:
            source_name = source_data.get('source', 'unknown')
            source_metrics = source_data.get('metrics', {})
            reliability = source_data.get('reliability', 0.5)  # Source reliability score

            # Add to data sources if new
            if source_name not in current_sources:
                current_sources.append(source_name)

            # Merge metrics with reliability weighting
            for metric, value in source_metrics.items():
                # If metric already exists, use weighted average based on reliability
                if metric in current_metrics:
                    # Default reliability for existing data
                    existing_reliability = 0.7

                    # Calculate weighted average
                    weighted_value = (
                        current_metrics[metric] * existing_reliability +
                        value * reliability
                    ) / (existing_reliability + reliability)

                    current_metrics[metric] = weighted_value
                else:
                    # New metric, just add it
                    current_metrics[metric] = value

        # Update the profile
        profile['metrics'] = current_metrics
        profile['data_sources'] = current_sources
        profile['last_merged'] = datetime.now().isoformat()

        # Save the updated profile
        self.save_athlete_profile(athlete_id, profile)

        return profile

    def extract_sport_relevant_metrics(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only the metrics relevant to an athlete's sport.

        Args:
            profile: Complete athlete profile

        Returns:
            Dictionary with only relevant metrics
        """
        sport = profile.get('sport', 'default').lower()
        metrics = profile.get('metrics', {})

        # Get weights for this sport
        weights = self.sport_specific_weights.get(sport, self.sport_specific_weights['default'])

        # Extract relevant metrics
        relevant_metrics = {}
        for metric, value in metrics.items():
            if metric in weights:
                relevant_metrics[metric] = value

        return {
            'id': profile.get('id'),
            'name': profile.get('name'),
            'sport': sport,
            'relevant_metrics': relevant_metrics
        }


# Create a singleton instance
player_intelligence = PlayerIntelligence()

# Example usage
if __name__ == "__main__":
    # Example athlete profile
    sample_profile = {
        "id": "messi_10",
        "name": "Lionel Messi",
        "sport": "Soccer",
        "team": "Inter Miami",
        "position": "Forward",
        "metrics": {
            "speed_kmh": 33.7,
            "xG": 0.87,
            "reaction_time_ms": 245,
            "work_rate_km_per_game": 9.8,
            "pass_completion": 88.5,
            "assists_per_game": 0.72,
            "shots_on_target": 3.2
        },
        "injury_history": [
            {
                "date": "2023-10-15",
                "type": "Hamstring",
                "severity": "minor",
                "days_out": 14
            }
        ],
        "last_updated": "2024-04-24"
    }

    # Save the sample profile
    player_intelligence.save_athlete_profile("messi_10", sample_profile)

    # Demonstrate loading the profile
    loaded_profile = player_intelligence.get_athlete_profile("messi_10")
    print(f"Loaded profile for: {loaded_profile.get('name')}")

    # Calculate readiness score
    readiness = player_intelligence.calculate_readiness_score("messi_10")
    print(f"Readiness score: {readiness}")

    # Calculate injury probability
    injury_prob = player_intelligence.calculate_injury_probability("messi_10")
    print(f"Injury probability: {injury_prob}%")

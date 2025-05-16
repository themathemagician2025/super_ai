# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Athlete Metrics Scraper

This module is responsible for scraping athlete metrics from various online sources
and APIs, including sports websites, wearable devices, and other data providers.
"""

import os
import json
import time
import random
import logging
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("athlete_scraper")

# Constants
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_DIR = BASE_DIR / 'data'
CACHE_DIR = DATA_DIR / 'cache' / 'scrapers'
PLAYERS_DIR = DATA_DIR / 'players'

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PLAYERS_DIR.mkdir(parents=True, exist_ok=True)

class AthleteMetricsScraper:
    """
    Scrapes athlete metrics from various online sources and APIs.
    """

    def __init__(self, use_cache: bool = True, cache_ttl: int = 86400):
        """
        Initialize the athlete metrics scraper.

        Args:
            use_cache: Whether to use cached data
            cache_ttl: Cache time-to-live in seconds (default: 24 hours)
        """
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # API keys would be stored securely in environment variables
        self.api_keys = {
            'sportradar': os.environ.get('SPORTRADAR_API_KEY', ''),
            'transfermarkt': os.environ.get('TRANSFERMARKT_API_KEY', ''),
            'whoscored': os.environ.get('WHOSCORED_API_KEY', ''),
            'fbref': os.environ.get('FBREF_API_KEY', '')
        }

        # Initialize scrapers for different sources
        self.scrapers = {
            'soccer': {
                'fbref': self._scrape_fbref,
                'transfermarkt': self._scrape_transfermarkt,
                'whoscored': self._scrape_whoscored
            },
            'basketball': {
                'nba': self._scrape_nba,
                'basketball_reference': self._scrape_basketball_reference
            },
            'tennis': {
                'tennis_abstract': self._scrape_tennis_abstract,
                'atp': self._scrape_atp,
                'wta': self._scrape_wta
            },
            'football': {
                'nfl': self._scrape_nfl,
                'pro_football_reference': self._scrape_pro_football_reference
            }
        }

        # Define source reliability scores (0-1)
        self.source_reliability = {
            'fbref': 0.9,
            'transfermarkt': 0.8,
            'whoscored': 0.85,
            'nba': 0.9,
            'atp': 0.95,
            'wta': 0.95,
            'nfl': 0.9,
            'wearable_data': 0.95,
            'crowd_sourced': 0.6
        }

    def scrape_athlete(self, name: str, sport: str, sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Scrape metrics for a specific athlete from multiple sources.

        Args:
            name: Athlete's name
            sport: Sport name (soccer, basketball, etc.)
            sources: Specific sources to scrape (if None, use all available)

        Returns:
            Dictionary with merged metrics from all sources
        """
        sport = sport.lower()
        logger.info(f"Scraping metrics for {name} ({sport})")

        if sport not in self.scrapers:
            logger.warning(f"No scrapers available for sport: {sport}")
            return {}

        # Determine which sources to use
        available_sources = list(self.scrapers[sport].keys())
        if sources:
            # Use intersection of requested sources and available sources
            sources_to_use = [s for s in sources if s in available_sources]
        else:
            sources_to_use = available_sources

        if not sources_to_use:
            logger.warning(f"No valid sources to scrape for {name} ({sport})")
            return {}

        # Generate a normalized athlete ID
        athlete_id = self._generate_athlete_id(name, sport)

        # Check cache first if enabled
        if self.use_cache:
            cached_data = self._get_cached_data(athlete_id)
            if cached_data:
                logger.info(f"Using cached data for {name}")
                return cached_data

        # Scrape data from each source
        all_metrics = []

        for source in sources_to_use:
            try:
                logger.info(f"Scraping {source} for {name}")
                scraper_func = self.scrapers[sport][source]

                # Execute the scraper
                metrics = scraper_func(name)

                if metrics:
                    # Add source metadata
                    source_data = {
                        'source': source,
                        'metrics': metrics,
                        'scraped_at': datetime.now().isoformat(),
                        'reliability': self.source_reliability.get(source, 0.7)
                    }
                    all_metrics.append(source_data)
                    logger.info(f"Successfully scraped {len(metrics)} metrics from {source}")
                else:
                    logger.warning(f"No data found on {source} for {name}")
            except Exception as e:
                logger.error(f"Error scraping {source} for {name}: {str(e)}")

        # Merge all source data
        if all_metrics:
            # Cache the raw source data
            self._cache_data(athlete_id, all_metrics)

            # Create a merged profile
            merged_profile = self._create_merged_profile(name, sport, athlete_id, all_metrics)
            return merged_profile
        else:
            logger.warning(f"No metrics found for {name} from any source")
            return {}

    def _create_merged_profile(self, name: str, sport: str, athlete_id: str,
                              source_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a merged athlete profile from multiple data sources."""
        # Create base profile
        profile = {
            'id': athlete_id,
            'name': name,
            'sport': sport,
            'metrics': {},
            'data_sources': [],
            'last_updated': datetime.now().isoformat()
        }

        # Extract team, position, etc. from sources
        for source_data in source_metrics:
            metrics = source_data.get('metrics', {})

            # Copy non-metric metadata to profile
            if 'team' in metrics and 'team' not in profile:
                profile['team'] = metrics.pop('team')

            if 'position' in metrics and 'position' not in profile:
                profile['position'] = metrics.pop('position')

            if 'age' in metrics and 'age' not in profile:
                profile['age'] = metrics.pop('age')

            if 'nationality' in metrics and 'nationality' not in profile:
                profile['nationality'] = metrics.pop('nationality')

            # Add source to the list
            source_name = source_data.get('source')
            if source_name and source_name not in profile['data_sources']:
                profile['data_sources'].append(source_name)

        # Calculate merged metrics using weighted averaging
        merged_metrics = {}

        # First pass: collect all unique metrics
        all_metrics = set()
        for source_data in source_metrics:
            metrics = source_data.get('metrics', {})
            all_metrics.update(metrics.keys())

        # Second pass: merge with reliability weighting
        for metric in all_metrics:
            values = []
            weights = []

            for source_data in source_metrics:
                source_metrics = source_data.get('metrics', {})
                reliability = source_data.get('reliability', 0.7)

                if metric in source_metrics:
                    value = source_metrics[metric]
                    if isinstance(value, (int, float)):
                        values.append(value)
                        weights.append(reliability)

            if values:
                # Calculate weighted average if we have values
                if len(values) > 1:
                    # Weighted average
                    merged_metrics[metric] = sum(v * w for v, w in zip(values, weights)) / sum(weights)
                else:
                    # Just one value
                    merged_metrics[metric] = values[0]

        profile['metrics'] = merged_metrics

        return profile

    def _generate_athlete_id(self, name: str, sport: str) -> str:
        """Generate a normalized athlete ID from name and sport."""
        # Remove spaces and special characters, convert to lowercase
        normalized_name = ''.join(c.lower() for c in name if c.isalnum())
        return f"{normalized_name}_{sport.lower()}"

    def _get_cached_data(self, athlete_id: str) -> Optional[Dict[str, Any]]:
        """Get cached data for an athlete if valid."""
        cache_file = CACHE_DIR / f"{athlete_id}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)

            # Check if cache is still valid
            cached_time = datetime.fromisoformat(cached_data.get('cached_at', '2000-01-01'))
            age = (datetime.now() - cached_time).total_seconds()

            if age < self.cache_ttl:
                return cached_data.get('data')

        except Exception as e:
            logger.error(f"Error reading cache file: {str(e)}")

        return None

    def _cache_data(self, athlete_id: str, data: Any) -> None:
        """Cache data for an athlete."""
        cache_file = CACHE_DIR / f"{athlete_id}.json"

        try:
            cache_data = {
                'athlete_id': athlete_id,
                'cached_at': datetime.now().isoformat(),
                'data': data
            }

            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            logger.info(f"Cached data for {athlete_id}")
        except Exception as e:
            logger.error(f"Error caching data: {str(e)}")

    def fetch_wearable_data(self, athlete_id: str, device_type: str,
                          api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch data from wearable devices (requires API access).

        Args:
            athlete_id: ID of the athlete
            device_type: Type of wearable device (e.g., 'whoop', 'garmin', 'fitbit')
            api_key: Optional API key for authentication

        Returns:
            Dictionary of wearable metrics
        """
        logger.info(f"Fetching wearable data for {athlete_id} from {device_type}")

        # In a real implementation, this would call the appropriate wearable API
        # For now, we'll simulate the data

        if device_type == 'whoop':
            data = self._simulate_whoop_data()
        elif device_type == 'garmin':
            data = self._simulate_garmin_data()
        elif device_type == 'fitbit':
            data = self._simulate_fitbit_data()
        else:
            logger.warning(f"Unknown wearable device type: {device_type}")
            return {}

        # Add metadata
        return {
            'source': f"wearable_{device_type}",
            'metrics': data,
            'scraped_at': datetime.now().isoformat(),
            'reliability': self.source_reliability.get('wearable_data', 0.9)
        }

    # Simulation functions for development/testing
    def _simulate_whoop_data(self) -> Dict[str, Any]:
        """Simulate WHOOP wearable data."""
        return {
            'recovery_score': random.randint(33, 100),
            'resting_heart_rate': random.randint(40, 70),
            'hrv_ms': random.randint(20, 100),
            'respiratory_rate': random.uniform(14.0, 18.0),
            'sleep_performance': random.randint(60, 95),
            'strain_score': random.randint(10, 20),
            'calories_burned': random.randint(2200, 3500)
        }

    def _simulate_garmin_data(self) -> Dict[str, Any]:
        """Simulate Garmin wearable data."""
        return {
            'vo2_max': random.randint(40, 70),
            'resting_heart_rate': random.randint(40, 70),
            'stress_level': random.randint(20, 80),
            'steps': random.randint(5000, 15000),
            'active_calories': random.randint(500, 2000),
            'training_load': random.randint(100, 500),
            'recovery_time_hours': random.randint(12, 72)
        }

    def _simulate_fitbit_data(self) -> Dict[str, Any]:
        """Simulate Fitbit wearable data."""
        return {
            'resting_heart_rate': random.randint(40, 70),
            'steps': random.randint(5000, 15000),
            'active_minutes': random.randint(30, 180),
            'sleep_efficiency': random.randint(70, 95),
            'calories_burned': random.randint(1800, 3200)
        }

    # Individual source scrapers
    def _scrape_fbref(self, player_name: str) -> Dict[str, Any]:
        """Scrape player stats from FBref."""
        # In a real implementation, this would:
        # 1. Search for the player
        # 2. Navigate to their page
        # 3. Extract statistics

        # For demonstration, we'll simulate the data
        metrics = {
            'position': random.choice(['Forward', 'Midfielder', 'Defender', 'Goalkeeper']),
            'team': f"Team {random.choice(['A', 'B', 'C', 'D', 'E'])}",
            'goals_per_90': round(random.uniform(0.1, 0.8), 2),
            'assists_per_90': round(random.uniform(0.1, 0.6), 2),
            'xG_per_90': round(random.uniform(0.1, 0.9), 2),
            'xA_per_90': round(random.uniform(0.1, 0.7), 2),
            'pass_completion': round(random.uniform(65, 95), 1),
            'progressive_passes': round(random.uniform(2, 12), 1),
            'shot_creating_actions': round(random.uniform(1, 5), 1),
            'tackles_per_90': round(random.uniform(0.5, 3.5), 1),
            'interceptions_per_90': round(random.uniform(0.2, 2.5), 1),
            'pressures_per_90': round(random.uniform(10, 30), 1)
        }

        # Add position-specific metrics
        if metrics['position'] == 'Forward':
            metrics.update({
                'shots_per_90': round(random.uniform(2, 4.5), 1),
                'shot_on_target_pct': round(random.uniform(30, 60), 1)
            })
        elif metrics['position'] == 'Midfielder':
            metrics.update({
                'progressive_carries': round(random.uniform(3, 8), 1),
                'distance_covered_km': round(random.uniform(10, 13), 1)
            })
        elif metrics['position'] == 'Defender':
            metrics.update({
                'blocks_per_90': round(random.uniform(1, 3), 1),
                'aerials_won_pct': round(random.uniform(50, 80), 1)
            })

        return metrics

    def _scrape_transfermarkt(self, player_name: str) -> Dict[str, Any]:
        """Scrape player data from Transfermarkt."""
        # Simulated data
        metrics = {
            'market_value_millions': round(random.uniform(1, 100), 1),
            'age': random.randint(17, 38),
            'height_cm': random.randint(165, 200),
            'nationality': random.choice(['England', 'Spain', 'France', 'Germany', 'Italy', 'Brazil']),
            'games_played': random.randint(10, 50),
            'goals_scored': random.randint(0, 30),
            'assists': random.randint(0, 20),
            'yellow_cards': random.randint(0, 10),
            'red_cards': random.randint(0, 2),
            'minutes_played': random.randint(900, 4500)
        }
        return metrics

    def _scrape_whoscored(self, player_name: str) -> Dict[str, Any]:
        """Scrape player data from WhoScored."""
        # Simulated data
        metrics = {
            'rating': round(random.uniform(6.0, 8.5), 1),
            'motm_awards': random.randint(0, 8),
            'pass_success': round(random.uniform(70, 92), 1),
            'dribbles_per_game': round(random.uniform(0.5, 4.0), 1),
            'aerial_duels_won': round(random.uniform(0.5, 5.0), 1),
            'tackles_per_game': round(random.uniform(0.5, 4.0), 1),
            'interceptions_per_game': round(random.uniform(0.2, 3.0), 1),
            'fouls_per_game': round(random.uniform(0.5, 2.5), 1),
            'offsides_per_game': round(random.uniform(0.1, 1.5), 1),
            'shots_per_game': round(random.uniform(0.5, 4.0), 1)
        }
        return metrics

    def _scrape_nba(self, player_name: str) -> Dict[str, Any]:
        """Scrape player data from NBA.com."""
        # Simulated data
        metrics = {
            'team': f"Team {random.choice(['A', 'B', 'C', 'D', 'E'])}",
            'position': random.choice(['PG', 'SG', 'SF', 'PF', 'C']),
            'height_cm': random.randint(180, 220),
            'weight_kg': random.randint(75, 120),
            'ppg': round(random.uniform(5, 30), 1),
            'rpg': round(random.uniform(2, 15), 1),
            'apg': round(random.uniform(1, 10), 1),
            'spg': round(random.uniform(0.5, 3), 1),
            'bpg': round(random.uniform(0.1, 3), 1),
            'fg_pct': round(random.uniform(40, 60), 1),
            'three_pt_pct': round(random.uniform(30, 45), 1),
            'ft_pct': round(random.uniform(70, 95), 1),
            'minutes_per_game': round(random.uniform(15, 38), 1),
            'PER': round(random.uniform(10, 30), 1),
            'true_shooting': round(random.uniform(50, 65), 1),
            'usage_rate': round(random.uniform(15, 35), 1),
            'win_shares': round(random.uniform(1, 15), 1)
        }
        return metrics

    def _scrape_basketball_reference(self, player_name: str) -> Dict[str, Any]:
        """Scrape player data from Basketball Reference."""
        # Simulated data similar to NBA but with some additional metrics
        metrics = {
            'team': f"Team {random.choice(['A', 'B', 'C', 'D', 'E'])}",
            'age': random.randint(19, 40),
            'games_played': random.randint(20, 82),
            'games_started': random.randint(0, 82),
            'mp_per_game': round(random.uniform(10, 40), 1),
            'VORP': round(random.uniform(-1, 8), 1),
            'BPM': round(random.uniform(-5, 10), 1),
            'offensive_rating': random.randint(90, 125),
            'defensive_rating': random.randint(90, 125),
            'assist_pct': round(random.uniform(5, 50), 1),
            'turnover_pct': round(random.uniform(5, 20), 1),
            'usg_pct': round(random.uniform(10, 35), 1)
        }
        return metrics

    def _scrape_tennis_abstract(self, player_name: str) -> Dict[str, Any]:
        """Scrape player data from Tennis Abstract."""
        # Simulated tennis data
        metrics = {
            'rank': random.randint(1, 200),
            'matches_played': random.randint(20, 80),
            'matches_won': random.randint(10, 70),
            'serve_rating': round(random.uniform(100, 300), 1),
            'return_rating': round(random.uniform(100, 300), 1),
            'first_serve_pct': round(random.uniform(50, 75), 1),
            'first_serve_pts_won_pct': round(random.uniform(65, 85), 1),
            'second_serve_pts_won_pct': round(random.uniform(40, 60), 1),
            'return_pts_won_pct': round(random.uniform(25, 45), 1),
            'break_pts_converted_pct': round(random.uniform(30, 50), 1),
            'break_pts_saved_pct': round(random.uniform(50, 75), 1),
            'hold_pct': round(random.uniform(60, 95), 1),
            'surface_win_percentages': {
                'hard': round(random.uniform(40, 80), 1),
                'clay': round(random.uniform(40, 80), 1),
                'grass': round(random.uniform(40, 80), 1)
            }
        }
        return metrics

    def _scrape_atp(self, player_name: str) -> Dict[str, Any]:
        """Scrape player data from ATP website."""
        # Simulated ATP-specific tennis data
        return self._generate_tennis_tour_data('atp')

    def _scrape_wta(self, player_name: str) -> Dict[str, Any]:
        """Scrape player data from WTA website."""
        # Simulated WTA-specific tennis data
        return self._generate_tennis_tour_data('wta')

    def _generate_tennis_tour_data(self, tour: str) -> Dict[str, Any]:
        """Generate tennis tour data (ATP or WTA)."""
        metrics = {
            'rank': random.randint(1, 200),
            'ranking_points': random.randint(500, 10000),
            'age': random.randint(18, 38),
            'height_cm': random.randint(165, 200),
            'weight_kg': random.randint(55, 95),
            'turned_pro': random.randint(2000, 2023),
            'career_titles': random.randint(0, 20),
            'YTD_win_loss': f"{random.randint(5, 50)}-{random.randint(5, 30)}",
            'career_win_loss': f"{random.randint(50, 800)}-{random.randint(50, 400)}",
            'current_streak': random.choice(['W', 'L']) + str(random.randint(1, 10)),
            'serve_speed_kmh': random.randint(160, 230)
        }

        # Add grand slam results
        slam_results = {}
        for slam in ['australian_open', 'french_open', 'wimbledon', 'us_open']:
            # Random result from R1 (first round) to W (winner)
            possible_results = ['R1', 'R2', 'R3', 'R4', 'QF', 'SF', 'F', 'W']
            slam_results[slam] = random.choice(possible_results)

        metrics['best_grand_slam_results'] = slam_results

        return metrics

    def _scrape_nfl(self, player_name: str) -> Dict[str, Any]:
        """Scrape player data from NFL website."""
        # Simulated NFL data
        position = random.choice(['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'K', 'P'])

        # Base metrics
        metrics = {
            'team': f"Team {random.choice(['A', 'B', 'C', 'D', 'E'])}",
            'position': position,
            'height_cm': random.randint(175, 205),
            'weight_kg': random.randint(75, 150),
            'age': random.randint(21, 38),
            'experience_years': random.randint(1, 15),
            'games_played': random.randint(1, 17),
            'games_started': random.randint(0, 17)
        }

        # Position-specific metrics
        if position == 'QB':
            metrics.update({
                'pass_attempts': random.randint(100, 600),
                'completions': random.randint(50, 400),
                'completion_pct': round(random.uniform(55, 72), 1),
                'passing_yards': random.randint(500, 5000),
                'passing_tds': random.randint(5, 50),
                'interceptions': random.randint(0, 20),
                'passer_rating': round(random.uniform(70, 120), 1),
                'qbr': round(random.uniform(40, 80), 1),
                'sacks': random.randint(5, 50)
            })
        elif position in ['RB', 'WR', 'TE']:
            metrics.update({
                'targets': random.randint(10, 150),
                'receptions': random.randint(5, 120),
                'receiving_yards': random.randint(50, 1500),
                'receiving_tds': random.randint(0, 15),
                'yards_per_reception': round(random.uniform(8, 20), 1),
                'rushes': random.randint(0, 300),
                'rushing_yards': random.randint(0, 2000),
                'rushing_tds': random.randint(0, 20),
                'yards_per_carry': round(random.uniform(3, 6), 1),
                'scrimmage_yards': random.randint(100, 2500)
            })
        elif position in ['DL', 'LB', 'DB']:
            metrics.update({
                'tackles': random.randint(10, 150),
                'sacks': random.randint(0, 20),
                'forced_fumbles': random.randint(0, 5),
                'fumble_recoveries': random.randint(0, 5),
                'interceptions': random.randint(0, 10),
                'passes_defended': random.randint(0, 20),
                'tackles_for_loss': random.randint(0, 25)
            })

        return metrics

    def _scrape_pro_football_reference(self, player_name: str) -> Dict[str, Any]:
        """Scrape player data from Pro Football Reference."""
        # Similar to NFL but with some additional advanced metrics
        # This would be an implementation similar to the NFL scraper above
        metrics = {
            'approximate_value': random.randint(0, 20),
            'pro_bowls': random.randint(0, 10),
            'all_pros': random.randint(0, 5),
            'games_played': random.randint(1, 250),
            'games_started': random.randint(0, 200),
            'career_win_share': round(random.uniform(0, 100), 1)
        }
        return metrics


# Create a singleton instance
athlete_scraper = AthleteMetricsScraper()

# Example usage
if __name__ == "__main__":
    # Test with some example athletes
    soccer_player = athlete_scraper.scrape_athlete("Lionel Messi", "soccer", ["fbref", "transfermarkt"])

    if soccer_player:
        print(f"Scraped data for {soccer_player.get('name')}:")
        print(f"Sources: {soccer_player.get('data_sources')}")
        print(f"Metrics: {len(soccer_player.get('metrics', {}))} metrics found")

        # Save to player directory
        player_file = PLAYERS_DIR / f"{soccer_player.get('id')}.json"
        with open(player_file, 'w') as f:
            json.dump(soccer_player, f, indent=2)
        print(f"Saved player data to {player_file}")

    # Test wearable data
    wearable_data = athlete_scraper.fetch_wearable_data("messi_soccer", "whoop")
    print(f"Wearable data: {wearable_data.get('metrics')}")

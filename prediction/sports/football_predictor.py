# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FootballPredictor:
    """Football prediction model integrated with Super AI decision making."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.results_data = []
        self.team_stats = {}
        self.form_window = 10  # Number of matches to consider for form
        
    def load_historical_data(self, data_path: str):
        """Load historical match data."""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        home_team, away_team, home_score, away_score = parts[:4]
                        self.results_data.append({
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_score': int(home_score),
                            'away_score': int(away_score)
                        })
            self._update_team_stats()
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            
    def _update_team_stats(self):
        """Update team statistics based on historical data."""
        self.team_stats = {}
        
        for match in self.results_data:
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Initialize team stats if not exists
            for team in [home_team, away_team]:
                if team not in self.team_stats:
                    self.team_stats[team] = {
                        'matches_played': 0,
                        'goals_scored': 0,
                        'goals_conceded': 0,
                        'wins': 0,
                        'draws': 0,
                        'losses': 0,
                        'home_matches': 0,
                        'away_matches': 0,
                        'recent_form': []  # List of results (1=win, 0=draw, -1=loss)
                    }
            
            # Update home team stats
            self.team_stats[home_team]['matches_played'] += 1
            self.team_stats[home_team]['home_matches'] += 1
            self.team_stats[home_team]['goals_scored'] += match['home_score']
            self.team_stats[home_team]['goals_conceded'] += match['away_score']
            
            # Update away team stats
            self.team_stats[away_team]['matches_played'] += 1
            self.team_stats[away_team]['away_matches'] += 1
            self.team_stats[away_team]['goals_scored'] += match['away_score']
            self.team_stats[away_team]['goals_conceded'] += match['home_score']
            
            # Update match results
            if match['home_score'] > match['away_score']:
                self.team_stats[home_team]['wins'] += 1
                self.team_stats[away_team]['losses'] += 1
                self.team_stats[home_team]['recent_form'].append(1)
                self.team_stats[away_team]['recent_form'].append(-1)
            elif match['home_score'] < match['away_score']:
                self.team_stats[home_team]['losses'] += 1
                self.team_stats[away_team]['wins'] += 1
                self.team_stats[home_team]['recent_form'].append(-1)
                self.team_stats[away_team]['recent_form'].append(1)
            else:
                self.team_stats[home_team]['draws'] += 1
                self.team_stats[away_team]['draws'] += 1
                self.team_stats[home_team]['recent_form'].append(0)
                self.team_stats[away_team]['recent_form'].append(0)
            
            # Keep only recent form
            for team in [home_team, away_team]:
                self.team_stats[team]['recent_form'] = self.team_stats[team]['recent_form'][-self.form_window:]
    
    def _calculate_team_features(self, team: str, is_home: bool) -> List[float]:
        """Calculate feature vector for a team."""
        stats = self.team_stats.get(team, {})
        if not stats:
            return [0.0] * 8  # Return zero features if team not found
            
        matches_played = max(1, stats['matches_played'])  # Avoid division by zero
        
        features = [
            stats['goals_scored'] / matches_played,  # Average goals scored
            stats['goals_conceded'] / matches_played,  # Average goals conceded
            stats['wins'] / matches_played,  # Win ratio
            stats['draws'] / matches_played,  # Draw ratio
            stats['losses'] / matches_played,  # Loss ratio
            sum(stats['recent_form'][-5:]) / 5 if stats['recent_form'] else 0,  # Recent form
            stats['home_matches'] / matches_played if is_home else stats['away_matches'] / matches_played,  # Home/Away ratio
            1.0 if is_home else 0.0  # Home advantage indicator
        ]
        
        return features
    
    def predict_match(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Predict the outcome of a match between two teams."""
        # Calculate team features
        home_features = self._calculate_team_features(home_team, is_home=True)
        away_features = self._calculate_team_features(away_team, is_home=False)
        
        # Combine features
        match_features = home_features + away_features
        
        # Simple prediction model based on weighted features
        home_strength = sum(f * w for f, w in zip(home_features, [1.2, -0.8, 1.5, 0.5, -0.8, 1.0, 1.2, 1.3]))
        away_strength = sum(f * w for f, w in zip(away_features, [1.0, -1.0, 1.2, 0.5, -0.8, 0.8, 1.0, 0.0]))
        
        # Calculate probabilities
        total_strength = home_strength + away_strength
        if total_strength == 0:
            home_win_prob = 0.33
            draw_prob = 0.34
            away_win_prob = 0.33
        else:
            home_win_prob = max(0.1, min(0.8, home_strength / total_strength))
            away_win_prob = max(0.1, min(0.8, away_strength / total_strength))
            draw_prob = 1.0 - (home_win_prob + away_win_prob)
        
        # Predict scores
        home_goals = max(0, round(home_strength * 1.5))
        away_goals = max(0, round(away_strength * 1.2))
        
        return {
            'home_win_probability': home_win_prob,
            'draw_probability': draw_prob,
            'away_win_probability': away_win_prob,
            'predicted_home_goals': home_goals,
            'predicted_away_goals': away_goals,
            'home_team_strength': home_strength,
            'away_team_strength': away_strength
        }
    
    def get_team_form_analysis(self, team: str) -> Dict[str, any]:
        """Get detailed form analysis for a team."""
        stats = self.team_stats.get(team, {})
        if not stats:
            return {'error': f'No data available for team: {team}'}
            
        matches_played = max(1, stats['matches_played'])
        
        return {
            'overall_win_rate': stats['wins'] / matches_played,
            'recent_form': stats['recent_form'][-5:],
            'goals_per_game': stats['goals_scored'] / matches_played,
            'goals_conceded_per_game': stats['goals_conceded'] / matches_played,
            'home_performance': {
                'matches': stats['home_matches'],
                'ratio': stats['home_matches'] / matches_played
            },
            'away_performance': {
                'matches': stats['away_matches'],
                'ratio': stats['away_matches'] / matches_played
            }
        }
        
    def get_head_to_head(self, team1: str, team2: str) -> List[Dict]:
        """Get head-to-head history between two teams."""
        h2h_matches = []
        
        for match in self.results_data:
            if (match['home_team'] == team1 and match['away_team'] == team2) or \
               (match['home_team'] == team2 and match['away_team'] == team1):
                h2h_matches.append(match)
        
        return h2h_matches[-5:]  # Return last 5 meetings 